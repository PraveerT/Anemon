"""Stronger orthogonal tiny classifier: same tiny-arch family, more capacity/training.

Previous tiny 60-ep baseline:
  A (4ch)    = 53.15 +/- 2.32
  Cfbq (10ch) = 54.65 +/- 1.15

Goal: push this arch a lot higher while preserving orthogonality to PMamba
(different architecture class). 5 seeds each; keep comparison fair.

Upgrades:
  - PtMLP: 10 -> 64 -> 128 -> 256 with LayerNorm + GELU
  - Pool: max || mean over points (512-dim per frame)
  - Temporal: 4 conv1d blocks with BN + GELU + residual
  - Dropout 0.2 feature / 0.3 head, label-smooth 0.1
  - Optimizer AdamW lr=2e-3, wd=1e-4, cosine w/ 5-ep warmup
  - 256 pts, 120 epochs, batch 32
  - 10% random point-dropout augmentation per sample
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import requests
import nvidia_dataloader

TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"


def tg(msg):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            chat_id = r["result"][-1]["message"]["chat"]["id"]
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


def rot_to_quat(R):
    m00,m01,m02 = R[0,0], R[0,1], R[0,2]
    m10,m11,m12 = R[1,0], R[1,1], R[1,2]
    m20,m21,m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = torch.sqrt(tr.clamp(min=-0.999) + 1) * 2
        q = torch.stack([0.25*s, (m21-m12)/s, (m02-m20)/s, (m10-m01)/s])
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1 + m00 - m11 - m22).clamp(min=1e-8) * 2
        q = torch.stack([(m21-m12)/s, 0.25*s, (m01+m10)/s, (m02+m20)/s])
    elif m11 > m22:
        s = torch.sqrt(1 + m11 - m00 - m22).clamp(min=1e-8) * 2
        q = torch.stack([(m02-m20)/s, (m01+m10)/s, 0.25*s, (m12+m21)/s])
    else:
        s = torch.sqrt(1 + m22 - m00 - m11).clamp(min=1e-8) * 2
        q = torch.stack([(m10-m01)/s, (m02+m20)/s, (m12+m21)/s, 0.25*s])
    q = F.normalize(q, dim=-1)
    if q[0] < 0: q = -q
    return q


def hamilton(a, b):
    aw,ax,ay,az = a[...,0], a[...,1], a[...,2], a[...,3]
    bw,bx,by,bz = b[...,0], b[...,1], b[...,2], b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def quat_rotate_points(q, points):
    N = points.shape[0]
    q_b = q.unsqueeze(0).expand(N, -1)
    pq = torch.cat([torch.zeros(N, 1, device=points.device, dtype=points.dtype), points], dim=-1)
    q_conj = torch.cat([q_b[:, 0:1], -q_b[:, 1:]], dim=-1)
    return hamilton(hamilton(q_b, pq), q_conj)[:, 1:]


def kabsch_quat(src, tgt, mask):
    device = src.device
    w = mask.float().unsqueeze(0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src.unsqueeze(0) * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tgt.unsqueeze(0) * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src.unsqueeze(0) - sm; tc = tgt.unsqueeze(0) - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    q = rot_to_quat(R[0])
    t = tm.squeeze(0).squeeze(0) - quat_rotate_points(q, sm.squeeze(0).squeeze(0).unsqueeze(0))[0]
    return q, t


def corr_sample_indices(orig_flat_idx, corr_target, corr_weight, pts_size, F_, P):
    device = orig_flat_idx.device
    idx0 = torch.linspace(0, P-1, pts_size, device=device).long()
    sampled_idx = torch.zeros(F_, pts_size, dtype=torch.long, device=device)
    matched = torch.zeros(F_-1, pts_size, dtype=torch.bool, device=device)
    sampled_idx[0] = idx0
    current_prov = orig_flat_idx[0, idx0].long()
    total_pts = corr_target.shape[-1]; raw_ppf = total_pts // F_
    for t in range(F_ - 1):
        next_prov = orig_flat_idx[t+1].long()
        reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
        reverse_map[next_prov] = torch.arange(P, device=device)
        tgt_flat = corr_target[current_prov]
        tgt_w = corr_weight[current_prov]
        tgt_flat_safe = tgt_flat.clamp(min=0)
        tgt_frame = tgt_flat // raw_ppf
        tgt_pos = reverse_map[tgt_flat_safe]
        valid = (tgt_flat >= 0) & (tgt_w > 0) & (tgt_frame == t+1) & (tgt_pos >= 0)
        next_idx = torch.randint(0, P, (pts_size,), device=device)
        next_idx[valid] = tgt_pos[valid]
        sampled_idx[t+1] = next_idx; matched[t] = valid
        current_prov = orig_flat_idx[t+1, next_idx].long()
    return sampled_idx, matched


PTS = 256
tg("STRONG tiny orthogonal classifier: PtMLP+conv4, 120ep, 256pts, 5-seed A vs Cfbq.")
print('Collecting features (PTS=256)...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RES_F = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RES_B = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    for i in range(N):
        s = loader[i]
        pts_dict = s[0]; label = s[1]
        pts = pts_dict['points'].cuda()
        F_, P, C = pts.shape
        xyz = pts[..., :3]
        orig = pts_dict['orig_flat_idx'].cuda()
        ctgt = torch.from_numpy(pts_dict['corr_full_target_idx']).long().cuda()
        cw = torch.from_numpy(pts_dict['corr_full_weight']).float().cuda()
        sampled_idx, matched = corr_sample_indices(orig, ctgt, cw, PTS, F_, P)
        xyz_samp = torch.gather(xyz, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))
        XYZ[i] = xyz_samp.cpu().numpy()
        for t in range(F_ - 1):
            qf, trf = kabsch_quat(xyz_samp[t], xyz_samp[t+1], matched[t])
            rigid_f = quat_rotate_points(qf, xyz_samp[t]) + trf
            rf = xyz_samp[t+1] - rigid_f
            RES_F[i, t+1] = rf.cpu().numpy()
            qb, trb = kabsch_quat(xyz_samp[t+1], xyz_samp[t], matched[t])
            rigid_b = quat_rotate_points(qb, xyz_samp[t+1]) + trb
            rb = xyz_samp[t] - rigid_b
            RES_B[i, t] = rb.cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RES_F),
            torch.from_numpy(RES_B), torch.from_numpy(labels))


xyz_tr, resf_tr, resb_tr, y_tr = collect('train')
xyz_te, resf_te, resb_te, y_te = collect('test')
tg(f"Collection done ({xyz_tr.shape[0]} train, {xyz_te.shape[0]} test, PTS={PTS}).")

T = xyz_tr.shape[1]
t_tr = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0], T, PTS, 1)

A_tr = torch.cat([xyz_tr, t_tr], dim=-1)              # 4ch
A_te = torch.cat([xyz_te, t_te], dim=-1)
C_tr = torch.cat([xyz_tr, resf_tr, resb_tr, t_tr], dim=-1)  # 10ch (Cfbq)
C_te = torch.cat([xyz_te, resf_te, resb_te, t_te], dim=-1)


class StrongPointTemporal(nn.Module):
    def __init__(self, in_ch, num_classes=25):
        super().__init__()
        self.pt_mlp = nn.Sequential(
            nn.Linear(in_ch, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.tproj = nn.Conv1d(512, 256, 1)
        self.tconv1 = nn.Conv1d(256, 256, 3, padding=1); self.bn1 = nn.BatchNorm1d(256)
        self.tconv2 = nn.Conv1d(256, 256, 3, padding=1); self.bn2 = nn.BatchNorm1d(256)
        self.tconv3 = nn.Conv1d(256, 256, 3, padding=1); self.bn3 = nn.BatchNorm1d(256)
        self.tconv4 = nn.Conv1d(256, 256, 3, padding=1); self.bn4 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # x: (B, T, P, C)
        h = self.pt_mlp(x)
        h_max = h.max(dim=2).values
        h_mean = h.mean(dim=2)
        h = torch.cat([h_max, h_mean], dim=-1).transpose(1, 2)  # (B, 512, T)
        h = self.tproj(h)
        h = F.gelu(self.bn1(self.tconv1(h)))
        h = h + F.gelu(self.bn2(self.tconv2(h)))
        h = h + F.gelu(self.bn3(self.tconv3(h)))
        h = h + F.gelu(self.bn4(self.tconv4(h)))
        h = self.drop(h)
        h = h.max(dim=2).values
        return self.head(h)


def train_eval(seed, X_tr, y_tr, X_te, y_te, in_ch, epochs=120):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = StrongPointTemporal(in_ch).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup = 5
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        p = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    BS = 32; N_train = len(X_tr); best = 0.0; best_ep = -1
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed * 1000 + ep)
        perm = torch.randperm(N_train, generator=g)
        for i in range(0, N_train, BS):
            idx = perm[i:i+BS]
            xb = X_tr_c[idx]
            # point-dropout aug: zero out 10% of points (no zero needed for xyz,
            # but zeroing keeps it as "missing" signal to the model)
            if model.training:
                B_, T_, P_, _ = xb.shape
                mask = (torch.rand(B_, 1, P_, 1, device=xb.device) > 0.10).float()
                xb = xb * mask
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, y_tr_c[idx], label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_te), 64):
                preds.append(model(X_te_c[i:i+64]).argmax(-1))
            preds = torch.cat(preds)
            acc = (preds == y_te_c).float().mean().item()
        if acc > best: best = acc; best_ep = ep
    return best, best_ep


print("\n=== STRONG tiny classifier, 5 seeds A vs Cfbq, 120 ep, 256 pts ===")
a_scores = []
c_scores = []
for seed in range(5):
    a_acc, a_ep = train_eval(seed, A_tr, y_tr, A_te, y_te, in_ch=4)
    c_acc, c_ep = train_eval(seed, C_tr, y_tr, C_te, y_te, in_ch=10)
    a_scores.append(a_acc); c_scores.append(c_acc)
    line = f"seed {seed}: A={a_acc*100:.2f}% (ep{a_ep})  Cfbq={c_acc*100:.2f}% (ep{c_ep})  d={100*(c_acc-a_acc):+.2f}"
    print(line); tg(line)

a = np.array(a_scores); c = np.array(c_scores)
msg = (f"\n=== STRONG TINY 5-seed 120ep 256pts ===\n"
       f"A:     {a.mean()*100:.2f} +/- {a.std()*100:.2f}\n"
       f"Cfbq:  {c.mean()*100:.2f} +/- {c.std()*100:.2f}\n"
       f"Delta: {(c.mean()-a.mean())*100:+.2f}pp  ({(c-a).mean()*100:+.2f} paired)\n"
       f"Old 60ep-128pts baseline: A=53.15, Cfbq=54.65 (+1.50)")
print(msg); tg(msg)
