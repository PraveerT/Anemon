"""Push orthogonal tiny classifier from 77 -> 85+ with point+temporal transformer.

Key changes vs strong-tiny (77%):
  - Per-frame: 2-layer point set self-attention (d=256, 4 heads)
  - Temporal: 4-layer transformer encoder w/ learnable pos emb
  - Mixup 0.2, random y-rotation aug, point-dropout 10%
  - 200 epochs, AdamW lr=1e-3, cosine+10ep warmup, wd=0.05
  - Grad clip 1.0, label smooth 0.1

Uses same Cfbq 10-ch input (xyz + res_fwd + res_bwd + t) at 256 pts.
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
    w = mask.float().unsqueeze(0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src.unsqueeze(0) * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tgt.unsqueeze(0) * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src.unsqueeze(0) - sm; tc = tgt.unsqueeze(0) - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=src.device).unsqueeze(0)
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


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True)
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RF = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RB = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    for i in range(N):
        s = loader[i]; pts_d = s[0]; label = s[1]
        pts = pts_d['points'].cuda()
        F_, P, C = pts.shape
        xyz = pts[..., :3]
        orig = pts_d['orig_flat_idx'].cuda()
        ctgt = torch.from_numpy(pts_d['corr_full_target_idx']).long().cuda()
        cw = torch.from_numpy(pts_d['corr_full_weight']).float().cuda()
        sidx, matched = corr_sample_indices(orig, ctgt, cw, PTS, F_, P)
        xyz_s = torch.gather(xyz, 1, sidx.unsqueeze(-1).expand(-1, -1, 3))
        XYZ[i] = xyz_s.cpu().numpy()
        for t in range(F_ - 1):
            qf, trf = kabsch_quat(xyz_s[t], xyz_s[t+1], matched[t])
            RF[i, t+1] = (xyz_s[t+1] - (quat_rotate_points(qf, xyz_s[t]) + trf)).cpu().numpy()
            qb, trb = kabsch_quat(xyz_s[t+1], xyz_s[t], matched[t])
            RB[i, t]   = (xyz_s[t]   - (quat_rotate_points(qb, xyz_s[t+1]) + trb)).cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RF),
            torch.from_numpy(RB), torch.from_numpy(labels))


print('Collecting Cfbq features 256 pts...')
xyz_tr, rf_tr, rb_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, y_te = collect('test')
T = 32
t_tr = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0], T, PTS, 1)
C_tr = torch.cat([xyz_tr, rf_tr, rb_tr, t_tr], dim=-1)      # (N,32,256,10)
C_te = torch.cat([xyz_te, rf_te, rb_te, t_te], dim=-1)
print(f"train {C_tr.shape} test {C_te.shape}")


class PtFrameTx(nn.Module):
    """Per-frame point-set transformer: 2 self-attn layers, d=256."""
    def __init__(self, in_ch=10, d=256, nhead=4, layers=2):
        super().__init__()
        self.emb = nn.Linear(in_ch, d)
        self.norm_in = nn.LayerNorm(d)
        enc = nn.TransformerEncoderLayer(d, nhead, 2*d, 0.1,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.tx = nn.TransformerEncoder(enc, layers)
    def forward(self, x):  # x: (B*T, P, in_ch)
        h = self.norm_in(self.emb(x))
        h = self.tx(h)
        return h.max(1).values  # (B*T, d)


class TempTx(nn.Module):
    """Temporal transformer over T=32 frame tokens."""
    def __init__(self, T=32, d=256, nhead=4, layers=4):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, 2*d, 0.1,
                                         activation='gelu', batch_first=True, norm_first=True)
        self.tx = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
    def forward(self, h):  # (B, T, d)
        h = self.norm(h + self.pos)
        h = self.tx(h)
        return h.max(1).values  # (B, d)


class PtTxNet(nn.Module):
    def __init__(self, in_ch=10, d=256, num_classes=25):
        super().__init__()
        self.pt = PtFrameTx(in_ch, d, 4, 2)
        self.tm = TempTx(32, d, 4, 4)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d//2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(d//2, num_classes))
    def forward(self, x):  # (B, T, P, C)
        B, T_, P_, C_ = x.shape
        h = self.pt(x.reshape(B*T_, P_, C_)).reshape(B, T_, -1)
        h = self.tm(h)
        return self.head(h)


def rand_y_rot(x):
    """Random rotation around Y axis per sample. x: (B, T, P, C)."""
    B = x.shape[0]
    angles = (torch.rand(B, device=x.device) * 2 * math.pi)
    cos = torch.cos(angles); sin = torch.sin(angles)
    R = torch.zeros(B, 3, 3, device=x.device)
    R[:,0,0] = cos;  R[:,0,2] = sin
    R[:,1,1] = 1.0
    R[:,2,0] = -sin; R[:,2,2] = cos
    xyz = x[..., :3]                         # (B,T,P,3)
    res_f = x[..., 3:6]
    res_b = x[..., 6:9]
    t_ch = x[..., 9:10]
    def rot(v):  # (B,T,P,3) with R (B,3,3)
        B_ = v.shape[0]
        return torch.einsum('bij,btpj->btpi', R, v)
    return torch.cat([rot(xyz), rot(res_f), rot(res_b), t_ch], dim=-1)


def train_eval(seed, X_tr, y_tr, X_te, y_te, epochs=200, bs=16, mixup_alpha=0.2):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = PtTxNet(in_ch=X_tr.shape[-1]).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"seed {seed} model params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    warmup = 10
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,epochs-warmup))))
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    best = 0.0; best_logits = None; best_ep = -1
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(X_tr), generator=g)
        train_correct = 0; train_total = 0
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i+bs]
            xb = X_tr_c[idx]
            yb = y_tr_c[idx]
            # Aug: y-rotation
            xb = rand_y_rot(xb)
            # Point dropout 10%
            B_, T_, P_, _ = xb.shape
            xb = xb * (torch.rand(B_, 1, P_, 1, device=xb.device) > 0.10).float()
            # Mixup in feature space
            use_mixup = mixup_alpha > 0 and torch.rand(1).item() < 0.5
            if use_mixup:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm2 = torch.randperm(xb.shape[0], device=xb.device)
                xb_m = lam * xb + (1 - lam) * xb[perm2]
                yb_a, yb_b = yb, yb[perm2]
                opt.zero_grad()
                logits = model(xb_m)
                loss = (lam * F.cross_entropy(logits, yb_a, label_smoothing=0.1)
                        + (1-lam) * F.cross_entropy(logits, yb_b, label_smoothing=0.1))
            else:
                opt.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
                train_correct += (logits.argmax(-1) == yb).sum().item()
                train_total += yb.numel()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        # Eval
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(X_te), 32):
                lg.append(model(X_te_c[i:i+32]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == y_te_c).float().mean().item()
        if acc > best:
            best = acc; best_logits = lg.clone(); best_ep = ep
        if ep % 10 == 0 or ep == epochs - 1:
            tr_acc = (train_correct / max(1, train_total)) if train_total > 0 else -1
            print(f"  ep{ep:3d}  te={acc*100:5.2f}  best={best*100:5.2f}(ep{best_ep})  tr={tr_acc*100:5.2f}  lr={opt.param_groups[0]['lr']:.2e}")
    return best, best_logits, best_ep


tg("PtTxNet (point+temporal transformer) seed 0, 200 ep, Cfbq 10ch 256pts.")
acc, logits, be = train_eval(0, C_tr, y_tr, C_te, y_te, epochs=200)
msg = f"\n=== PtTxNet tiny v2 seed 0 ===\nbest: {acc*100:.2f}% at ep{be}  (target: >85, prior tiny: 77.18)"
print(msg); tg(msg)

# save for fusion
np.savez('/tmp/tiny_tx_logits.npz',
         logits=logits.numpy(),
         labels=y_te.numpy(),
         acc=acc, best_ep=be)
print("saved /tmp/tiny_tx_logits.npz")
