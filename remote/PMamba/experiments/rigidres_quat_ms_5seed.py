"""5-seed multi-scale QCC subtraction with EXPLICIT quaternion rotation.

All rigid predictions via quaternion sandwich: p_rigid = q·p·q* + t.
Where q = rot_to_quat(Kabsch(src, tgt)).

Variants:
  A:         [xyz, t]                            4-ch
  B_q:       [xyz, res_1, t]                     7-ch  (single-scale)
  Cfb_q:     [xyz, res_fwd, res_bwd, t]         10-ch  (bidirectional)
  Dms_q:     [xyz, res_1, res_2, res_4, t]      13-ch  (multi-scale)

Winner becomes the feature set for full PMamba training.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    """Rotate points (N,3) by unit quaternion q (4,)."""
    N = points.shape[0]
    q_b = q.unsqueeze(0).expand(N, -1)
    pq = torch.cat([torch.zeros(N, 1, device=points.device, dtype=points.dtype), points], dim=-1)
    q_conj = torch.cat([q_b[:, 0:1], -q_b[:, 1:]], dim=-1)
    return hamilton(hamilton(q_b, pq), q_conj)[:, 1:]


def kabsch_quat(src, tgt, mask):
    """Kabsch returning q and t in pure-quaternion form."""
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
    sm_v = sm.squeeze(0).squeeze(0)
    tm_v = tm.squeeze(0).squeeze(0)
    t = tm_v - quat_rotate_points(q, sm_v.unsqueeze(0))[0]
    return q, t


def quat_residual(src, tgt, mask):
    """res = tgt - (q·src·q* + t), per-point (N,3)."""
    q, t = kabsch_quat(src, tgt, mask)
    rigid = quat_rotate_points(q, src) + t
    return tgt - rigid


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


PTS = 128
tg("5-SEED multi-scale QCC SUBTRACTION (literal quaternion). A/B/Cfb/Dms.")
print('Collecting...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    R1_F = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    R1_B = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    R2 = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    R4 = np.zeros((N, 32, PTS, 3), dtype=np.float32)
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
            res_f = quat_residual(xyz_samp[t], xyz_samp[t+1], matched[t])
            R1_F[i, t+1] = res_f.cpu().numpy()
            res_b = quat_residual(xyz_samp[t+1], xyz_samp[t], matched[t])
            R1_B[i, t] = res_b.cpu().numpy()
        for t in range(2, F_):
            mchain = matched[t-2] & matched[t-1]
            res_2 = quat_residual(xyz_samp[t-2], xyz_samp[t], mchain)
            R2[i, t] = res_2.cpu().numpy()
        for t in range(4, F_):
            mchain = matched[t-4] & matched[t-3] & matched[t-2] & matched[t-1]
            res_4 = quat_residual(xyz_samp[t-4], xyz_samp[t], mchain)
            R4[i, t] = res_4.cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(R1_F), torch.from_numpy(R1_B),
            torch.from_numpy(R2), torch.from_numpy(R4), torch.from_numpy(labels))


xyz_tr, r1f_tr, r1b_tr, r2_tr, r4_tr, y_tr = collect('train')
xyz_te, r1f_te, r1b_te, r2_te, r4_te, y_te = collect('test')
tg("Collection done. Training A/Bq/Cfbq/Dmsq × 5 seeds.")

T = xyz_tr.shape[1]
t_tr = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0],  T, PTS, 1)

A_tr = torch.cat([xyz_tr, t_tr], dim=-1)
A_te = torch.cat([xyz_te, t_te], dim=-1)
Bq_tr = torch.cat([xyz_tr, r1f_tr, t_tr], dim=-1)
Bq_te = torch.cat([xyz_te, r1f_te, t_te], dim=-1)
Cq_tr = torch.cat([xyz_tr, r1f_tr, r1b_tr, t_tr], dim=-1)
Cq_te = torch.cat([xyz_te, r1f_te, r1b_te, t_te], dim=-1)
Dq_tr = torch.cat([xyz_tr, r1f_tr, r2_tr, r4_tr, t_tr], dim=-1)
Dq_te = torch.cat([xyz_te, r1f_te, r2_te, r4_te, t_te], dim=-1)


class TinyPointTemporal(nn.Module):
    def __init__(self, in_ch, num_classes=25, per_point=64, temporal=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_ch, per_point), nn.GELU(),
                                 nn.Linear(per_point, per_point))
        self.conv1 = nn.Conv1d(per_point, temporal, 3, padding=1)
        self.conv2 = nn.Conv1d(temporal, temporal, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(temporal, num_classes)
    def forward(self, x):
        h = self.mlp(x); h = h.max(dim=2).values; h = h.transpose(1, 2)
        h = F.gelu(self.conv1(h)); h = F.gelu(self.conv2(h))
        return self.fc(self.pool(h).squeeze(-1))


def train_eval(seed, X_tr, y_tr, X_te, y_te, in_ch, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = TinyPointTemporal(in_ch).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    BS = 32; N_train = len(X_tr); best = 0
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed * 1000 + ep)
        perm = torch.randperm(N_train, generator=g)
        for i in range(0, N_train, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            loss = F.cross_entropy(model(X_tr_c[idx]), y_tr_c[idx])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_te), 64):
                preds.append(model(X_te_c[i:i+64]).argmax(-1))
            preds = torch.cat(preds)
            acc = (preds == y_te_c).float().mean().item()
        if acc > best: best = acc
    return best


rows = []
for seed in range(5):
    a = train_eval(seed, A_tr, y_tr, A_te, y_te, 4)
    b = train_eval(seed, Bq_tr, y_tr, Bq_te, y_te, 7)
    c = train_eval(seed, Cq_tr, y_tr, Cq_te, y_te, 10)
    d = train_eval(seed, Dq_tr, y_tr, Dq_te, y_te, 13)
    msg = f"seed {seed}: A={a*100:.2f} Bq={b*100:.2f} Cfbq={c*100:.2f} Dmsq={d*100:.2f}"
    rows.append((a, b, c, d)); print(msg); tg(msg)

A_ = [r[0] for r in rows]; B_ = [r[1] for r in rows]
C_ = [r[2] for r in rows]; D_ = [r[3] for r in rows]

summary = f"""
=== QUATERNION-LITERAL multi-scale QCC 5 seeds ===
A:     {np.mean(A_)*100:.2f} ± {np.std(A_)*100:.2f}
Bq:    {np.mean(B_)*100:.2f} ± {np.std(B_)*100:.2f}   (Bq-A:   {(np.mean(B_)-np.mean(A_))*100:+.2f})
Cfbq:  {np.mean(C_)*100:.2f} ± {np.std(C_)*100:.2f}   (Cfbq-Bq: {(np.mean(C_)-np.mean(B_))*100:+.2f})
Dmsq:  {np.mean(D_)*100:.2f} ± {np.std(D_)*100:.2f}   (Dmsq-Bq: {(np.mean(D_)-np.mean(B_))*100:+.2f})
Winner: {['A','Bq','Cfbq','Dmsq'][int(np.argmax([np.mean(A_), np.mean(B_), np.mean(C_), np.mean(D_)]))]}
"""
print(summary); tg(summary)
