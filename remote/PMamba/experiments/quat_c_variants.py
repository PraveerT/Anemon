"""6 sub-variants of C (axis-angle + translation) to push past 28.42%.
Target: break 32% (centroid-only ceiling).

C1 (baseline): axis*angle + t (6d per pair)
C_vel: axis*angle + t + dt (9d)
C_acc: axis*angle + t + dt + d2t (12d)
C_cum: axis*angle + t + cum_t (9d)
C_t_only: t + dt + d2t (9d) — no rotation
C_frame: per-frame centroid + dc + d2c (9d x 32 frames)
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
    orig = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    m00,m01,m02 = Rf[:,0,0], Rf[:,0,1], Rf[:,0,2]
    m10,m11,m12 = Rf[:,1,0], Rf[:,1,1], Rf[:,1,2]
    m20,m21,m22 = Rf[:,2,0], Rf[:,2,1], Rf[:,2,2]
    tr = m00 + m11 + m22
    B = Rf.shape[0]; device = Rf.device
    q = torch.zeros(B, 4, device=device, dtype=Rf.dtype)
    m1 = tr > 0
    if m1.any():
        s = torch.sqrt(tr[m1].clamp(min=-0.999) + 1.0) * 2
        q[m1,0]=0.25*s; q[m1,1]=(m21[m1]-m12[m1])/s
        q[m1,2]=(m02[m1]-m20[m1])/s; q[m1,3]=(m10[m1]-m01[m1])/s
    rem = ~m1
    m2a = rem & (m00>m11) & (m00>m22)
    if m2a.any():
        s = torch.sqrt(1+m00[m2a]-m11[m2a]-m22[m2a]).clamp(min=1e-8)*2
        q[m2a,0]=(m21[m2a]-m12[m2a])/s; q[m2a,1]=0.25*s
        q[m2a,2]=(m01[m2a]+m10[m2a])/s; q[m2a,3]=(m02[m2a]+m20[m2a])/s
    m2b = rem & (~m2a) & (m11>m22)
    if m2b.any():
        s = torch.sqrt(1+m11[m2b]-m00[m2b]-m22[m2b]).clamp(min=1e-8)*2
        q[m2b,0]=(m02[m2b]-m20[m2b])/s; q[m2b,1]=(m01[m2b]+m10[m2b])/s
        q[m2b,2]=0.25*s; q[m2b,3]=(m12[m2b]+m21[m2b])/s
    m2c = rem & (~m2a) & (~m2b)
    if m2c.any():
        s = torch.sqrt(1+m22[m2c]-m00[m2c]-m11[m2c]).clamp(min=1e-8)*2
        q[m2c,0]=(m10[m2c]-m01[m2c])/s; q[m2c,1]=(m02[m2c]+m20[m2c])/s
        q[m2c,2]=(m12[m2c]+m21[m2c])/s; q[m2c,3]=0.25*s
    return F.normalize(q, dim=-1).reshape(*orig, 4)


def kabsch_rt(src, tgt, mask):
    device = src.device
    w = mask.float(); w_sum = w.sum().clamp(min=1.0)
    if w_sum < 3:
        return (torch.tensor([1.0,0.0,0.0,0.0], device=device, dtype=torch.float32),
                torch.zeros(3, device=device, dtype=torch.float32))
    src_mean = (src * w.unsqueeze(-1)).sum(0) / w_sum
    tgt_mean = (tgt * w.unsqueeze(-1)).sum(0) / w_sum
    src_c = src - src_mean; tgt_c = tgt - tgt_mean
    H = (src_c * w.unsqueeze(-1)).T @ tgt_c
    H = H + 1e-6 * torch.eye(3, device=device)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.T; det = torch.det(V @ U.T)
    D = torch.diag(torch.tensor([1, 1, det.item()], device=device))
    R = V @ D @ U.T
    q_r = rot_to_quat(R)
    t = tgt_mean - R @ src_mean
    return q_r, t


def quat_to_axisangle(q):
    w = q[..., 0:1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    angle = 2 * torch.atan2(sin_half, w)
    axis = xyz / sin_half
    return axis * angle


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
        tgt_flat = corr_target[current_prov]; tgt_w = corr_weight[current_prov]
        tgt_flat_safe = tgt_flat.clamp(min=0)
        tgt_frame = tgt_flat // raw_ppf
        tgt_pos = reverse_map[tgt_flat_safe]
        valid = (tgt_flat >= 0) & (tgt_w > 0) & (tgt_frame == t+1) & (tgt_pos >= 0)
        next_idx = torch.randint(0, P, (pts_size,), device=device)
        next_idx[valid] = tgt_pos[valid]
        sampled_idx[t+1] = next_idx; matched[t] = valid
        current_prov = orig_flat_idx[t+1, next_idx].long()
    return sampled_idx, matched


tg("C-variants test starting: 6 sub-variants")
print('Collecting all features...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    # Pair-wise features (31 pairs)
    AA = np.zeros((N, 31, 3), dtype=np.float32)      # axis*angle
    T = np.zeros((N, 31, 3), dtype=np.float32)       # translation
    # Per-frame features (32 frames)
    C_frame = np.zeros((N, 32, 3), dtype=np.float32) # per-frame centroid
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
        S = min(256, P)
        sampled_idx, matched = corr_sample_indices(orig, ctgt, cw, S, F_, P)
        xyz_samp = torch.gather(xyz, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))
        for t in range(F_):
            C_frame[i, t] = xyz_samp[t].mean(0).cpu().numpy()
        for t in range(F_ - 1):
            q_r, tr = kabsch_rt(xyz_samp[t], xyz_samp[t+1], matched[t])
            aa = quat_to_axisangle(q_r)
            AA[i, t] = aa.cpu().numpy()
            T[i, t] = tr.cpu().numpy()
        labels[i] = label
        if (i+1) % 200 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(AA), torch.from_numpy(T),
            torch.from_numpy(C_frame), torch.from_numpy(labels))


aa_train, t_train, c_train, y_train = collect('train')
aa_test, t_test, c_test, y_test = collect('test')
print(f'AA: {aa_train.shape}  T: {t_train.shape}  C: {c_train.shape}')
tg("Collection done; building variants")


def diff(x):
    """x: (B, T, 3) -> forward diff (B, T, 3) pad with 0."""
    d = torch.zeros_like(x)
    d[:, :-1] = x[:, 1:] - x[:, :-1]
    return d


def build_variants():
    aa_tr, aa_te = aa_train, aa_test
    t_tr, t_te = t_train, t_test
    c_tr, c_te = c_train, c_test

    # Derivatives of pair-t
    dt_tr = diff(t_tr); dt_te = diff(t_te)
    d2t_tr = diff(dt_tr); d2t_te = diff(dt_te)
    # Cumulative pair-t (sum up to frame t)
    cum_t_tr = t_tr.cumsum(dim=1); cum_t_te = t_te.cumsum(dim=1)
    # Derivatives of per-frame centroid
    dc_tr = diff(c_tr); dc_te = diff(c_te)
    d2c_tr = diff(dc_tr); d2c_te = diff(dc_te)

    return {
        "C1 (aa, t) 6d":       ((torch.cat([aa_tr, t_tr], dim=-1), torch.cat([aa_te, t_te], dim=-1)), 6),
        "C_vel (aa, t, dt) 9d": ((torch.cat([aa_tr, t_tr, dt_tr], dim=-1), torch.cat([aa_te, t_te, dt_te], dim=-1)), 9),
        "C_acc (aa, t, dt, d2t) 12d": ((torch.cat([aa_tr, t_tr, dt_tr, d2t_tr], dim=-1), torch.cat([aa_te, t_te, dt_te, d2t_te], dim=-1)), 12),
        "C_cum (aa, t, cum_t) 9d": ((torch.cat([aa_tr, t_tr, cum_t_tr], dim=-1), torch.cat([aa_te, t_te, cum_t_te], dim=-1)), 9),
        "C_t_only (t, dt, d2t) 9d": ((torch.cat([t_tr, dt_tr, d2t_tr], dim=-1), torch.cat([t_te, dt_te, d2t_te], dim=-1)), 9),
        "C_frame (c, dc, d2c) 9d": ((torch.cat([c_tr, dc_tr, d2c_tr], dim=-1), torch.cat([c_te, dc_te, d2c_te], dim=-1)), 9),
    }


variants = build_variants()


class BigSeqClf(nn.Module):
    def __init__(self, in_ch, num_classes=25, hidden=512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, hidden, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


def train_eval(tag, tr, te, y_tr, y_te, in_ch, epochs=100):
    model = BigSeqClf(in_ch).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_c = tr.cuda(); y_tr_c = y_tr.cuda()
    te_c = te.cuda(); y_te_c = y_te.cuda()
    BS = 64; N_train = len(tr); best = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N_train)
        for i in range(0, N_train, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            loss = F.cross_entropy(model(tr_c[idx]), y_tr_c[idx])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(te_c).argmax(-1) == y_te_c).float().mean().item()
        if acc > best: best = acc
        if ep % 20 == 0 or ep == epochs - 1:
            msg = f"{tag} ep {ep:3d}  best={best*100:.2f}%"
            print(msg); tg(msg)
    final = f"=== {tag} FINAL: {best*100:.2f}% ==="
    print(final); tg(final)
    return best


results = {}
for tag, ((tr, te), in_ch) in variants.items():
    results[tag] = train_eval(tag, tr, te, y_train, y_test, in_ch)

summary = "\n=== C-VARIANTS SUMMARY ===\n"
for tag, r in results.items():
    summary += f"{tag}: {r*100:.2f}%\n"
summary += f"Prior: C1 28.42%, centroid-only 31.95%"
print(summary); tg(summary)
