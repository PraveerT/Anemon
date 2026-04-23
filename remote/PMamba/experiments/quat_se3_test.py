"""Two sanity tests with telegram pushes:
1. Centroid-only: 3d per-frame centroid trajectory -> classify
2. Dual-quaternion (q_r + q_d): full SE(3) per frame-pair -> classify
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


def hamilton(a, b):
    aw,ax,ay,az = a[...,0],a[...,1],a[...,2],a[...,3]
    bw,bx,by,bz = b[...,0],b[...,1],b[...,2],b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def kabsch_rt_masked(src, tgt, mask):
    """Kabsch returning (q_r, t) as 7 elements together."""
    device = src.device
    w = mask.float(); w_sum = w.sum().clamp(min=1.0)
    if w_sum < 3:
        return torch.tensor([1,0,0,0], device=device), torch.zeros(3, device=device)
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


def dual_quat(q_r, t):
    """DQ dual part: q_d = 0.5 * [0, t] * q_r."""
    t_quat = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)
    return 0.5 * hamilton(t_quat, q_r)


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


tg("SE3 test: collecting features (centroid traj + dual quat)")
print('Collecting centroids + dual quats...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    C_traj = np.zeros((N, 32, 3), dtype=np.float32)          # per-frame centroid
    DQ = np.zeros((N, 31, 8), dtype=np.float32)              # dual quat per pair
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
        # Centroid trajectory
        for t in range(F_):
            C_traj[i, t] = xyz_samp[t].mean(0).cpu().numpy()
        # DQ per pair
        for t in range(F_ - 1):
            q_r, trans = kabsch_rt_masked(xyz_samp[t], xyz_samp[t+1], matched[t])
            q_d = dual_quat(q_r, trans)
            DQ[i, t, :4] = q_r.cpu().numpy()
            DQ[i, t, 4:] = q_d.cpu().numpy()
        labels[i] = label
        if (i+1) % 200 == 0:
            print(f'  {phase} {i+1}/{N}')
    return torch.from_numpy(C_traj), torch.from_numpy(DQ), torch.from_numpy(labels)


c_train, dq_train, y_train = collect('train')
c_test, dq_test, y_test = collect('test')
print(f'c traj: {c_train.shape}  dq: {dq_train.shape}')
tg("Collection done; training centroid-only + DQ classifiers")

# Sign-canonicalize q_r for DQ
for dq in [dq_train, dq_test]:
    flip = dq[..., 0] < 0
    dq[flip] = -dq[flip]


class SeqClf(nn.Module):
    def __init__(self, in_ch, num_classes=25, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv1d(256, hidden, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x)); x = F.gelu(self.conv2(x)); x = F.gelu(self.conv3(x))
        return self.fc(self.pool(x).squeeze(-1))


def train_eval(tag, q_tr, y_tr, q_te, y_te, in_ch, epochs=80):
    model = SeqClf(in_ch).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    q_tr_c = q_tr.cuda(); y_tr_c = y_tr.cuda()
    q_te_c = q_te.cuda(); y_te_c = y_te.cuda()
    BS = 64; N_train = len(q_tr); best = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N_train)
        for i in range(0, N_train, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            loss = F.cross_entropy(model(q_tr_c[idx]), y_tr_c[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(q_te_c).argmax(-1) == y_te_c).float().mean().item()
        if acc > best: best = acc
        if ep % 10 == 0 or ep == epochs - 1:
            msg = f"{tag} ep {ep:2d}  test={acc*100:.2f}%  best={best*100:.2f}%"
            print(msg); tg(msg)
    final = f"=== {tag} FINAL BEST: {best*100:.2f}% ==="
    print(final); tg(final)
    return best


print('\n--- CENTROID-ONLY (translation) ---')
b_c = train_eval("Centroid", c_train, y_train, c_test, y_test, in_ch=3, epochs=80)

print('\n--- DUAL QUATERNION (SE3 per pair) ---')
b_d = train_eval("DQ", dq_train, y_train, dq_test, y_test, in_ch=8, epochs=80)

summary = (f"\n=== SE(3) SUMMARY ===\n"
           f"Centroid-only: {b_c*100:.2f}%\n"
           f"Dual quaternion: {b_d*100:.2f}%\n"
           f"Baselines: quat 13.90%, K=2 16.60%, artic 15.56%, random 4%")
print(summary); tg(summary)
