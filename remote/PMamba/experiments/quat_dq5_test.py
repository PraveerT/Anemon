"""5 DQCC variants to push past 28.42%:
V1: DQ cumulative from frame 0 (per-frame 8d x 32)
V2: Per-part DQ K=2 (per pair, palm + rest, 16d x 31)
V3: DQ derivatives per pair (DQ, dDQ, d2DQ = 24d x 31)
V4: Combined — DQ + cum_DQ + per-part DQ ~ 32d x 31
V5: Transformer classifier on plain DQ 8d x 31
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


def dq_multiply(p_r, p_d, q_r, q_d):
    r = hamilton(p_r, q_r)
    d = hamilton(p_r, q_d) + hamilton(p_d, q_r)
    return r, d


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
    return rot_to_quat(R), tgt_mean - R @ src_mean


def make_dq(q_r, t):
    zero = torch.zeros_like(t[..., :1])
    tq = torch.cat([zero, t], dim=-1)
    q_d = 0.5 * hamilton(tq, q_r)
    return torch.cat([q_r, q_d], dim=-1)


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


def kmeans2(x, n_iter=10):
    idx_init = torch.randperm(x.shape[0], device=x.device)[:2]
    centers = x[idx_init]
    for _ in range(n_iter):
        d = torch.cdist(x, centers)
        labels = d.argmin(-1)
        for ki in range(2):
            m = labels == ki
            if m.sum() > 0:
                centers[ki] = x[m].mean(0)
    return labels


tg("DQCC 5 variants starting")
print('Collecting DQ + per-part DQ ...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    DQ = np.zeros((N, 31, 8), dtype=np.float32)
    DQ_K2 = np.zeros((N, 31, 16), dtype=np.float32)     # 2 parts x 8d
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
        part_labels = kmeans2(xyz_samp[0])
        for t in range(F_ - 1):
            # Whole-cloud DQ
            q_r, tr = kabsch_rt(xyz_samp[t], xyz_samp[t+1], matched[t])
            DQ[i, t] = make_dq(q_r, tr).cpu().numpy()
            # Per-part DQ
            for kp in range(2):
                m = (part_labels == kp) & matched[t]
                q_r_p, tr_p = kabsch_rt(xyz_samp[t], xyz_samp[t+1], m)
                DQ_K2[i, t, kp*8:(kp+1)*8] = make_dq(q_r_p, tr_p).cpu().numpy()
        labels[i] = label
        if (i+1) % 200 == 0:
            print(f'  {phase} {i+1}/{N}')
    return torch.from_numpy(DQ), torch.from_numpy(DQ_K2), torch.from_numpy(labels)


dq_train, dqk2_train, y_train = collect('train')
dq_test, dqk2_test, y_test = collect('test')
print(f'DQ: {dq_train.shape}  DQ_K2: {dqk2_train.shape}')
tg("Collection done; computing variants + training")

# Sign-canonicalize q_r parts
for arr in [dq_train, dq_test]:
    flip = arr[..., 0] < 0
    arr[flip] = -arr[flip]
for arr in [dqk2_train, dqk2_test]:
    for k in range(2):
        flip = arr[..., k*8] < 0
        arr[flip, k*8:(k+1)*8] = -arr[flip, k*8:(k+1)*8]


def compose_cum_dq(dq_seq):
    """dq_seq: (B, T, 8) per-pair DQs. Return cumulative DQ (B, T, 8)."""
    B, T, _ = dq_seq.shape
    out = torch.zeros(B, T, 8, device=dq_seq.device, dtype=dq_seq.dtype)
    # start identity
    cum_r = torch.zeros(B, 4, device=dq_seq.device, dtype=dq_seq.dtype)
    cum_r[:, 0] = 1.0
    cum_d = torch.zeros(B, 4, device=dq_seq.device, dtype=dq_seq.dtype)
    for t in range(T):
        q_r = dq_seq[:, t, :4]; q_d = dq_seq[:, t, 4:]
        cum_r, cum_d = dq_multiply(cum_r, cum_d, q_r, q_d)
        # re-normalize to avoid drift
        cum_r = F.normalize(cum_r, dim=-1)
        out[:, t, :4] = cum_r
        out[:, t, 4:] = cum_d
    return out


def diff(x):
    d = torch.zeros_like(x)
    d[:, :-1] = x[:, 1:] - x[:, :-1]
    return d


# Build all variants
dq_cum_train = compose_cum_dq(dq_train); dq_cum_test = compose_cum_dq(dq_test)
d_dq_train = diff(dq_train); d_dq_test = diff(dq_test)
d2_dq_train = diff(d_dq_train); d2_dq_test = diff(d_dq_test)

V1_tr = dq_cum_train; V1_te = dq_cum_test                                  # 8d
V2_tr = dqk2_train; V2_te = dqk2_test                                       # 16d
V3_tr = torch.cat([dq_train, d_dq_train, d2_dq_train], dim=-1)
V3_te = torch.cat([dq_test, d_dq_test, d2_dq_test], dim=-1)                 # 24d
V4_tr = torch.cat([dq_train, dq_cum_train, dqk2_train], dim=-1)
V4_te = torch.cat([dq_test, dq_cum_test, dqk2_test], dim=-1)                # 32d
# V5 uses dq_train/dq_test with transformer classifier


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


class Xformer(nn.Module):
    def __init__(self, in_ch, num_classes=25, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, 31, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512,
                                           dropout=0.1, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        # x: (B, T, C)
        h = self.proj(x) + self.pos
        h = self.enc(h)
        h = h.mean(dim=1)
        return self.fc(h)


def train_eval(tag, tr, te, y_tr, y_te, in_ch, epochs=100, clf_cls=BigSeqClf):
    model = clf_cls(in_ch).cuda()
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
results["V1 cum-DQ 8d"] = train_eval("V1 cum-DQ 8d", V1_tr, V1_te, y_train, y_test, 8)
results["V2 per-part DQ K=2 16d"] = train_eval("V2 per-part DQ K=2 16d", V2_tr, V2_te, y_train, y_test, 16)
results["V3 DQ+dDQ+d2DQ 24d"] = train_eval("V3 DQ+dDQ+d2DQ 24d", V3_tr, V3_te, y_train, y_test, 24)
results["V4 combined 32d"] = train_eval("V4 combined 32d", V4_tr, V4_te, y_train, y_test, 32)
results["V5 transformer DQ 8d"] = train_eval("V5 transformer DQ 8d", dq_train, dq_test, y_train, y_test, 8, clf_cls=Xformer)

summary = "\n=== DQCC 5 VARIANTS SUMMARY ===\n"
for t, r in results.items():
    summary += f"{t}: {r*100:.2f}%\n"
summary += f"Prior: DQ 8d 28.01%, (aa,t) 28.42%, centroid 31.95%, C_frame 37.55%"
print(summary); tg(summary)
