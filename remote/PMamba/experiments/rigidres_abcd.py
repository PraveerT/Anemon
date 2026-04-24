"""A/B/C_fb/D_ms test:

A: [xyz, t]                                4-ch
B: [xyz, res_1, t]                         7-ch
C_fb: [xyz, res_fwd_1, res_bwd_1, t]      10-ch
D_ms: [xyz, res_1, res_2, res_4, t]       13-ch  (multi-scale forward)

res_k(t) = p(t) - (R_k·p(t-k) + t_k), where R_k = Kabsch(frame t-k, frame t).
5 seeds, same tiny model, same recipe.
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


def kabsch_Rt(src, tgt, mask):
    B = src.shape[0]; device = src.device
    w = mask.float()
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tgt * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src - sm; tc = tgt - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    t = tm.squeeze(1) - torch.bmm(R, sm.transpose(-1, -2)).squeeze(-1)
    return R, t


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


def compute_residual_at_scale(xyz_samp, matched, F_, k):
    """res_k: (F, P, 3). res_k[t] = p(t) - (R·p(t-k) + t_rigid) for t>=k, else 0.
    matched_k via chain: point must be matched in all k consecutive steps."""
    F_plus, P, _ = xyz_samp.shape
    res = torch.zeros(F_, P, 3, device=xyz_samp.device, dtype=xyz_samp.dtype)
    for t in range(k, F_):
        # Chain correspondence mask: valid from t-k to t requires matched[t-k]..matched[t-1]
        mchain = matched[t-k]
        for m in range(t-k+1, t):
            mchain = mchain & matched[m]
        R, tr = kabsch_Rt(
            xyz_samp[t-k:t-k+1], xyz_samp[t:t+1], mchain.unsqueeze(0),
        )
        rigid_pred = torch.bmm(R, xyz_samp[t-k:t-k+1].transpose(-1, -2)).transpose(-1, -2) + tr.unsqueeze(1)
        res[t] = (xyz_samp[t:t+1] - rigid_pred)[0]
    return res


PTS = 128
tg("A/B/C_fb/D_ms: adding multi-scale residual. 5 seeds.")
print('Collecting...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    R1_F = np.zeros((N, 32, PTS, 3), dtype=np.float32)   # res_1 forward
    R1_B = np.zeros((N, 32, PTS, 3), dtype=np.float32)   # res_1 backward
    R2 = np.zeros((N, 32, PTS, 3), dtype=np.float32)     # res_2 (gap 2)
    R4 = np.zeros((N, 32, PTS, 3), dtype=np.float32)     # res_4 (gap 4)
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
        # Forward+backward at scale 1
        for t in range(F_ - 1):
            R_f, tr_f = kabsch_Rt(xyz_samp[t:t+1], xyz_samp[t+1:t+2], matched[t:t+1])
            rigid_f = torch.bmm(R_f, xyz_samp[t:t+1].transpose(-1, -2)).transpose(-1, -2) + tr_f.unsqueeze(1)
            R1_F[i, t+1] = (xyz_samp[t+1:t+2] - rigid_f)[0].cpu().numpy()
            R_b, tr_b = kabsch_Rt(xyz_samp[t+1:t+2], xyz_samp[t:t+1], matched[t:t+1])
            rigid_b = torch.bmm(R_b, xyz_samp[t+1:t+2].transpose(-1, -2)).transpose(-1, -2) + tr_b.unsqueeze(1)
            R1_B[i, t] = (xyz_samp[t:t+1] - rigid_b)[0].cpu().numpy()
        # Multi-scale forward residuals at gaps 2 and 4
        res_2 = compute_residual_at_scale(xyz_samp, matched, F_, 2)
        res_4 = compute_residual_at_scale(xyz_samp, matched, F_, 4)
        R2[i] = res_2.cpu().numpy()
        R4[i] = res_4.cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(R1_F), torch.from_numpy(R1_B),
            torch.from_numpy(R2), torch.from_numpy(R4), torch.from_numpy(labels))


xyz_tr, r1f_tr, r1b_tr, r2_tr, r4_tr, y_tr = collect('train')
xyz_te, r1f_te, r1b_te, r2_te, r4_te, y_te = collect('test')
print(f'xyz: {xyz_tr.shape}  res_1: {r1f_tr.shape}  res_2: {r2_tr.shape}  res_4: {r4_tr.shape}')
tg("Collection done. Training A/B/C_fb/D_ms × 5 seeds.")

T = xyz_tr.shape[1]
t_tr = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0],  T, PTS, 1)

A_tr = torch.cat([xyz_tr, t_tr], dim=-1)
A_te = torch.cat([xyz_te, t_te], dim=-1)
B_tr = torch.cat([xyz_tr, r1f_tr, t_tr], dim=-1)
B_te = torch.cat([xyz_te, r1f_te, t_te], dim=-1)
Cfb_tr = torch.cat([xyz_tr, r1f_tr, r1b_tr, t_tr], dim=-1)
Cfb_te = torch.cat([xyz_te, r1f_te, r1b_te, t_te], dim=-1)
Dms_tr = torch.cat([xyz_tr, r1f_tr, r2_tr, r4_tr, t_tr], dim=-1)
Dms_te = torch.cat([xyz_te, r1f_te, r2_te, r4_te, t_te], dim=-1)


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
    b = train_eval(seed, B_tr, y_tr, B_te, y_te, 7)
    c = train_eval(seed, Cfb_tr, y_tr, Cfb_te, y_te, 10)
    d = train_eval(seed, Dms_tr, y_tr, Dms_te, y_te, 13)
    msg = f"seed {seed}: A={a*100:.2f}  B={b*100:.2f}  C_fb={c*100:.2f}  D_ms={d*100:.2f}  (D-B)={(d-b)*100:+.2f}"
    rows.append((a, b, c, d)); print(msg); tg(msg)

A_ = [r[0] for r in rows]; B_ = [r[1] for r in rows]; C_ = [r[2] for r in rows]; D_ = [r[3] for r in rows]
dBA = [b-a for a,b,c,d in rows]
dCB = [c-b for a,b,c,d in rows]
dDB = [d-b for a,b,c,d in rows]

summary = f"""
=== A/B/C_fb/D_ms 5 seeds ===
A:     {np.mean(A_)*100:.2f} ± {np.std(A_)*100:.2f}
B:     {np.mean(B_)*100:.2f} ± {np.std(B_)*100:.2f}   (B-A: {np.mean(dBA)*100:+.2f})
C_fb:  {np.mean(C_)*100:.2f} ± {np.std(C_)*100:.2f}   (C-B: {np.mean(dCB)*100:+.2f})
D_ms:  {np.mean(D_)*100:.2f} ± {np.std(D_)*100:.2f}   (D-B: {np.mean(dDB)*100:+.2f}, min {min(dDB)*100:+.2f}, max {max(dDB)*100:+.2f})
Winner: {'D_ms' if np.mean(D_) > max(np.mean(C_), np.mean(B_)) else 'C_fb' if np.mean(C_) > np.mean(B_) else 'B'}
"""
print(summary); tg(summary)
