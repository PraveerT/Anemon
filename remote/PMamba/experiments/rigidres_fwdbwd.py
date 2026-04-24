"""C_fb: rigidres + backward residual.

For each pair (t, t+1):
  res_fwd = p(t+1) - (R_fwd·p(t) + t_fwd)   [current B]
  res_bwd = p(t) - (R_bwd·p(t+1) + t_bwd)   [independent Kabsch, reversed roles]

Input C_fb = [xyz(3), res_fwd(3), res_bwd(3), t(1)] = 10-ch.
Both residuals per-point, different temporal directions.
5 seeds. Compare to B (rigidres 7-ch).
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


PTS = 128
tg("Backward-forward cycle test: C_fb = B + res_bwd (10-ch). 5 seeds.")
print('Collecting...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RES_FWD = np.zeros((N, 32, PTS, 3), dtype=np.float32)     # frame 0 = 0 (no prior)
    RES_BWD = np.zeros((N, 32, PTS, 3), dtype=np.float32)     # frame F-1 = 0 (no next)
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
            # Forward Kabsch (t -> t+1)
            R_f, tr_f = kabsch_Rt(xyz_samp[t:t+1], xyz_samp[t+1:t+2], matched[t:t+1])
            rigid_fwd = torch.bmm(R_f, xyz_samp[t:t+1].transpose(-1, -2)).transpose(-1, -2) + tr_f.unsqueeze(1)
            res_fwd = xyz_samp[t+1:t+2] - rigid_fwd
            RES_FWD[i, t+1] = res_fwd[0].cpu().numpy()
            # Backward Kabsch (t+1 -> t) independently
            R_b, tr_b = kabsch_Rt(xyz_samp[t+1:t+2], xyz_samp[t:t+1], matched[t:t+1])
            rigid_bwd = torch.bmm(R_b, xyz_samp[t+1:t+2].transpose(-1, -2)).transpose(-1, -2) + tr_b.unsqueeze(1)
            res_bwd = xyz_samp[t:t+1] - rigid_bwd
            RES_BWD[i, t] = res_bwd[0].cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RES_FWD),
            torch.from_numpy(RES_BWD), torch.from_numpy(labels))


xyz_tr, resf_tr, resb_tr, y_tr = collect('train')
xyz_te, resf_te, resb_te, y_te = collect('test')
print(f'xyz: {xyz_tr.shape}  resf: {resf_tr.shape}  resb: {resb_tr.shape}')
tg("Collection done. Training A/B/C_fb × 5 seeds.")

T = xyz_tr.shape[1]
t_tr = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0],  T, PTS, 1)

A_tr = torch.cat([xyz_tr, t_tr], dim=-1)
A_te = torch.cat([xyz_te, t_te], dim=-1)
B_tr = torch.cat([xyz_tr, resf_tr, t_tr], dim=-1)
B_te = torch.cat([xyz_te, resf_te, t_te], dim=-1)
C_tr = torch.cat([xyz_tr, resf_tr, resb_tr, t_tr], dim=-1)
C_te = torch.cat([xyz_te, resf_te, resb_te, t_te], dim=-1)


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
    c = train_eval(seed, C_tr, y_tr, C_te, y_te, 10)
    msg = f"seed {seed}: A={a*100:.2f}  B={b*100:.2f}  C_fb={c*100:.2f}  (C-B)={(c-b)*100:+.2f}"
    rows.append((a, b, c)); print(msg); tg(msg)

A_ = [r[0] for r in rows]; B_ = [r[1] for r in rows]; C_ = [r[2] for r in rows]
dBA = [b-a for a,b,c in rows]; dCB = [c-b for a,b,c in rows]

summary = f"""
=== FWD-BWD CYCLE (C_fb vs B) 5 seeds ===
A: {np.mean(A_)*100:.2f} ± {np.std(A_)*100:.2f}
B (rigidres): {np.mean(B_)*100:.2f} ± {np.std(B_)*100:.2f}
C_fb (+ backward residual): {np.mean(C_)*100:.2f} ± {np.std(C_)*100:.2f}
B - A: {np.mean(dBA)*100:+.2f} ± {np.std(dBA)*100:.2f}
C_fb - B: {np.mean(dCB)*100:+.2f} ± {np.std(dCB)*100:.2f}  (min {min(dCB)*100:+.2f}, max {max(dCB)*100:+.2f})
Verdict: C_fb {'beats B' if np.mean(dCB) > 0.5/100 else 'ties B' if abs(np.mean(dCB)) < 0.5/100 else 'loses to B'}
"""
print(summary); tg(summary)
