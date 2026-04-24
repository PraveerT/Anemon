"""Seal the rigidres hypothesis: 5 seeds × (A, B) + per-class analysis.

Same tiny model + schedule as rigidres_abtest.py.
For each seed in [0, 1, 2, 3, 4]:
  train A (xyz+t, 4-ch) and B (xyz+res+t, 7-ch).
Report mean, std, min, max delta.
Per-class: compare B vs A predictions on all classes; report which classes benefit.
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
tg("Rigidres SEAL: collecting data then 5-seed A/B with per-class analysis")
print('Collecting samples...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RES = np.zeros((N, 32, PTS, 3), dtype=np.float32)
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
            R, tr = kabsch_Rt(xyz_samp[t:t+1], xyz_samp[t+1:t+2], matched[t:t+1])
            rigid_pred = torch.bmm(R, xyz_samp[t:t+1].transpose(-1, -2)).transpose(-1, -2) + tr.unsqueeze(1)
            res = xyz_samp[t+1:t+2] - rigid_pred
            RES[i, t+1] = res[0].cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  {phase} {i+1}/{N}')
    return torch.from_numpy(XYZ), torch.from_numpy(RES), torch.from_numpy(labels)


xyz_train, res_train, y_train = collect('train')
xyz_test, res_test, y_test = collect('test')
print(f'xyz: {xyz_train.shape}')
tg(f"Collection done. 5 seeds × A/B training beginning.")

T = xyz_train.shape[1]
t_tr = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_train.shape[0], T, PTS, 1)
t_te = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_test.shape[0],  T, PTS, 1)

A_tr = torch.cat([xyz_train, t_tr], dim=-1)
A_te = torch.cat([xyz_test,  t_te], dim=-1)
B_tr = torch.cat([xyz_train, res_train, t_tr], dim=-1)
B_te = torch.cat([xyz_test,  res_test,  t_te], dim=-1)


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
        h = self.mlp(x)
        h = h.max(dim=2).values
        h = h.transpose(1, 2)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        return self.fc(self.pool(h).squeeze(-1))


def train_eval(seed, X_tr, y_tr, X_te, y_te, in_ch, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = TinyPointTemporal(in_ch).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    BS = 32; N_train = len(X_tr); best = 0; best_preds = None
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
            preds_list = []
            for i in range(0, len(X_te), 64):
                preds_list.append(model(X_te_c[i:i+64]).argmax(-1))
            preds = torch.cat(preds_list)
            acc = (preds == y_te_c).float().mean().item()
        if acc > best:
            best = acc
            best_preds = preds.cpu()
    return best, best_preds


results = []
preds_A_all, preds_B_all = [], []
for seed in range(5):
    acc_A, pa = train_eval(seed, A_tr, y_train, A_te, y_test, 4)
    acc_B, pb = train_eval(seed, B_tr, y_train, B_te, y_test, 7)
    delta = acc_B - acc_A
    results.append((seed, acc_A, acc_B, delta))
    preds_A_all.append(pa); preds_B_all.append(pb)
    msg = f"seed {seed}: A={acc_A*100:.2f}  B={acc_B*100:.2f}  delta={delta*100:+.2f}pp"
    print(msg); tg(msg)

accs_A = [r[1] for r in results]
accs_B = [r[2] for r in results]
deltas = [r[3] for r in results]
mean_A = np.mean(accs_A) * 100
std_A = np.std(accs_A) * 100
mean_B = np.mean(accs_B) * 100
std_B = np.std(accs_B) * 100
mean_delta = np.mean(deltas) * 100
std_delta = np.std(deltas) * 100
min_delta = min(deltas) * 100
max_delta = max(deltas) * 100

# Per-class analysis: for each seed, compare class accuracies
y_test_np = y_test.numpy()
class_deltas = np.zeros(25)
class_counts = np.zeros(25, dtype=int)
for pa, pb in zip(preds_A_all, preds_B_all):
    pa = pa.numpy(); pb = pb.numpy()
    for c in range(25):
        mask = y_test_np == c
        if mask.sum() > 0:
            acc_A_c = (pa[mask] == c).mean()
            acc_B_c = (pb[mask] == c).mean()
            class_deltas[c] += (acc_B_c - acc_A_c)
            class_counts[c] += 1
class_mean_delta = class_deltas / np.maximum(class_counts, 1) * 100
# Sort: classes where B helps most
sort_idx = np.argsort(class_mean_delta)[::-1]

summary = f"""
=== RIGIDRES SEAL (5 seeds) ===
Per-seed deltas: {[f'{d*100:+.2f}' for d in deltas]}
A: {mean_A:.2f} ± {std_A:.2f}
B: {mean_B:.2f} ± {std_B:.2f}
Delta: {mean_delta:+.2f}pp ± {std_delta:.2f}pp  (min {min_delta:+.2f}, max {max_delta:+.2f})

Per-class delta (B - A, averaged over 5 seeds, top helped):
""" + "\n".join([f"  class {c}: {class_mean_delta[c]:+.2f}pp" for c in sort_idx[:5]]) + f"""

Bottom 5 (where rigidres hurts):
""" + "\n".join([f"  class {c}: {class_mean_delta[c]:+.2f}pp" for c in sort_idx[-5:]])

print(summary); tg(summary)
