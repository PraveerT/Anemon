"""Minimal A/B test: does rigid-subtraction residual actually help?

Tiny PointNet-like model:
  per-point linear -> max-pool over P -> temporal 1D-conv -> classify.

Variant A: input per-point = [xyz (3), t (1)] = 4-ch
Variant B: input per-point = [xyz (3), res (3), t (1)] = 7-ch

Same arch, same training recipe, same seed. Compare final test accs.
If rigidres is a real signal, B > A noticeably.

Target runtime: ~5 min collection + 5 min training per variant.
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
    """Batched Kabsch. src, tgt: (B, N, 3). mask: (B, N) float. Returns R (B,3,3), t (B,3)."""
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
    D = torch.diag_embed(torch.stack(
        [torch.ones_like(det), torch.ones_like(det), det], dim=-1))
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
print(f'Collecting samples (pts={PTS})...')
tg(f"rigidres A/B test: collecting pts={PTS}")


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
        # Residuals per pair, apply to frame t+1 (frame 0 residual = 0)
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
print(f'xyz: {xyz_train.shape}  res: {res_train.shape}')
tg(f"Collection done. train {xyz_train.shape[0]}, test {xyz_test.shape[0]}. Training A (xyz+t) now.")

# Time channel (normalized)
T = xyz_train.shape[1]
time_ch_train = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_train.shape[0], T, PTS, 1)
time_ch_test  = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_test.shape[0],  T, PTS, 1)

def inputA(xyz, t):
    return torch.cat([xyz, t], dim=-1)                          # (B, T, P, 4)

def inputB(xyz, res, t):
    return torch.cat([xyz, res, t], dim=-1)                     # (B, T, P, 7)


class TinyPointTemporal(nn.Module):
    """per-point MLP -> max-pool over P -> temporal 1d-conv -> classify."""
    def __init__(self, in_ch, num_classes=25, per_point=64, temporal=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, per_point),
            nn.GELU(),
            nn.Linear(per_point, per_point),
        )
        self.conv1 = nn.Conv1d(per_point, temporal, 3, padding=1)
        self.conv2 = nn.Conv1d(temporal, temporal, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(temporal, num_classes)

    def forward(self, x):
        # x: (B, T, P, C)
        B, T, P, C = x.shape
        h = self.mlp(x)                                          # (B, T, P, K)
        h = h.max(dim=2).values                                  # (B, T, K)
        h = h.transpose(1, 2)                                    # (B, K, T)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        return self.fc(self.pool(h).squeeze(-1))


def train_eval(tag, X_tr, y_tr, X_te, y_te, in_ch, epochs=60):
    torch.manual_seed(0)
    model = TinyPointTemporal(in_ch).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    BS = 32; N_train = len(X_tr); best = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N_train)
        for i in range(0, N_train, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            out = model(X_tr_c[idx])
            loss = F.cross_entropy(out, y_tr_c[idx])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            # Chunked eval to avoid OOM
            preds = []
            for i in range(0, len(X_te), 64):
                preds.append(model(X_te_c[i:i+64]).argmax(-1))
            preds = torch.cat(preds)
            acc = (preds == y_te_c).float().mean().item()
        if acc > best: best = acc
        if ep % 10 == 0 or ep == epochs - 1:
            msg = f"{tag} ep {ep:2d}  test={acc*100:.2f}%  best={best*100:.2f}%"
            print(msg); tg(msg)
    final = f"=== {tag} FINAL: {best*100:.2f}% ==="
    print(final); tg(final)
    return best


# Build A inputs
A_tr = inputA(xyz_train, time_ch_train)
A_te = inputA(xyz_test, time_ch_test)

# Build B inputs
B_tr = inputB(xyz_train, res_train, time_ch_train)
B_te = inputB(xyz_test, res_test, time_ch_test)

print(f'\\nA shape: {A_tr.shape}  B shape: {B_tr.shape}')

acc_A = train_eval("A (xyz+t, 4-ch)", A_tr, y_train, A_te, y_test, 4)
acc_B = train_eval("B (xyz+res+t, 7-ch)", B_tr, y_train, B_te, y_test, 7)

summary = f"""
=== RIGIDRES A/B TEST ===
A (xyz+t, 4-ch):      {acc_A*100:.2f}%
B (xyz+res+t, 7-ch):  {acc_B*100:.2f}%
Delta (B-A):          {(acc_B-acc_A)*100:+.2f}pp
"""
print(summary); tg(summary)
