"""QuMamba: Quaternion-valued State-Space Model for SO(3)-equivariant temporal
processing of articulated motion.

Mathematical construction:
  State h_t in H^N (N quaternion features, 4N reals)
  Recurrence: h_t = h_{t-1} (*) qA + x_t (*) qB
  Output:     y_t = h_t (*) qC
  Where (*) is Hamilton (quaternion) product, qA, qB, qC are LEARNED unit
  quaternions, and channel-mixing is done via REAL-VALUED matrices that act
  on quaternion features element-wise (preserving SO(3)-equivariance under
  global LEFT-multiplication).

Theorem (SO(3)-invariance of classifier):
  If input is rotated x_t -> q * x_t for any unit quaternion q (uniformly across t),
  then h_t -> q * h_t (by linearity of the recurrence in h_{t-1} and x_t).
  Output magnitudes |y_t| = |q * h_t * qC| = |h_t * qC| are invariant.
  Therefore the classifier built on magnitude readouts is SO(3)-invariant.

Bridges:
  - REQNN (TPAMI 2024): static spatial SO(3)-equivariance via quaternion features
  - Mamba-3 (arXiv 2603.15569): complex-valued state for 1D rotational dynamics
  - QuMamba (this work): quaternion-valued state for 3D rotational dynamics
"""
import os, re, time, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
T_FIXED = 32
NUM_CLASSES = 25
NUM_EPOCHS = 120
BS = 32
LR = 5e-4
WD = 1e-4
HIDDEN = 192       # match DQNet-Mamba's d_model for fair comparison
N_LAYERS = 4
WORK_DIR = '/notebooks/PMamba/experiments/work_dir/qumamba_v1/'
os.makedirs(WORK_DIR, exist_ok=True)

BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
B = 20

def parse_annot(path):
    out = {}
    with open(path) as f:
        for line in f:
            mp_ = re.search(r'path:(\S+)', line); ml = re.search(r'label:(\d+)', line)
            if mp_ and ml: out[mp_.group(1)] = int(ml.group(1)) - 1
    return out
train_lbl = parse_annot(f'{ANNOT_ROOT}/nvgesture_train_correct_cvpr2016_v2.lst')
test_lbl  = parse_annot(f'{ANNOT_ROOT}/nvgesture_test_correct_cvpr2016_v2.lst')
print('loading landmarks...')
sk = dict(np.load(SK, allow_pickle=False))


def fillnan(arr):
    valid = np.isfinite(arr[..., 0]).all(axis=-1)
    last = None; out = arr.copy()
    for t in range(out.shape[0]):
        if valid[t]: last = out[t]
        elif last is not None: out[t] = last
    for t in range(out.shape[0]):
        if not np.isfinite(out[t]).all():
            for t2 in range(t+1, out.shape[0]):
                if valid[t2]: out[t] = out[t2]; break
            else: out[t] = 0
    return out

def vec_to_quat_np(V):
    n = np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9
    u = V / n
    cos_h = np.clip((1 + u[..., 2:3]) * 0.5, 1e-9, 1.0)
    w = np.sqrt(cos_h)
    sin_h = np.sqrt(np.clip(1 - cos_h, 0, 1))
    axis = np.zeros_like(u); axis[..., 0] = -u[..., 1]; axis[..., 1] = u[..., 0]
    s = np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9
    return np.concatenate([w, axis/s*sin_h], axis=-1).astype(np.float32)

def encode_sample(lm):
    """(T, 21, 3) -> (T, 20, 4) per-bone direction quaternions."""
    T = lm.shape[0]
    out = np.zeros((T, B, 4), dtype=np.float32)
    for b, (p, c) in enumerate(BONES):
        out[:, b, :] = vec_to_quat_np(lm[:, c, :] - lm[:, p, :])
    return out

print('precomputing per-bone direction quaternions...')
encoded = {}
for k, lm_raw in sk.items():
    if lm_raw.shape[0] < 4: continue
    lm = fillnan(lm_raw)
    q = encode_sample(lm)
    idx = np.linspace(0, q.shape[0]-1, T_FIXED).astype(np.int64)
    encoded[k] = q[idx]
print(f'{len(encoded)} encoded')


# ============ Quaternion ops (torch) ============
def qmul(p, q):
    """Hamilton product. p, q: (..., 4). Returns (..., 4)."""
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], dim=-1)

def qnorm_unit(q):
    return q / (q.norm(dim=-1, keepdim=True) + 1e-9)


# ============ QuMamba block ============

class QuMambaBlock(nn.Module):
    """One quaternion-valued SSM layer.
    h_t = (h_{t-1} channel-mix W_h) (*) qA + (x_t channel-mix W_x) (*) qB
    y_t = h_t (*) qC

    Channel mixing: real W ∈ R^{N_out × N_in} acts on the 4-component quaternions
    element-wise (each component of each quaternion mixed independently across
    channels). Preserves SO(3)-equivariance under uniform left-multiplication.
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W_h = nn.Linear(n_out, n_out, bias=False)  # state -> state mix
        self.W_x = nn.Linear(n_in, n_out, bias=False)   # input -> state mix
        self.W_y = nn.Linear(n_out, n_out, bias=False)  # output projection
        # Learned quaternion parameters qA, qB, qC (per-channel)
        self.qA_raw = nn.Parameter(torch.randn(n_out, 4) * 0.05)
        self.qB_raw = nn.Parameter(torch.randn(n_out, 4) * 0.05)
        self.qC_raw = nn.Parameter(torch.randn(n_out, 4) * 0.05)
        with torch.no_grad():
            self.qA_raw[:, 0] = 1.0
            self.qB_raw[:, 0] = 1.0
            self.qC_raw[:, 0] = 1.0
        # Decay factor for stability (ensures |qA| < 1)
        self.log_decay = nn.Parameter(torch.zeros(n_out))

    def _mix_quat(self, q_features, W):
        """q_features: (B, T, N_in, 4). W: Linear(N_in, N_out).
        Mix across channel dim independently for each quaternion component.
        Returns (B, T, N_out, 4).
        """
        # Treat each of 4 components as a separate scalar feature mixed via W
        return W(q_features.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, x):
        # x: (B, T, n_in, 4) — sequence of quaternion features
        Bz, T, _, _ = x.shape
        # Pre-mix input across channels
        x_mix = self._mix_quat(x, self.W_x)  # (B, T, n_out, 4)
        qA = qnorm_unit(self.qA_raw) * torch.sigmoid(self.log_decay).unsqueeze(-1)
        qB = qnorm_unit(self.qB_raw)
        qC = qnorm_unit(self.qC_raw)
        # Recurrence
        h = torch.zeros(Bz, x_mix.shape[2], 4, device=x.device)
        outs = []
        for t in range(T):
            h_mix = self._mix_quat(h.unsqueeze(1), self.W_h).squeeze(1)
            h = qmul(h_mix, qA) + qmul(x_mix[:, t], qB)
            y = qmul(h, qC)
            outs.append(y)
        out = torch.stack(outs, dim=1)
        # Project output channels (real-mix again)
        out = self._mix_quat(out, self.W_y)
        return out


class QuMamba(nn.Module):
    def __init__(self, hidden=HIDDEN, n_layers=N_LAYERS, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        # Lift: 20 bone direction quaternions -> hidden quaternion features per frame
        self.lift = nn.Linear(B, hidden, bias=False)  # mixes across bones for each component
        # QuMamba layers
        self.blocks = nn.ModuleList([QuMambaBlock(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        # Invariant readout: per-channel quaternion magnitude (4*hidden -> hidden real values)
        # Plus pairwise inner products across channels (small set)
        d_inv = hidden  # only use magnitudes for now
        self.head = nn.Sequential(
            nn.Linear(d_inv, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def _mix_bones(self, x):
        # x: (B, T, 20, 4) -> apply real Linear(20, hidden) per quaternion component
        return self.lift(x.transpose(-1, -2)).transpose(-1, -2)  # (B, T, hidden, 4)

    def forward(self, x):
        # x: (B, T, 20, 4)
        x = self._mix_bones(x)  # (B, T, hidden, 4)
        for blk, norm in zip(self.blocks, self.norms):
            residual = x
            # Apply LayerNorm on quaternion magnitudes per channel (preserves direction)
            mags = x.norm(dim=-1, keepdim=True) + 1e-9
            normed_mags = norm(mags.squeeze(-1)).unsqueeze(-1)
            x_normed = x / mags * normed_mags
            x_blk = blk(x_normed)
            x = self.dropout(x_blk) + residual
        # Take last time step's quaternion magnitudes (SO(3)-invariant)
        last = x[:, -1]  # (B, hidden, 4)
        mag = last.norm(dim=-1)  # (B, hidden) — SO(3) invariant
        return self.head(mag)


class DQDataset(Dataset):
    def __init__(self, label_map):
        self.items = [(k, label_map[k]) for k in encoded if k in label_map]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        k, lbl = self.items[i]
        return torch.from_numpy(encoded[k]), lbl

ds_tr = DQDataset(train_lbl); ds_te = DQDataset(test_lbl)
print(f'train {len(ds_tr)} test {len(ds_te)}')

loader_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True, num_workers=4)
loader_te = DataLoader(ds_te, batch_size=BS, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = QuMamba(hidden=HIDDEN, n_layers=N_LAYERS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'QuMamba params: {n_params:,}')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

best_te = 0
for ep in range(1, NUM_EPOCHS + 1):
    model.train()
    losses = []
    for x, y in loader_tr:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    sched.step()
    model.eval()
    correct = 0; total = 0; all_probs = []; all_lbl = []
    with torch.no_grad():
        for x, y in loader_te:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
            all_lbl.append(y.cpu().numpy())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    te_acc = correct / total * 100
    if te_acc > best_te:
        best_te = te_acc
        np.savez(os.path.join(WORK_DIR, 'best_probs.npz'),
                 probs=np.concatenate(all_probs), labels=np.concatenate(all_lbl))
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                   os.path.join(WORK_DIR, 'best_model.pt'))
    print(f'ep {ep:3d}  loss={np.mean(losses):.4f}  test={te_acc:.2f}  best={best_te:.2f}', flush=True)
print(f'\nBEST: {best_te:.2f}  (DQNet-Mamba=77.80, DQNet-Transformer=71.78)')
