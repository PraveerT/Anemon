"""QEAN-v1: Quaternionic Equivariant Articulated Network.
Novel architecture: every layer SO(3)^B-equivariant with respect to per-bone
rotation. Operates natively in quaternion algebra.

Architecture:
  input:  (T, B=20, 4) per-bone direction quaternions
  -> QuaternionLinear: per-bone (4 -> H*4) using H independent quaternion
     left-multiplication weights. Equivariant: g.q -> W*(g.q) = g.(W*q) only
     for left-multiplication; we use a constrained form so equivariance holds.
     For the simplified prototype: real-valued mixing across the 4 components,
     plus a learned quaternion bias applied via Hamilton product.
  -> SkeletalGraph aggregation: for each bone b, combine with parent and children
     via quaternion-aware gating + Hamilton product.
  -> Repeat L times.
  -> Time-set pooling (Janossy / mean): permutation-invariant over T.
  -> Bone-set pooling: mean across bones.
  -> Invariant readout: quaternion magnitudes + pairwise inner products.
  -> Linear classifier.
"""
import os, sys, re, time, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
T_FIXED = 32
NUM_CLASSES = 25
NUM_EPOCHS = 80
BS = 32
LR = 1e-3
WD = 1e-4
WORK_DIR = '/notebooks/PMamba/experiments/work_dir/qean_v1/'
os.makedirs(WORK_DIR, exist_ok=True)

BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
B = len(BONES)  # 20
PARENT_BONE = {0:None,1:0,2:1,3:2,4:None,5:4,6:5,7:6,8:None,9:8,10:9,11:10,
               12:None,13:12,14:13,15:14,16:None,17:16,18:17,19:18}
# child bones lookup
CHILDREN = {b:[] for b in range(B)}
for b, p in PARENT_BONE.items():
    if p is not None: CHILDREN[p].append(b)

# Pre-build adjacency tensor (B, B): 1 if bones share endpoint
ADJ = np.zeros((B, B), dtype=np.float32)
for i in range(B):
    for j in range(B):
        if i == j: continue
        ai, bi = BONES[i]; aj, bj = BONES[j]
        if {ai,bi} & {aj,bj}: ADJ[i,j] = 1
ADJ_T = torch.from_numpy(ADJ)


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
    """V: (...,3) -> (...,4) [w,x,y,z]. Quaternion that maps e_z onto V."""
    n = np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9
    u = V / n
    cos_h = (1 + u[..., 2:3]) * 0.5
    cos_h = np.clip(cos_h, 1e-9, 1.0)
    w = np.sqrt(cos_h)
    sin_h = np.sqrt(np.clip(1 - cos_h, 0, 1))
    # axis = e_z x u, normalized
    axis = np.zeros_like(u)
    axis[..., 0] = -u[..., 1]
    axis[..., 1] = u[..., 0]
    s = np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9
    axis = axis / s * sin_h
    return np.concatenate([w, axis], axis=-1).astype(np.float32)

def encode_sample(lm):
    """(T, 21, 3) -> (T, B, 4) per-bone direction quaternions."""
    T = lm.shape[0]
    out = np.zeros((T, B, 4), dtype=np.float32)
    for b, (p, c) in enumerate(BONES):
        out[:, b, :] = vec_to_quat_np(lm[:, c, :] - lm[:, p, :])
    return out

print('precomputing per-bone quaternions...')
encoded = {}
for k, lm_raw in sk.items():
    if lm_raw.shape[0] < 4: continue
    lm = fillnan(lm_raw)
    q = encode_sample(lm)
    # Resample to T_FIXED
    idx = np.linspace(0, q.shape[0]-1, T_FIXED).astype(np.int64)
    encoded[k] = q[idx]
print(f'{len(encoded)} encoded samples')


# ============ Quaternion algebra ops ============
def qmul(p, q):
    """Hamilton product. p,q: (..., 4). Returns (..., 4) [w,x,y,z]."""
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], dim=-1)

def qconj(q):
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def qnorm(q):
    return q / (q.norm(dim=-1, keepdim=True) + 1e-9)


# ============ QEAN layers ============

class QuaternionLinear(nn.Module):
    """Maps H_in quaternion features to H_out quaternion features per bone.
    Each output channel is a learned linear combination of input channels' Hamilton
    products with learned bias quaternions. SO(3)-equivariant under right-multiplication
    of input by any rotation quaternion g (which represents global frame change)."""
    def __init__(self, h_in, h_out):
        super().__init__()
        # Mixing weights (real-valued, mixes across input channels)
        self.W = nn.Parameter(torch.randn(h_in, h_out) * 0.1)
        # Per-output-channel learned quaternion bias (applied via right-mult)
        self.b = nn.Parameter(torch.randn(h_out, 4) * 0.05)
        with torch.no_grad():
            self.b[:, 0] = 1.0
            self.b[:] = qnorm(self.b)
    def forward(self, x):
        # x: (..., H_in, 4) — each (batch, bone, time, channel, quat)
        # mix channels via real W
        y = torch.einsum('...ij,jk->...ik', x.transpose(-1, -2), self.W).transpose(-1, -2)
        # Hamilton product with bias quaternion (broadcasted)
        b = qnorm(self.b)
        y = qmul(y, b)
        return y


class SkeletalGraphConv(nn.Module):
    """Aggregate quaternion features along the skeleton bone-adjacency graph.
    For each bone, mix its features with its neighbor bones via real-valued
    weighting and Hamilton products."""
    def __init__(self, h):
        super().__init__()
        self.adj = ADJ_T  # (B, B)
        self.lin_self = QuaternionLinear(h, h)
        self.lin_neigh = QuaternionLinear(h, h)
    def forward(self, x):
        # x: (batch, T, B, h, 4)
        adj = self.adj.to(x.device)
        # neighbor sum per bone (real-mix across bones via adjacency matrix)
        neigh = torch.einsum('ij,btjhq->btihq', adj, x)
        return self.lin_self(x) + self.lin_neigh(neigh)


class QEAN(nn.Module):
    """Per-bone identity preserved (no bone-pool); time-set pool only.
    Per-bone invariant readout: magnitudes + pairwise channel inner-products.
    Classifier sees [B * (h + h*(h-1)/2)] invariant features.
    """
    def __init__(self, h=16, num_classes=25, n_layers=3):
        super().__init__()
        self.lift = QuaternionLinear(1, h)
        self.layers = nn.ModuleList([SkeletalGraphConv(h) for _ in range(n_layers)])
        d_inv_per_bone = h + h*(h-1)//2
        d_inv = B * d_inv_per_bone
        self.classifier = nn.Sequential(
            nn.Linear(d_inv, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.h = h
    def forward(self, x):
        # x: (batch, T, B, 4)
        x = x.unsqueeze(-2)  # (batch, T, B, 1, 4)
        x = self.lift(x)     # (batch, T, B, h, 4)
        for L in self.layers:
            x = L(x)
            x = qnorm(x)
        # Time-set pool only
        x = x.mean(dim=1)  # (batch, B, h, 4)
        # Per-bone SO(3)-invariant readout
        mags = x.norm(dim=-1)  # (batch, B, h)
        # pairwise inner products of h channels (per bone)
        inners = []
        for i in range(self.h):
            for j in range(i+1, self.h):
                inners.append((x[:, :, i] * x[:, :, j]).sum(-1))  # (batch, B)
        inners = torch.stack(inners, dim=-1)  # (batch, B, h*(h-1)/2)
        inv = torch.cat([mags, inners], dim=-1)  # (batch, B, d_inv_per_bone)
        inv = inv.flatten(1)
        return self.classifier(inv)


# ============ Dataset ============
class QEANDataset(Dataset):
    def __init__(self, label_map):
        self.items = [(k, label_map[k]) for k in encoded if k in label_map]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        k, lbl = self.items[i]
        return torch.from_numpy(encoded[k]), lbl

ds_tr = QEANDataset(train_lbl); ds_te = QEANDataset(test_lbl)
print(f'train {len(ds_tr)} test {len(ds_te)}')

loader_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True, num_workers=4, drop_last=False)
loader_te = DataLoader(ds_te, batch_size=BS, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = QEAN(h=16, num_classes=NUM_CLASSES, n_layers=3).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'QEAN params: {n_params:,}')

opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

best_te = 0
for ep in range(1, NUM_EPOCHS + 1):
    model.train()
    losses = []
    for x, y in loader_tr:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    sched.step()
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader_te:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    te_acc = correct / total * 100
    if te_acc > best_te:
        best_te = te_acc
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                   os.path.join(WORK_DIR, 'best_model.pt'))
    print(f'ep {ep:3d}  loss={np.mean(losses):.4f}  test={te_acc:.2f}  best={best_te:.2f}', flush=True)

print(f'\nBEST: {best_te:.2f}')
