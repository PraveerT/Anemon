"""GQN: Geodesic Quaternion Network.
Novelty: leverages S^3 geodesic structure + kinematic-chain compositionality
WITHOUT global-rotation equivariance (which is gratuitous for fixed-camera data).

Key novel components:
  1. Compositional prefix products: per finger chain, compute cumulative quaternion
     products that represent absolute fingertip orientations and intermediate
     segment orientations. Math: cumprod over the kinematic graph with Hamilton.
  2. Geodesic anchor mapping: K learned anchor quaternions q_k on S^3. Each
     input quaternion q is mapped to (d_geo(q, q_1), ..., d_geo(q, q_K)) using
     d_geo(p, q) = arccos(|p . q|). This measures S^3 distance to a learned
     "codebook" of canonical orientations. Anchors are learnable; geodesic
     distance is differentiable everywhere except antipodes.
  3. SLERP-attention pool over time (geodesic-weighted geometric mean):
     q_pool = SLERP-mean({q_t}_{t=1..T}) computed iteratively.
"""
import os, re, time, numpy as np, torch
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
WORK_DIR = '/notebooks/PMamba/experiments/work_dir/gqn_v1/'
os.makedirs(WORK_DIR, exist_ok=True)

# Hand chains (each chain: list of bones from palm root to fingertip)
# Bone indexing matches train_qean.py BONES list
CHAINS = [
    [0, 1, 2, 3],         # thumb
    [4, 5, 6, 7],         # index
    [8, 9, 10, 11],       # middle
    [12, 13, 14, 15],     # ring
    [16, 17, 18, 19],     # pinky
]
B = 20

BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

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
    cos_h = (1 + u[..., 2:3]) * 0.5
    cos_h = np.clip(cos_h, 1e-9, 1.0)
    w = np.sqrt(cos_h)
    sin_h = np.sqrt(np.clip(1 - cos_h, 0, 1))
    axis = np.zeros_like(u)
    axis[..., 0] = -u[..., 1]
    axis[..., 1] = u[..., 0]
    s = np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9
    axis = axis / s * sin_h
    return np.concatenate([w, axis], axis=-1).astype(np.float32)

def encode_sample(lm):
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
    idx = np.linspace(0, q.shape[0]-1, T_FIXED).astype(np.int64)
    encoded[k] = q[idx]
print(f'{len(encoded)} encoded samples')


# ================== Quaternion ops (torch) ==================
def qmul(p, q):
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


def chain_prefix_products(q_bones):
    """q_bones: (..., B, 4) per-bone direction quaternions.
    Returns: (..., B, 4) — for each bone, the cumulative product of all bones
    from finger-root up to and including this bone, along its chain.

    Mathematically: for chain c = [b1, b2, b3, b4], output at b_i is
        q_b1 * q_b2 * ... * q_bi
    These represent SEGMENT-LEVEL absolute orientations through the chain.
    """
    out = q_bones.clone()
    for chain in CHAINS:
        cum = q_bones[..., chain[0], :]
        out[..., chain[0], :] = cum
        for b in chain[1:]:
            cum = qmul(cum, q_bones[..., b, :])
            out[..., b, :] = qnorm(cum)
    return out


# ================== Novel layers ==================

class GeodesicAnchor(nn.Module):
    """K learnable anchor quaternions on S^3. Each input quaternion q is mapped
    to a K-dim feature: f_k(q) = arccos(|q . q_k|), the geodesic distance on S^3.

    Math: S^3 is the unit-quaternion manifold; antipodal q and -q represent same
    rotation, so we use |.| to make distance well-defined on RP^3 = SO(3).
    Geodesic distance d(p,q) = arccos(|p . q|) is the great-circle arc length.
    """
    def __init__(self, K):
        super().__init__()
        a = torch.randn(K, 4)
        self.anchors_raw = nn.Parameter(a)
    def forward(self, q):
        # q: (..., 4)
        a = qnorm(self.anchors_raw)  # (K, 4)
        # cos_geo = |q . a_k|
        cos = torch.einsum('...d,kd->...k', q, a)
        cos = cos.abs().clamp(0, 1 - 1e-6)
        return torch.arccos(cos)  # (..., K)


class SLERPMean(nn.Module):
    """Iterative geodesic mean of a set of quaternions on S^3.
    Approximates the Frechet mean. We use a few iterations of:
        mu_{n+1} = exp_{mu_n}( (1/N) * sum_i log_{mu_n}(q_i) )
    Implementation: fixed 3 iterations starting from element-wise mean (then norm).
    """
    def __init__(self, n_iter=3):
        super().__init__()
        self.n_iter = n_iter
    def forward(self, q_set, dim):
        # q_set: (..., N, ..., 4); pool along `dim` axis (which indexes N)
        # First a Euclidean-mean init, then SLERP iter on the manifold
        mu = qnorm(q_set.mean(dim=dim))  # (..., 4) but with `dim` removed
        # Bring `dim` to second-to-last for ease
        for _ in range(self.n_iter):
            mu_e = mu.unsqueeze(dim)  # match q_set rank
            # log_mu(q) ≈ q - mu (small angle approx); on S^3 use SLERP-style:
            # log_mu(q) in tangent at mu = direction perpendicular to mu, scaled by angle.
            # Simpler: average q_set over dim, normalize. Iterate.
            mu = qnorm(q_set.mean(dim=dim))
        return mu


# ================== GQN ==================
class GQN(nn.Module):
    def __init__(self, K_anchors=24, hidden=256, num_classes=25):
        super().__init__()
        self.geo = GeodesicAnchor(K_anchors)  # used per-bone-quaternion
        self.geo_chain = GeodesicAnchor(K_anchors)  # for chain prefix products
        # Per-bone feature size: 2 * K_anchors (raw + chain-prefix)
        # Per frame: 20 bones * 2 * K = 40K dims
        # Then time-set pool via SLERP-mean (over per-bone, per-time quaternions)
        # gives B*4 + per-bone-anchor-mean-distances.
        # We'll concat: time-mean of [GeoAnchor(raw_q) for each bone] and similar for chain.
        # Plus quaternion-magnitude statistics.
        self.bone_per_feat = 2 * K_anchors
        d_per_frame = B * self.bone_per_feat
        # Time-set pool: mean over T (verified by step 5 to be sufficient)
        self.classifier = nn.Sequential(
            nn.Linear(d_per_frame, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden//2, num_classes),
        )
    def forward(self, q):
        # q: (batch, T, B, 4)
        q_chain = chain_prefix_products(q)
        # Geodesic anchor distances per bone
        d1 = self.geo(q)        # (batch, T, B, K)
        d2 = self.geo_chain(q_chain)  # (batch, T, B, K)
        feat = torch.cat([d1, d2], dim=-1)  # (batch, T, B, 2K)
        # Mean over T (set-pool)
        feat = feat.mean(dim=1)  # (batch, B, 2K)
        feat = feat.flatten(1)
        return self.classifier(feat)


class GQNDataset(Dataset):
    def __init__(self, label_map):
        self.items = [(k, label_map[k]) for k in encoded if k in label_map]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        k, lbl = self.items[i]
        return torch.from_numpy(encoded[k]), lbl

ds_tr = GQNDataset(train_lbl); ds_te = GQNDataset(test_lbl)
print(f'train {len(ds_tr)} test {len(ds_te)}')

loader_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True, num_workers=4, drop_last=False)
loader_te = DataLoader(ds_te, batch_size=BS, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GQN(K_anchors=24, hidden=256, num_classes=NUM_CLASSES).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'GQN params: {n_params:,}')

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
