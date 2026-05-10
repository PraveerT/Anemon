"""Investigate why DQN (38.59) lost so much vs shallow MLP on dual quat (64.11).

Test ladder:
  A) Plain MLP on DQN's exact encoded input (no DQ algebra at all).
     If 64% -> DQN encoding matches diagnostic; the DQ layers are the loss.
     If 38% -> there's an encoding/loading bug.
  B) DQN with classifier directly on flat input, NO DQ layers (skip lift+conv).
     Should match A.
  C) DQN with DQ layers but flat readout (skip dq_norm_invariants, keep all 8D).
     Tests if readout is the bottleneck.
  D) DQN as-is (for reference).
"""
import os, sys, re, time, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Reuse encoded data from train_dqn.py infrastructure
sys.path.insert(0, '/notebooks')
SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
T_FIXED = 32
NUM_CLASSES = 25
BS = 32
EP = 100

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

def qmul_np(p, q):
    pw,px,py,pz = p[...,0],p[...,1],p[...,2],p[...,3]
    qw,qx,qy,qz = q[...,0],q[...,1],q[...,2],q[...,3]
    return np.stack([pw*qw-px*qx-py*qy-pz*qz, pw*qx+px*qw+py*qz-pz*qy,
                      pw*qy-px*qz+py*qw+pz*qx, pw*qz+px*qy-py*qx+pz*qw], axis=-1)

def encode_sample(lm):
    T = lm.shape[0]
    out = np.zeros((T, B, 8), dtype=np.float32)
    for b, (p, c) in enumerate(BONES):
        bone_vec = lm[:, c, :] - lm[:, p, :]
        rot_q = vec_to_quat_np(bone_vec)
        mid = (lm[:, c, :] + lm[:, p, :]) / 2
        t_pure = np.concatenate([np.zeros((T, 1), dtype=np.float32), mid], axis=-1)
        pos_q = qmul_np(t_pure, rot_q) * 0.5
        out[:, b, :4] = rot_q
        out[:, b, 4:] = pos_q
    return out

print('precompute...')
encoded = {}
for k, lm_raw in sk.items():
    if lm_raw.shape[0] < 4: continue
    lm = fillnan(lm_raw)
    dq = encode_sample(lm)
    idx = np.linspace(0, dq.shape[0]-1, T_FIXED).astype(np.int64)
    encoded[k] = dq[idx]
print(f'{len(encoded)} samples')

# Build flat dataset for tests A, B
X_tr, y_tr, X_te, y_te = [], [], [], []
for k in encoded:
    feat = encoded[k].astype(np.float32).reshape(-1)  # (T_FIXED * B * 8) = 5120
    if k in train_lbl: X_tr.append(feat); y_tr.append(train_lbl[k])
    elif k in test_lbl: X_te.append(feat); y_te.append(test_lbl[k])
X_tr = np.array(X_tr); y_tr = np.array(y_tr, dtype=np.int64)
X_te = np.array(X_te); y_te = np.array(y_te, dtype=np.int64)
mean = X_tr.mean(0); std = X_tr.std(0) + 1e-7
X_tr_n = ((X_tr - mean) / std).astype(np.float32)
X_te_n = ((X_te - mean) / std).astype(np.float32)
print(f'X shape: {X_tr.shape}, train {len(X_tr)} test {len(X_te)}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Xt = torch.from_numpy(X_tr_n).to(device); yt = torch.from_numpy(y_tr).to(device)
Xv = torch.from_numpy(X_te_n).to(device); yv = torch.from_numpy(y_te).to(device)


# === Test A: shallow MLP on DQN's exact input ===
print('\n=== Test A: 3-layer MLP on flattened DQN input ===')
torch.manual_seed(0)
mlp = nn.Sequential(
    nn.Linear(5120, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 25),
).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
best_a = 0
for ep in range(EP):
    mlp.train(); perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 64):
        idx = perm[i:i+64]
        loss = F.cross_entropy(mlp(Xt[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        te = (mlp(Xv).argmax(1) == yv).float().mean().item() * 100
    if te > best_a: best_a = te
print(f'  best A = {best_a:.2f}')


# === Test C: DQN architecture but FLAT readout (keep all 8D, no invariants) ===
print('\n=== Test C: DQN arch with FLAT readout (no DQ-invariant compression) ===')
def qmul(p, q):
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], dim=-1)

def dq_mul(a, b):
    ar, ad = a[..., :4], a[..., 4:]
    br, bd = b[..., :4], b[..., 4:]
    return torch.cat([qmul(ar, br), qmul(ar, bd) + qmul(ad, br)], dim=-1)

ADJ = np.zeros((B, B), dtype=np.float32)
for i in range(B):
    for j in range(B):
        if i == j: continue
        if {BONES[i][0],BONES[i][1]} & {BONES[j][0],BONES[j][1]}: ADJ[i,j] = 1
ADJ_T = torch.from_numpy(ADJ).to(device)

class DQLinear(nn.Module):
    def __init__(self, h_in, h_out):
        super().__init__()
        self.W_dq = nn.Parameter(torch.randn(h_in, h_out, 8) * 0.05)
        with torch.no_grad():
            self.W_dq[..., 0] = 1.0
        self.W_mix = nn.Parameter(torch.randn(h_in, h_out) * 0.1)
        self.b = nn.Parameter(torch.zeros(h_out, 8))
    def forward(self, x):
        outs = []
        for o in range(self.W_dq.shape[1]):
            W_o = self.W_dq[:, o, :]
            mixed = dq_mul(x, W_o)
            w = F.softmax(self.W_mix[:, o], dim=0)
            agg = (mixed * w.view(*([1]*(mixed.ndim-2)), -1, 1)).sum(dim=-2)
            outs.append(agg)
        return torch.stack(outs, dim=-2) + self.b

class SkeletalDQConv(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lin_self = DQLinear(h, h)
        self.lin_neigh = DQLinear(h, h)
    def forward(self, x):
        neigh = torch.einsum('ij,btjho->btiho', ADJ_T, x)
        return self.lin_self(x) + self.lin_neigh(neigh)

class DQN_FlatReadout(nn.Module):
    def __init__(self, h=8, n_layers=2):
        super().__init__()
        self.lift = DQLinear(1, h)
        self.layers = nn.ModuleList([SkeletalDQConv(h) for _ in range(n_layers)])
        d = B * h * 8
        self.classifier = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 25),
        )
        self.h = h
    def forward(self, x):
        x = x.unsqueeze(-2)
        x = self.lift(x)
        for L in self.layers: x = L(x)
        x = x.mean(dim=1)  # time pool
        return self.classifier(x.flatten(1))

# DQN takes (batch, T, B, 8) reshaped from flat
X_tr_dq = torch.from_numpy(X_tr.astype(np.float32).reshape(-1, T_FIXED, B, 8)).to(device)
X_te_dq = torch.from_numpy(X_te.astype(np.float32).reshape(-1, T_FIXED, B, 8)).to(device)

torch.manual_seed(0)
m = DQN_FlatReadout(h=8, n_layers=2).to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
best_c = 0
for ep in range(EP):
    m.train(); perm = torch.randperm(len(X_tr_dq))
    for i in range(0, len(X_tr_dq), BS):
        idx = perm[i:i+BS]
        loss = F.cross_entropy(m(X_tr_dq[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        # eval in chunks
        preds = []
        for i in range(0, len(X_te_dq), BS):
            preds.append(m(X_te_dq[i:i+BS]).argmax(1))
        preds = torch.cat(preds)
        te = (preds == yv).float().mean().item() * 100
    if te > best_c: best_c = te
print(f'  best C = {best_c:.2f}')


# === Test B: Skip DQ algebra entirely, just per-bone MLP ===
print('\n=== Test B: per-bone MLP (no algebra), then time-pool, then classifier ===')
class PerBoneNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.bone_mlp = nn.Sequential(nn.Linear(8, h), nn.GELU(), nn.Linear(h, h))
        d = B * h
        self.classifier = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 25),
        )
    def forward(self, x):
        # x: (batch, T, B, 8)
        h = self.bone_mlp(x)  # (batch, T, B, h)
        h = h.mean(dim=1)     # time pool
        return self.classifier(h.flatten(1))

torch.manual_seed(0)
m = PerBoneNet(h=64).to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
best_b = 0
for ep in range(EP):
    m.train(); perm = torch.randperm(len(X_tr_dq))
    for i in range(0, len(X_tr_dq), BS):
        idx = perm[i:i+BS]
        loss = F.cross_entropy(m(X_tr_dq[idx]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X_te_dq), BS):
            preds.append(m(X_te_dq[i:i+BS]).argmax(1))
        preds = torch.cat(preds)
        te = (preds == yv).float().mean().item() * 100
    if te > best_b: best_b = te
print(f'  best B = {best_b:.2f}')

print(f'\n=== SUMMARY ===')
print(f'Diagnostic shallow MLP on dual_quat (prior):  64.11')
print(f'A) plain MLP on DQN-flattened input:          {best_a:.2f}')
print(f'B) per-bone MLP + time-pool + classifier:     {best_b:.2f}')
print(f'C) DQN arch with FLAT readout (no invariants): {best_c:.2f}')
print(f'D) original DQN (DQ-invariant readout):       38.59')
