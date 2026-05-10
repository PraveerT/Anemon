"""Diagnostic: do COMPOSITIONALITY (chain prefix products) and GEODESIC structure
(anchor distances on S^3) carry information BEYOND raw bone quaternions?

If yes: the novel layer concept is justified.
If no: the math constructs are decorative; no publishable contribution.

Same shallow MLP framework as steps 1, 2, 5.
"""
import os, re, time, numpy as np
import torch, torch.nn as nn

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
T_FIXED = 32

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

CHAINS = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]]
B = 20
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

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

def encode_bone_dir(lm):
    out = np.zeros((lm.shape[0], B, 4), dtype=np.float32)
    for b, (p, c) in enumerate(BONES):
        out[:, b, :] = vec_to_quat_np(lm[:, c, :] - lm[:, p, :])
    return out

def qmul_np(p, q):
    pw,px,py,pz = p[...,0],p[...,1],p[...,2],p[...,3]
    qw,qx,qy,qz = q[...,0],q[...,1],q[...,2],q[...,3]
    return np.stack([pw*qw-px*qx-py*qy-pz*qz, pw*qx+px*qw+py*qz-pz*qy,
                      pw*qy-px*qz+py*qw+pz*qx, pw*qz+px*qy-py*qx+pz*qw], axis=-1)

def chain_prefix(q):
    """q: (T, B, 4). Compute prefix product per chain. Returns (T, B, 4)."""
    out = q.copy()
    for chain in CHAINS:
        cum = q[:, chain[0], :]
        out[:, chain[0], :] = cum
        for b in chain[1:]:
            cum = qmul_np(cum, q[:, b, :])
            n = np.linalg.norm(cum, axis=-1, keepdims=True) + 1e-9
            cum = cum / n
            out[:, b, :] = cum
    return out

def geodesic_anchor_features(q, anchors):
    """q: (T, B, 4), anchors: (K, 4). Returns (T, B, K) geodesic distances."""
    cos = np.einsum('tbd,kd->tbk', q, anchors)
    cos = np.abs(cos).clip(0, 1 - 1e-6)
    return np.arccos(cos)

# Generate fixed random anchors (for diagnostic — equivalent to learnable but fixed)
rng = np.random.RandomState(0)
ANCHORS = rng.randn(24, 4).astype(np.float32)
ANCHORS = ANCHORS / (np.linalg.norm(ANCHORS, axis=-1, keepdims=True) + 1e-9)

print('precomputing features...')
def resample(x, T):
    idx = np.linspace(0, len(x)-1, T).astype(np.int64)
    return x[idx]

# Storage
features = {}  # key -> dict of {feat_name: array}
for k, lm_raw in sk.items():
    if lm_raw.shape[0] < 4: continue
    lm = fillnan(lm_raw)
    q_raw = encode_bone_dir(lm)         # (T, 20, 4)
    q_chain = chain_prefix(q_raw)        # (T, 20, 4) compositional
    geo_raw = geodesic_anchor_features(q_raw, ANCHORS)  # (T, 20, 24)
    geo_chain = geodesic_anchor_features(q_chain, ANCHORS)  # (T, 20, 24)
    feats = {
        'q_raw':     resample(q_raw, T_FIXED).reshape(T_FIXED, -1),       # (32, 80)
        'q_chain':   resample(q_chain, T_FIXED).reshape(T_FIXED, -1),     # (32, 80)
        'geo_raw':   resample(geo_raw, T_FIXED).reshape(T_FIXED, -1),     # (32, 480)
        'geo_chain': resample(geo_chain, T_FIXED).reshape(T_FIXED, -1),   # (32, 480)
    }
    features[k] = feats
print(f'{len(features)} samples')


def shallow(X_tr, y_tr, X_te, y_te, ep_max=80):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = X_tr.shape[1]
    if D == 0: return 0.0
    mlp = nn.Sequential(
        nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 25),
    ).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X_tr).to(device); yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_te).to(device); yv = torch.from_numpy(y_te).to(device)
    best_te = 0
    for ep in range(ep_max):
        mlp.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]
            loss = nn.functional.cross_entropy(mlp(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            te_acc = (mlp(Xv).argmax(1) == yv).float().mean().item() * 100
        if te_acc > best_te: best_te = te_acc
    return best_te

def build_xy(feat_keys):
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for k, fd in features.items():
        x = np.concatenate([fd[fk].reshape(-1) for fk in feat_keys]).astype(np.float32)
        if not np.isfinite(x).all(): x = np.nan_to_num(x)
        if k in train_lbl: X_tr.append(x); y_tr.append(train_lbl[k])
        elif k in test_lbl: X_te.append(x); y_te.append(test_lbl[k])
    X_tr = np.array(X_tr, dtype=np.float32); y_tr = np.array(y_tr, dtype=np.int64)
    X_te = np.array(X_te, dtype=np.float32); y_te = np.array(y_te, dtype=np.int64)
    mean = X_tr.mean(0); std = X_tr.std(0) + 1e-7
    return (X_tr-mean)/std, y_tr, (X_te-mean)/std, y_te

CONFIGS = [
    ('q_raw',                       ['q_raw']),
    ('q_chain (compositional)',     ['q_chain']),
    ('q_raw + q_chain',             ['q_raw', 'q_chain']),
    ('geo_raw (anchor dists)',      ['geo_raw']),
    ('geo_chain',                   ['geo_chain']),
    ('q_raw + geo_raw',             ['q_raw', 'geo_raw']),
    ('q_raw + geo_chain',           ['q_raw', 'geo_chain']),
    ('q_raw + q_chain + geo_raw + geo_chain', ['q_raw', 'q_chain', 'geo_raw', 'geo_chain']),
]

print(f'\n{"config":<55} {"dim":>6} {"acc%":>7}  vs_chance')
print('-'*80)
for name, keys in CONFIGS:
    t0 = time.time()
    Xtr, ytr, Xte, yte = build_xy(keys)
    acc = shallow(Xtr, ytr, Xte, yte)
    print(f'{name:<55} {Xtr.shape[1]:>6}  {acc:>5.2f}    {acc/4.0:.1f}x  ({time.time()-t0:.0f}s)', flush=True)
print('\n--- references ---')
print(f'{"landmarks_raw (xyz)":<55} {2016:>6}  {62.03:>5.2f}    15.5x')
print(f'{"finger_curl_angles":<55} {480:>6}  {38.38:>5.2f}    9.6x')
print(f'{"bone_direction_quaternion (=q_raw, prior result)":<55} {2560:>6}  {59.34:>5.2f}    14.8x')
