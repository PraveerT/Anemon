"""Step 2: Test multiple aux signals from MediaPipe landmarks via shallow MLP.
Find which signal carries gesture-discriminative info beyond chance.
"""
import os, re, numpy as np
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
print(f'train annot: {len(train_lbl)}, test annot: {len(test_lbl)}')
print('loading landmarks...')
sk = dict(np.load(SK, allow_pickle=False))
print(f'{len(sk)} samples')

def fillnan(arr):
    """Forward + backfill NaN frames."""
    valid = np.isfinite(arr[..., 0]).all(axis=-1)
    last = None
    out = arr.copy()
    for t in range(out.shape[0]):
        if valid[t]: last = out[t]
        elif last is not None: out[t] = last
    for t in range(out.shape[0]):
        if not np.isfinite(out[t]).all():
            for t2 in range(t+1, out.shape[0]):
                if valid[t2]: out[t] = out[t2]; break
            else: out[t] = 0
    return out

def resample(x, T_target):
    if len(x) < 2: return None
    idx = np.linspace(0, len(x)-1, T_target).astype(np.int64)
    return x[idx]

# Feature extractors — each takes (T, 21, 3) filled landmarks, returns (T_out, D)
TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
PIP  = [3, 7, 11, 15, 19]   # one joint up

def feat_lm(lm):           return lm.reshape(lm.shape[0], -1)                  # (T, 63)
def feat_vel(lm):          return np.diff(lm, axis=0).reshape(lm.shape[0]-1, -1)  # (T-1, 63)
def feat_acc(lm):          return np.diff(lm, n=2, axis=0).reshape(lm.shape[0]-2, -1)  # (T-2, 63)
def feat_jerk(lm):         return np.diff(lm, n=3, axis=0).reshape(lm.shape[0]-3, -1)  # (T-3, 63)
def feat_speed(lm):
    v = np.diff(lm, axis=0)
    return np.linalg.norm(v, axis=-1)  # (T-1, 21)
def feat_centroid(lm):
    c = lm.mean(axis=1)  # (T, 3) per-frame centroid
    return c
def feat_centroid_vel(lm):
    return np.diff(lm.mean(axis=1), axis=0)  # (T-1, 3)
def feat_bbox(lm):
    mn = lm.min(axis=1); mx = lm.max(axis=1)
    extent = mx - mn  # (T, 3)
    return extent
def feat_bbox_area(lm):
    e = feat_bbox(lm)
    return (e[:, 0] * e[:, 1])[:, None]  # (T, 1)
def feat_finger_spread(lm):
    # pairwise distances among 5 fingertips
    tips = lm[:, TIPS, :]  # (T, 5, 3)
    diffs = tips[:, :, None, :] - tips[:, None, :, :]  # (T, 5, 5, 3)
    d = np.linalg.norm(diffs, axis=-1)  # (T, 5, 5)
    iu = np.triu_indices(5, k=1)
    return d[:, iu[0], iu[1]]  # (T, 10)
def feat_finger_angles(lm):
    """3 angles per finger (between consecutive segments)."""
    out = []
    for f, idxs in enumerate([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]):
        # vectors between consecutive joints
        pts = lm[:, [0]+idxs, :]  # (T, 5, 3): wrist + 4 finger joints
        v = np.diff(pts, axis=1)  # (T, 4, 3)
        # angles between consecutive vectors
        for i in range(3):
            a = v[:, i, :]; b = v[:, i+1, :]
            cos = (a*b).sum(-1) / (np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1) + 1e-7)
            out.append(np.clip(cos, -1, 1))
    return np.stack(out, axis=1)  # (T, 15)
def feat_palm_normal(lm):
    """Normal vector of palm plane: cross(wrist->index_mcp, wrist->pinky_mcp)."""
    v1 = lm[:, 5, :] - lm[:, 0, :]
    v2 = lm[:, 17, :] - lm[:, 0, :]
    n = np.cross(v1, v2)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-7)
    return n  # (T, 3)
def feat_palm_normal_vel(lm):
    return np.diff(feat_palm_normal(lm), axis=0)
def feat_combined(lm):
    """LM + velocity (the obvious motion super-set)."""
    v = np.diff(lm, axis=0)
    v_padded = np.concatenate([v, v[-1:]], axis=0)
    return np.concatenate([lm.reshape(lm.shape[0],-1), v_padded.reshape(v_padded.shape[0],-1)], axis=1)  # (T, 126)

EXTRACTORS = {
    'landmarks_raw':       feat_lm,
    'velocity':            feat_vel,
    'acceleration':        feat_acc,
    'jerk':                feat_jerk,
    'tip_speeds':          feat_speed,
    'centroid_pos':        feat_centroid,
    'centroid_velocity':   feat_centroid_vel,
    'bbox_extent':         feat_bbox,
    'bbox_area':           feat_bbox_area,
    'fingertip_spread':    feat_finger_spread,
    'finger_curl_angles':  feat_finger_angles,
    'palm_normal':         feat_palm_normal,
    'palm_normal_vel':     feat_palm_normal_vel,
    'lm+velocity':         feat_combined,
}

def build_xy(extractor):
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for k, lm_raw in sk.items():
        if lm_raw.shape[0] < 4: continue
        lm = fillnan(lm_raw)
        feat = extractor(lm)
        if feat is None or feat.shape[0] < 2: continue
        feat_r = resample(feat, T_FIXED)  # (T_FIXED, D)
        x = feat_r.reshape(-1).astype(np.float32)
        if k in train_lbl: X_tr.append(x); y_tr.append(train_lbl[k])
        elif k in test_lbl: X_te.append(x); y_te.append(test_lbl[k])
    X_tr = np.array(X_tr, dtype=np.float32); y_tr = np.array(y_tr, dtype=np.int64)
    X_te = np.array(X_te, dtype=np.float32); y_te = np.array(y_te, dtype=np.int64)
    # Standardize
    mean = X_tr.mean(0); std = X_tr.std(0) + 1e-7
    X_tr = (X_tr - mean) / std; X_te = (X_te - mean) / std
    return X_tr, y_tr, X_te, y_te

def shallow_test(X_tr, y_tr, X_te, y_te, ep_max=80):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = X_tr.shape[1]
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
    return best_te, D

print(f'\n{"signal":<24} {"dim":>6} {"best_te %":>10}  vs_chance')
print('-' * 60)
for name, ext in EXTRACTORS.items():
    try:
        Xtr, ytr, Xte, yte = build_xy(ext)
        acc, D = shallow_test(Xtr, ytr, Xte, yte)
        print(f'{name:<24} {Xtr.shape[1]:>6}  {acc:>8.2f}    {acc/4.0:.1f}x', flush=True)
    except Exception as e:
        print(f'{name:<24} ERROR: {e}', flush=True)

# Reference: prior QCC test was 9.96 (per-finger Kabsch)
print(f'{"per_finger_Kabsch (ref)":<24} {640:>6}  {9.96:>8.2f}    {9.96/4.0:.1f}x  (from prior)')
print('chance = 4.00%')
