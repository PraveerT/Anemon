"""QEAN step 0: diagnose information content of per-bone quaternion encodings.
Two encodings:
  (1) bone direction quaternion (rotation from canonical e_z=(0,0,1) to bone direction)
  (2) joint-relative quaternion (rotation from parent bone's direction to this bone's)
Both encodings are SO(3)-equivariant by construction.

Compare vs landmarks_raw (62.03) and finger_curl_angles (38.38).
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

def resample(x, T):
    idx = np.linspace(0, len(x)-1, T).astype(np.int64)
    return x[idx]

# Hand skeleton: 20 bones, each (parent, child)
BONES = [
    (0,1),(1,2),(2,3),(3,4),         # thumb chain
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
]
# Parent bone of each bone (None for the 5 root-attached bones)
PARENT_BONE = {0:None, 1:0, 2:1, 3:2, 4:None, 5:4, 6:5, 7:6, 8:None, 9:8, 10:9, 11:10,
               12:None, 13:12, 14:13, 15:14, 16:None, 17:16, 18:17, 19:18}

def vec_to_quat(v):
    """Quaternion that rotates (0,0,1) onto v. Returns [w,x,y,z]."""
    n = np.linalg.norm(v) + 1e-9
    u = v / n
    z = np.array([0., 0., 1.])
    cos_h = (1 + u.dot(z)) * 0.5
    if cos_h < 1e-9:  # 180-deg case
        # any axis perpendicular to z
        return np.array([0., 1., 0., 0.])
    w = np.sqrt(cos_h)
    axis = np.cross(z, u)
    s = np.linalg.norm(axis) + 1e-9
    sin_h = np.sqrt(max(0, 1 - cos_h))
    return np.array([w, *(axis / s * sin_h)])

def vec_to_quat_batch(V):
    """V: (..., 3). Returns (..., 4) [w,x,y,z]."""
    out = np.zeros(V.shape[:-1] + (4,), dtype=np.float32)
    flat_V = V.reshape(-1, 3)
    flat_out = out.reshape(-1, 4)
    for i, v in enumerate(flat_V):
        flat_out[i] = vec_to_quat(v)
    return out

def quat_mul(q1, q2):
    """Hamilton product of quaternions [w,x,y,z]."""
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def encode_bone_dir(lm):
    """Encoding 1: per-bone direction quaternion. Returns (T, 20, 4)."""
    out = np.zeros((lm.shape[0], len(BONES), 4), dtype=np.float32)
    for b, (p, c) in enumerate(BONES):
        bone_vecs = lm[:, c, :] - lm[:, p, :]  # (T, 3)
        out[:, b, :] = vec_to_quat_batch(bone_vecs)
    return out

def encode_bone_joint_rel(lm):
    """Encoding 2: joint-relative quaternion = q_child * q_parent^-1 between each
    bone and its parent. For root-attached bones, just bone direction quat."""
    abs_q = encode_bone_dir(lm)  # (T, 20, 4)
    out = np.zeros_like(abs_q)
    for b in range(len(BONES)):
        pb = PARENT_BONE[b]
        if pb is None:
            out[:, b, :] = abs_q[:, b, :]
        else:
            for t in range(lm.shape[0]):
                out[t, b, :] = quat_mul(abs_q[t, b, :], quat_conj(abs_q[t, pb, :]))
    return out

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

def build_xy(extractor):
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for k, lm_raw in sk.items():
        if lm_raw.shape[0] < 4: continue
        lm = fillnan(lm_raw)
        try: feat = extractor(lm)
        except Exception as e: continue
        if feat is None or feat.shape[0] < 2: continue
        feat_r = resample(feat, T_FIXED)
        x = feat_r.reshape(-1).astype(np.float32)
        if not np.isfinite(x).all(): x = np.nan_to_num(x)
        if k in train_lbl: X_tr.append(x); y_tr.append(train_lbl[k])
        elif k in test_lbl: X_te.append(x); y_te.append(test_lbl[k])
    X_tr = np.array(X_tr, dtype=np.float32); y_tr = np.array(y_tr, dtype=np.int64)
    X_te = np.array(X_te, dtype=np.float32); y_te = np.array(y_te, dtype=np.int64)
    mean = X_tr.mean(0); std = X_tr.std(0) + 1e-7
    return (X_tr-mean)/std, y_tr, (X_te-mean)/std, y_te

print(f'\n{"encoding":<28} {"dim":>6} {"acc%":>7}  vs_chance')
print('-'*60)

specs = [
    ('bone_direction_quaternion', encode_bone_dir),
    ('joint_relative_quaternion', encode_bone_joint_rel),
]

for name, ext in specs:
    t0 = time.time()
    Xtr, ytr, Xte, yte = build_xy(ext)
    acc = shallow(Xtr, ytr, Xte, yte)
    print(f'{name:<28} {Xtr.shape[1]:>6}  {acc:>5.2f}    {acc/4.0:.1f}x  ({time.time()-t0:.0f}s)', flush=True)

print('\n--- references ---')
print(f'{"landmarks_raw":<28} {2016:>6}  {62.03:>5.2f}    15.5x')
print(f'{"finger_curl_angles":<28} {480:>6}  {38.38:>5.2f}    9.6x')
print(f'{"per_finger_Kabsch (QCC)":<28} {640:>6}  {9.96:>5.2f}    2.5x')
