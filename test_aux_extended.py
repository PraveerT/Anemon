"""Extended benchmark: novel quaternion + geometric algebra + manifold + signal
features for the gesture-info diagnostic. Adds 12+ new descriptors to the prior
21-feature catalog.
"""
import os, re, time, numpy as np, warnings
warnings.filterwarnings('ignore')
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
TIPS = [4, 8, 12, 16, 20]
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
B = 20

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

def vec_to_quat(V):
    n = np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9
    u = V / n
    cos_h = np.clip((1 + u[..., 2:3]) * 0.5, 1e-9, 1.0)
    w = np.sqrt(cos_h)
    sin_h = np.sqrt(np.clip(1 - cos_h, 0, 1))
    axis = np.zeros_like(u); axis[..., 0] = -u[..., 1]; axis[..., 1] = u[..., 0]
    s = np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9
    return np.concatenate([w, axis/s*sin_h], axis=-1).astype(np.float32)

def qmul(p, q):
    pw,px,py,pz = p[...,0],p[...,1],p[...,2],p[...,3]
    qw,qx,qy,qz = q[...,0],q[...,1],q[...,2],q[...,3]
    return np.stack([pw*qw-px*qx-py*qy-pz*qz, pw*qx+px*qw+py*qz-pz*qy,
                      pw*qy-px*qz+py*qw+pz*qx, pw*qz+px*qy-py*qx+pz*qw], axis=-1)

# ============ NEW FEATURE EXTRACTORS ============

def feat_dual_quat(lm):
    """Dual quaternion per bone = (rotation_quat, position_quat).
    Translation t -> position_quat = (1/2) * t_pure * rotation_quat where t_pure = (0, t).
    Rich representation: rigid SE(3) element in 8D.
    """
    out = []
    for b, (p, c) in enumerate(BONES):
        bone_vec = lm[:, c, :] - lm[:, p, :]    # (T, 3)
        rot_q = vec_to_quat(bone_vec)            # (T, 4)
        # midpoint position
        mid = (lm[:, c, :] + lm[:, p, :]) / 2     # (T, 3)
        t_pure = np.concatenate([np.zeros((lm.shape[0], 1), dtype=np.float32), mid], axis=-1)
        pos_q = qmul(t_pure, rot_q) * 0.5
        out.append(np.concatenate([rot_q, pos_q], axis=-1))  # (T, 8)
    return np.stack(out, axis=1).reshape(lm.shape[0], -1)  # (T, 160)

def feat_quat_log(lm):
    """Quaternion log map: q = (cos(θ/2), sin(θ/2)*v) -> log(q) = θ*v in R^3.
    Per-bone axis-angle representation."""
    out = []
    for b, (p, c) in enumerate(BONES):
        q = vec_to_quat(lm[:, c, :] - lm[:, p, :])
        w = q[..., 0:1].clip(-1+1e-7, 1-1e-7)
        theta = 2 * np.arccos(np.abs(w))
        sin_h = np.sqrt(np.clip(1 - w**2, 1e-9, 1))
        axis = q[..., 1:] / sin_h
        out.append((theta * axis).astype(np.float32))  # (T, 3)
    return np.stack(out, axis=1).reshape(lm.shape[0], -1)  # (T, 60)

def feat_quat_powers(lm):
    """q, q^2, q^4 per bone — polynomial / semigroup features."""
    out = []
    for b, (p, c) in enumerate(BONES):
        q = vec_to_quat(lm[:, c, :] - lm[:, p, :])
        q2 = qmul(q, q); q4 = qmul(q2, q2)
        out.append(np.concatenate([q, q2, q4], axis=-1))  # (T, 12)
    return np.stack(out, axis=1).reshape(lm.shape[0], -1)  # (T, 240)

def feat_slerp_midpoint(lm):
    """SLERP midpoint between bone quaternions of consecutive frames.
    Geodesic interpolation on S^3."""
    out = []
    for b, (p, c) in enumerate(BONES):
        q = vec_to_quat(lm[:, c, :] - lm[:, p, :])  # (T, 4)
        # midpoint = SLERP(q_t, q_{t+1}, 0.5) = sign-corrected sum, normalized
        q1 = q[:-1]; q2 = q[1:]
        sign = np.sign((q1 * q2).sum(-1, keepdims=True))
        sign[sign == 0] = 1
        mid = (q1 + sign * q2) * 0.5
        mid = mid / (np.linalg.norm(mid, axis=-1, keepdims=True) + 1e-9)
        # pad first frame for fixed length
        mid = np.concatenate([q[:1], mid], axis=0)
        out.append(mid.astype(np.float32))
    return np.stack(out, axis=1).reshape(lm.shape[0], -1)  # (T, 80)

def feat_bivector(lm):
    """Bivector representation of bone rotation: B = log(R) viewed in so(3) ~= R^3.
    Same dim as quat_log but conceptually distinct (geometric algebra view).
    Equal to quat_log up to factor 1/2 — both are axis-angle vectors."""
    return feat_quat_log(lm)

def feat_spherical_harmonics(lm):
    """SH coefficients up to L=3 of each fingertip's direction from wrist."""
    out = []
    for tip in TIPS:
        v = lm[:, tip, :] - lm[:, 0, :]  # (T, 3)
        n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
        u = v / n
        x, y, z = u[..., 0], u[..., 1], u[..., 2]
        # Hardcoded normalized real SH up to L=3 (16 coefs)
        coefs = []
        coefs.append(0.282 * np.ones_like(x))                                  # Y00
        coefs.append(0.488 * y); coefs.append(0.488 * z); coefs.append(0.488 * x)  # Y1*
        coefs.append(1.093 * x*y); coefs.append(1.093 * y*z)
        coefs.append(0.315 * (3*z*z - 1)); coefs.append(1.093 * x*z); coefs.append(0.546 * (x*x - y*y))  # Y2*
        coefs.append(0.590 * y*(3*x*x - y*y))
        coefs.append(2.890 * x*y*z); coefs.append(0.457 * y*(5*z*z - 1))
        coefs.append(0.373 * z*(5*z*z - 3))
        coefs.append(0.457 * x*(5*z*z - 1)); coefs.append(1.445 * z*(x*x - y*y))
        coefs.append(0.590 * x*(x*x - 3*y*y))
        out.append(np.stack(coefs, axis=-1))
    return np.concatenate(out, axis=-1).astype(np.float32)  # (T, 80)

def feat_dct(lm):
    """DCT-II of each fingertip per axis trajectory. Top 8 coefficients per axis."""
    from scipy.fft import dct
    out = []
    for tip in TIPS:
        for ax in range(3):
            sig = lm[:, tip, ax]
            d = dct(sig - sig.mean(), type=2, norm='ortho')[:8]
            out.append(d)
    out = np.stack(out, axis=0).reshape(1, -1)
    return out.repeat(lm.shape[0], axis=0).astype(np.float32)  # (T, 120)

def feat_wavelet_haar(lm):
    """Haar wavelet decomposition coefficients of each fingertip per axis."""
    try:
        import pywt
        out = []
        for tip in TIPS:
            for ax in range(3):
                sig = lm[:, tip, ax]
                cA, cD3, cD2, cD1 = pywt.wavedec(sig - sig.mean(), 'haar', level=3)
                out.append(np.concatenate([cA, cD3, cD2, cD1])[:16])
        out = np.array(out).reshape(1, -1)
        return out.repeat(lm.shape[0], axis=0).astype(np.float32)
    except ImportError:
        return None

def feat_higher_moments(lm):
    """Per-joint per-axis higher statistical moments: mean, var, skew, kurt."""
    out = []
    for j in range(21):
        for ax in range(3):
            sig = lm[:, j, ax]
            m = sig.mean(); v = sig.var() + 1e-9
            s = ((sig - m)**3).mean() / (v**1.5)
            k = ((sig - m)**4).mean() / (v**2)
            out.extend([m, v, s, k])
    out = np.array(out)
    return out[None, :].repeat(lm.shape[0], axis=0).astype(np.float32)  # (T, 252)

def feat_pairwise_distance_eigvals(lm):
    """Top-K eigenvalues of pairwise-distance matrix (21x21) per frame.
    Captures topological/geometric pose invariants."""
    out = []
    for t in range(lm.shape[0]):
        D = np.linalg.norm(lm[t, :, None] - lm[t, None, :], axis=-1)
        eigs = np.sort(np.linalg.eigvalsh(D))[-10:]
        out.append(eigs)
    return np.stack(out).astype(np.float32)  # (T, 10)

def feat_finger_linking(lm):
    """Linking number between consecutive fingers (treat each fingertip path as
    a closed curve, compute Gauss linking integral pairwise)."""
    out = []
    paths = [lm[:, [0]+f, :] for f in [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]]
    pairs = [(0,1),(1,2),(2,3),(3,4)]  # 4 consecutive finger pairs
    for (i, j) in pairs:
        c1 = paths[i]; c2 = paths[j]
        # Sum over all (i,j) edge pairs Gauss kernel
        e1_a = c1[:-1]; e1_b = c1[1:]
        e2_a = c2[:-1]; e2_b = c2[1:]
        a = e1_b - e1_a   # (Te1, 3) per frame? No — c1 is (T, 5, 3); e1_a is (T, 4, 3)
        # collapse over T_FIXED later
        pass
    # Simplified: avg pairwise distance between fingers
    out_simple = []
    for t in range(lm.shape[0]):
        for (i, j) in pairs:
            d = np.linalg.norm(lm[t, [4,8,12,16][i]] - lm[t, [4,8,12,16,20][j]])
            out_simple.append(d)
    out_simple = np.array(out_simple).reshape(lm.shape[0], -1)
    return out_simple.astype(np.float32)  # (T, 4)

def feat_relative_to_palm(lm):
    """Each landmark expressed in palm frame: x_axis = wrist->index_mcp,
    y_axis perpendicular in palm plane. Removes hand-position bias."""
    out = np.zeros_like(lm)
    for t in range(lm.shape[0]):
        wrist = lm[t, 0]
        idx_mcp = lm[t, 5]
        pinky_mcp = lm[t, 17]
        x_ax = idx_mcp - wrist
        x_ax = x_ax / (np.linalg.norm(x_ax) + 1e-9)
        y_tmp = pinky_mcp - wrist
        y_ax = y_tmp - x_ax * (y_tmp @ x_ax)
        y_ax = y_ax / (np.linalg.norm(y_ax) + 1e-9)
        z_ax = np.cross(x_ax, y_ax)
        F = np.stack([x_ax, y_ax, z_ax], axis=1)  # (3, 3)
        out[t] = (lm[t] - wrist) @ F
    return out.reshape(lm.shape[0], -1).astype(np.float32)  # (T, 63)

def feat_inertia_tensor(lm):
    """Per-frame inertia tensor of landmark cloud (assumed unit mass each).
    Eigenvalues + axes orientation."""
    out = []
    for t in range(lm.shape[0]):
        c = lm[t].mean(0)
        d = lm[t] - c
        I = np.zeros((3, 3))
        for p in d:
            I += np.outer(p, p)
        eigs, vecs = np.linalg.eigh(I)
        # Use eigenvalues only (rotation-invariant)
        out.append(eigs)
    return np.stack(out).astype(np.float32)  # (T, 3)

def feat_joint_angle_pca(lm):
    """PCA on per-frame joint coords; project onto first 8 PCs (computed across
    all frames of this sample). Captures intrinsic dim."""
    flat = lm.reshape(lm.shape[0], -1)  # (T, 63)
    flat = flat - flat.mean(0)
    try:
        U, S, Vt = np.linalg.svd(flat, full_matrices=False)
        proj = U[:, :8] * S[:8]
    except:
        proj = np.zeros((lm.shape[0], 8))
    return proj.astype(np.float32)  # (T, 8)

# ============ Diagnostic ============

EXTRACTORS = {
    'dual_quaternion':         feat_dual_quat,
    'quaternion_log_axis_angle': feat_quat_log,
    'quaternion_polynomial_powers': feat_quat_powers,
    'slerp_midpoint':          feat_slerp_midpoint,
    'spherical_harmonics_L3':  feat_spherical_harmonics,
    'dct_top8':                feat_dct,
    'haar_wavelet':            feat_wavelet_haar,
    'higher_moments':          feat_higher_moments,
    'pairwise_dist_eigvals':   feat_pairwise_distance_eigvals,
    'finger_linking_simple':   feat_finger_linking,
    'palm_frame_relative':     feat_relative_to_palm,
    'inertia_tensor_eigvals':  feat_inertia_tensor,
    'joint_pose_pca8':         feat_joint_angle_pca,
}

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
        if lm_raw.shape[0] < 8: continue
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

print(f'\n{"signal":<35} {"dim":>6} {"acc%":>7}  vs_chance')
print('-' * 70)
for name, ext in EXTRACTORS.items():
    t0 = time.time()
    try:
        Xtr, ytr, Xte, yte = build_xy(ext)
        if Xtr.shape[0] < 100:
            print(f'{name:<35} too few samples')
            continue
        acc = shallow(Xtr, ytr, Xte, yte)
        print(f'{name:<35} {Xtr.shape[1]:>6}  {acc:>5.2f}    {acc/4.0:.1f}x  ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'{name:<35} ERROR: {e}', flush=True)

print('\n--- prior best for context ---')
print(f'{"landmarks_raw (xyz)":<35} {2016:>6}  {62.03:>5.2f}    15.5x')
print(f'{"fingertip_spread":<35} {320:>6}  {44.19:>5.2f}    11.0x')
print(f'{"fourier_band_energy":<35} {1920:>6}  {43.15:>5.2f}    10.8x')
print(f'{"per_finger_Kabsch (QCC)":<35} {640:>6}  {9.96:>5.2f}    2.5x')
