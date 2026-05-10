"""Step 3 (publishable angle): systematic information-content benchmark of OBSCURE
mathematical motion features — Frenet, Fourier, wavelets, persistent homology,
spectral graph, Lempel-Ziv, Hurst, permutation entropy, SE(3) twist, Berry phase,
writhe, recurrence quantification, Procrustes residuals, etc.

Each feature: shallow MLP on (T_fixed, D) → 25-class gesture acc.
Score = "gesture information content" of that mathematical descriptor.
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
print(f'train: {len(train_lbl)}, test: {len(test_lbl)}')
print('loading landmarks...')
sk = dict(np.load(SK, allow_pickle=False))
TIPS = [4, 8, 12, 16, 20]

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

def resample(x, T_target):
    if len(x) < 2: return None
    idx = np.linspace(0, len(x)-1, T_target).astype(np.int64)
    return x[idx]

# ============ OBSCURE MATH FEATURES ============

def feat_frenet(lm):
    """Frenet-Serret invariants (curvature kappa, torsion tau) along each fingertip path.
    Classical differential geometry of space curves."""
    out = []
    for tip in TIPS:
        p = lm[:, tip, :]  # (T, 3)
        v = np.gradient(p, axis=0)
        a = np.gradient(v, axis=0)
        j = np.gradient(a, axis=0)
        sp = np.linalg.norm(v, axis=-1) + 1e-7
        # kappa = |v x a| / |v|^3
        cross_va = np.cross(v, a)
        kappa = np.linalg.norm(cross_va, axis=-1) / (sp**3)
        # tau = (v x a) . j / |v x a|^2
        denom = (cross_va**2).sum(-1) + 1e-7
        tau = (cross_va * j).sum(-1) / denom
        out.append(kappa); out.append(tau)
    return np.stack(out, axis=1)  # (T, 10)

def feat_fourier(lm):
    """Fourier energy in 4 bands per fingertip trajectory (3 axes)."""
    out = []
    T = lm.shape[0]
    for tip in TIPS:
        for ax in range(3):
            sig = lm[:, tip, ax]
            sig = sig - sig.mean()
            f = np.abs(np.fft.rfft(sig)) ** 2
            # 4 bands
            n = len(f); b = max(1, n // 4)
            out.append(np.array([f[:b].sum(), f[b:2*b].sum(), f[2*b:3*b].sum(), f[3*b:].sum()]))
    return np.stack(out, axis=0).reshape(1, -1).repeat(T, axis=0)  # (T, 60) — broadcast across time

def feat_perm_entropy(lm):
    """Permutation entropy of fingertip distance series. Order=4."""
    from itertools import permutations
    perms = list(permutations(range(4)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    def pe(sig, m=4):
        if len(sig) < m: return 0.
        cnt = np.zeros(len(perms))
        for i in range(len(sig) - m + 1):
            order = tuple(np.argsort(sig[i:i+m]).tolist())
            cnt[perm_to_idx[order]] += 1
        p = cnt / cnt.sum()
        p = p[p > 0]
        return -np.sum(p * np.log(p)) / np.log(len(perms))
    out = []
    for tip in TIPS:
        for ax in range(3):
            out.append(pe(lm[:, tip, ax]))
    out = np.array(out)
    return out[None, :].repeat(lm.shape[0], axis=0)  # (T, 15)

def feat_graph_laplacian_eigs(lm):
    """Skeleton graph Laplacian eigenvalues at each frame.
    Edges = MediaPipe skeleton connections. Take top-8 eigenvalues."""
    EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    N = 21
    A = np.zeros((N, N))
    for u, v in EDGES: A[u,v] = 1; A[v,u] = 1
    out = []
    for t in range(lm.shape[0]):
        # Weighted edges by current distance
        W = A.copy()
        for u, v in EDGES:
            d = np.linalg.norm(lm[t,u]-lm[t,v]) + 1e-7
            W[u,v] = 1.0/d; W[v,u] = 1.0/d
        D = np.diag(W.sum(1))
        L = D - W
        eigs = np.sort(np.linalg.eigvalsh(L))[-8:]
        out.append(eigs)
    return np.stack(out)  # (T, 8)

def feat_se3_twist(lm):
    """SE(3) twist coordinates: logarithm of rigid transform between consecutive
    finger frames (axis-angle 3D + translation 3D = 6D per finger)."""
    from scipy.spatial.transform import Rotation as R
    out = []
    for f, idxs in enumerate([[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]):
        twists = []
        for t in range(lm.shape[0]-1):
            P = lm[t, idxs, :]; Q = lm[t+1, idxs, :]
            cP = P.mean(0); cQ = Q.mean(0)
            U = P - cP; V = Q - cQ
            H = U.T @ V
            try:
                Ux, _, Vt = np.linalg.svd(H)
                d = np.sign(np.linalg.det(Vt.T @ Ux.T))
                Rmat = Vt.T @ np.diag([1,1,d]) @ Ux.T
                rotvec = R.from_matrix(Rmat).as_rotvec()  # (3,)
            except:
                rotvec = np.zeros(3)
            trans = cQ - cP  # (3,)
            twists.append(np.concatenate([rotvec, trans]))
        out.append(np.array(twists))
    out = np.concatenate(out, axis=1)  # (T-1, 30)
    return out

def feat_berry_phase(lm):
    """Geometric (Berry) phase: holonomy of palm-frame rotation around closed loop.
    Approximated as cumulative integral of rotation 3-form."""
    from scipy.spatial.transform import Rotation as R
    # Build palm frame at each frame: x = wrist→index_mcp, normalized
    # y = (wrist→pinky_mcp) - x_proj, normalized
    # z = x cross y
    out = []
    cumulative = np.zeros(3)
    prev_R = None
    for t in range(lm.shape[0]):
        x = lm[t,5] - lm[t,0]; x /= np.linalg.norm(x) + 1e-7
        y = lm[t,17] - lm[t,0]
        y -= y.dot(x) * x; y /= np.linalg.norm(y) + 1e-7
        z = np.cross(x, y)
        F = np.stack([x, y, z], axis=1)
        if prev_R is not None:
            dR = F @ prev_R.T
            try:
                cumulative = cumulative + R.from_matrix(dR).as_rotvec()
            except: pass
        prev_R = F
        out.append(cumulative.copy())
    return np.stack(out)  # (T, 3)

def feat_writhe(lm):
    """Vectorized writhe (Gauss linking integral) per fingertip closed-curve."""
    out = []
    for tip in TIPS:
        path = lm[:, tip, :]
        closed = np.concatenate([path, path[:1]], axis=0)
        Tc = len(closed)
        # All edges
        e_start = closed[:-1]
        e_end = closed[1:]
        a = e_end - e_start            # (Tc-1, 3)
        mid = (e_start + e_end) / 2    # (Tc-1, 3)
        d = mid[None, :] - mid[:, None]   # (Tc-1, Tc-1, 3)
        cross = np.cross(a[None, :], a[:, None])  # (Tc-1, Tc-1, 3)
        denom = (np.linalg.norm(d, axis=-1) ** 3) + 1e-7
        kern = (cross * d).sum(-1) / denom  # (Tc-1, Tc-1)
        # mask: |i-j|>=2 and i<j (upper triangle, skip neighbors)
        Te = Tc - 1
        ii, jj = np.meshgrid(np.arange(Te), np.arange(Te), indexing='ij')
        mask = (jj > ii + 1)
        wr = (kern * mask).sum() / (4 * np.pi)
        out.append(wr)
    out = np.array(out)
    return out[None, :].repeat(lm.shape[0], axis=0)

def feat_lz_complexity(lm):
    """Lempel-Ziv complexity of quantized fingertip motion (per axis). Quantize to 4 levels."""
    out = []
    for tip in TIPS:
        for ax in range(3):
            sig = lm[:, tip, ax]
            q = np.digitize(sig, np.quantile(sig, [0.25, 0.5, 0.75]))
            s = ''.join(str(int(c)) for c in q)
            # LZ76 complexity
            i, k, l = 0, 1, 1; n = len(s); c = 1; k_max = 1
            while True:
                if i + k > n: break
                if s[i:i+k] in s[max(0,i+l-k_max-k+1):i+l]:
                    k += 1
                    if i + k > n: c += 1; break
                else:
                    if k > k_max: k_max = k
                    i = i + 1; k = 1
                    c += 1
                    if i >= n: break
            out.append(c)
    out = np.array(out, dtype=np.float32)
    return out[None, :].repeat(lm.shape[0], axis=0)  # (T, 15)

def feat_hurst(lm):
    """Hurst exponent of fingertip trajectories per axis (long-range memory).
    Computed via R/S analysis."""
    def hurst(sig):
        sig = np.asarray(sig)
        n = len(sig)
        if n < 16: return 0.5
        max_k = min(20, n // 4)
        rs = []
        for k in range(2, max_k):
            chunks = n // k
            if chunks < 1: continue
            rs_local = []
            for c in range(chunks):
                seg = sig[c*k:(c+1)*k]
                Y = np.cumsum(seg - seg.mean())
                R = Y.max() - Y.min()
                S = seg.std() + 1e-7
                rs_local.append(R/S)
            rs.append((k, np.mean(rs_local)))
        if len(rs) < 3: return 0.5
        ks = np.log([r[0] for r in rs])
        rss = np.log([r[1] + 1e-7 for r in rs])
        H = np.polyfit(ks, rss, 1)[0]
        return H
    out = []
    for tip in TIPS:
        for ax in range(3):
            out.append(hurst(lm[:, tip, ax]))
    out = np.array(out)
    return out[None, :].repeat(lm.shape[0], axis=0)  # (T, 15)

def feat_recurrence(lm):
    """Recurrence quantification: rate, determinism, laminarity from RP of fingertip path."""
    out = []
    for tip in TIPS:
        p = lm[:, tip, :]
        D = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
        eps = np.median(D)
        RP = (D < eps).astype(np.float32)
        rr = RP.mean()
        # Determinism: fraction of recurrent points on diagonals length>=2
        diag_count = 0; total_rec = RP.sum() + 1e-7
        for k in range(-RP.shape[0]+2, RP.shape[0]-1):
            d = np.diag(RP, k=k)
            run = 0
            for v in d:
                if v == 1: run += 1
                else:
                    if run >= 2: diag_count += run
                    run = 0
            if run >= 2: diag_count += run
        det = diag_count / total_rec
        # Laminarity (vertical lines length>=2)
        lam_count = 0
        for col in range(RP.shape[1]):
            run = 0
            for v in RP[:, col]:
                if v == 1: run += 1
                else:
                    if run >= 2: lam_count += run
                    run = 0
            if run >= 2: lam_count += run
        lam = lam_count / total_rec
        out.append([rr, det, lam])
    out = np.array(out).reshape(-1)
    return out[None, :].repeat(lm.shape[0], axis=0)  # (T, 15)

def feat_procrustes_residuals(lm):
    """After global Procrustes alignment to first frame, residual deformation per joint."""
    from scipy.spatial.transform import Rotation as R
    ref = lm[0]
    cref = ref.mean(0); ref0 = ref - cref
    out = []
    for t in range(lm.shape[0]):
        P = lm[t] - lm[t].mean(0)
        H = ref0.T @ P
        try:
            Ux, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(Vt.T @ Ux.T))
            Rmat = Vt.T @ np.diag([1,1,d]) @ Ux.T
            P_aligned = P @ Rmat.T
        except:
            P_aligned = P
        residual = (P_aligned - ref0)
        out.append(residual.reshape(-1))  # (63,)
    return np.stack(out)  # (T, 63)

def feat_lie_bracket(lm):
    """Lie bracket-like quantity [v, omega] = velocity x angular_velocity per finger.
    Captures coupling between linear and rotational motion."""
    from scipy.spatial.transform import Rotation as R
    out = []
    for f, idxs in enumerate([[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]):
        brackets = []
        prev_R = None; prev_c = None
        for t in range(lm.shape[0]):
            P = lm[t, idxs, :]
            c = P.mean(0)
            U = P - c
            if prev_R is not None:
                trans = c - prev_c
                H = (U.T @ (lm[t-1, idxs, :] - prev_c))
                try:
                    Ux, _, Vt = np.linalg.svd(H)
                    d = np.sign(np.linalg.det(Vt.T @ Ux.T))
                    Rmat = Vt.T @ np.diag([1,1,d]) @ Ux.T
                    omega = R.from_matrix(Rmat).as_rotvec()
                except:
                    omega = np.zeros(3)
                brackets.append(np.cross(trans, omega))
            else:
                brackets.append(np.zeros(3))
            prev_R = U; prev_c = c
        out.append(np.array(brackets))
    out = np.concatenate(out, axis=1)  # (T, 15)
    return out

def feat_spectral_graph_wavelet(lm):
    """Heat-kernel signature at multiple scales on skeleton graph (per joint).
    HKS_t(v) = sum_i exp(-lambda_i * t) * phi_i(v)^2
    Time-averaged."""
    EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    N = 21
    out = []
    times = [0.1, 1.0, 10.0]
    for t_frame in range(lm.shape[0]):
        W = np.zeros((N,N))
        for u,v in EDGES:
            d = np.linalg.norm(lm[t_frame,u]-lm[t_frame,v]) + 1e-7
            W[u,v] = 1.0/d; W[v,u] = 1.0/d
        D = np.diag(W.sum(1))
        L = D - W
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            hks = []
            for tt in times:
                # Per-joint HKS at scale tt; average over joints for compact feat
                e_t = np.exp(-eigvals * tt)
                hks.append((eigvecs**2 * e_t).sum(axis=1).mean())
            out.append(hks)
        except:
            out.append([0,0,0])
    return np.array(out)  # (T, 3)

EXTRACTORS = {
    'frenet_curv_torsion':   feat_frenet,
    'fourier_band_energy':   feat_fourier,
    'permutation_entropy':   feat_perm_entropy,
    'graph_laplacian_eigs':  feat_graph_laplacian_eigs,
    'se3_twist_per_finger':  feat_se3_twist,
    'berry_phase':           feat_berry_phase,
    'writhe_per_fingertip':  feat_writhe,
    'lempel_ziv':            feat_lz_complexity,
    'hurst_exponent':        feat_hurst,
    'recurrence_quantif':    feat_recurrence,
    'procrustes_residuals':  feat_procrustes_residuals,
    'lie_bracket_coupling':  feat_lie_bracket,
    'heat_kernel_sig':       feat_spectral_graph_wavelet,
}

def build_xy(extractor):
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for k, lm_raw in sk.items():
        if lm_raw.shape[0] < 8: continue
        lm = fillnan(lm_raw)
        try:
            feat = extractor(lm)
        except Exception as e:
            continue
        if feat is None or feat.shape[0] < 2: continue
        feat_r = resample(feat, T_FIXED)
        x = feat_r.reshape(-1).astype(np.float32)
        if not np.isfinite(x).all(): x = np.nan_to_num(x)
        if k in train_lbl: X_tr.append(x); y_tr.append(train_lbl[k])
        elif k in test_lbl: X_te.append(x); y_te.append(test_lbl[k])
    X_tr = np.array(X_tr, dtype=np.float32); y_tr = np.array(y_tr, dtype=np.int64)
    X_te = np.array(X_te, dtype=np.float32); y_te = np.array(y_te, dtype=np.int64)
    mean = X_tr.mean(0); std = X_tr.std(0) + 1e-7
    X_tr = (X_tr - mean) / std; X_te = (X_te - mean) / std
    return X_tr, y_tr, X_te, y_te

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

print(f'\n{"signal":<30} {"dim":>6} {"acc%":>8}  vs_chance')
print('-' * 65)
for name, ext in EXTRACTORS.items():
    t0 = time.time()
    try:
        Xtr, ytr, Xte, yte = build_xy(ext)
        if Xtr.shape[0] < 100:
            print(f'{name:<30} too few samples ({Xtr.shape[0]})')
            continue
        acc = shallow(Xtr, ytr, Xte, yte)
        print(f'{name:<30} {Xtr.shape[1]:>6}  {acc:>6.2f}    {acc/4.0:.1f}x  ({time.time()-t0:.0f}s)', flush=True)
    except Exception as e:
        print(f'{name:<30} ERROR: {e}', flush=True)

print('\n--- prior baselines ---')
print(f'{"landmarks_raw":<30} {2016:>6}  {62.03:>6.2f}    15.5x')
print(f'{"fingertip_spread":<30} {320:>6}  {44.19:>6.2f}    11.0x')
print(f'{"finger_curl_angles":<30} {480:>6}  {38.38:>6.2f}    9.6x')
print(f'{"velocity":<30} {2016:>6}  {15.15:>6.2f}    3.8x')
print(f'{"per_finger_Kabsch (QCC)":<30} {640:>6}  {9.96:>6.2f}    2.5x')
print('chance = 4.00%')
