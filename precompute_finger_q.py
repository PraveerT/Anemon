"""Precompute per-finger Kabsch quaternions from MediaPipe skeleton landmarks.
For each sample, for each frame pair (t, t+1), compute 5 quaternions (one per finger)
using Kabsch on the finger's landmark group.

Output: dict {sample_path: (T_minus_1, 5, 4) float32}
- 5 fingers: thumb (1-4), index (5-8), middle (9-12), ring (13-16), pinky (17-20)
- Wrist (0) is shared base for all
"""
import numpy as np

SKELETON_PATH = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
OUT_PATH = '/notebooks/PMamba/dataset/Nvidia/Processed/finger_quat_targets.npz'

# MediaPipe hand landmark indices per finger (4 joints each + wrist anchor 0)
FINGER_GROUPS = [
    [0, 1, 2, 3, 4],   # thumb
    [0, 5, 6, 7, 8],   # index
    [0, 9, 10, 11, 12],# middle
    [0, 13, 14, 15, 16],# ring
    [0, 17, 18, 19, 20],# pinky
]

def kabsch_quat(p_src, p_tgt):
    """Per-pair Kabsch. p_src, p_tgt: (K, 3). Return q (4,) [w, x, y, z]."""
    c_s = p_src.mean(0, keepdims=True)
    c_t = p_tgt.mean(0, keepdims=True)
    u = p_src - c_s
    v = p_tgt - c_t
    H = u.T @ v  # (3, 3)
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    # Convert R to quaternion
    tr = R.trace()
    if tr > 0:
        S = 2 * np.sqrt(tr + 1)
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = 2 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = 2 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float32)
    n = np.linalg.norm(q)
    return q / n if n > 1e-7 else np.array([1, 0, 0, 0], dtype=np.float32)


print('Loading skeleton...')
sk = dict(np.load(SKELETON_PATH, allow_pickle=False))
print(f'{len(sk)} samples')

out = {}
for i, (key, lm) in enumerate(sk.items()):
    # lm: (T, 21, 3), may have NaN
    T = lm.shape[0]
    if T < 2:
        out[key] = np.zeros((0, 5, 4), dtype=np.float32)
        continue
    # Forward-fill NaN
    valid = np.isfinite(lm[..., 0]).all(axis=-1)
    last_valid = None
    lm_filled = lm.copy()
    for t in range(T):
        if valid[t]: last_valid = lm_filled[t]
        elif last_valid is not None: lm_filled[t] = last_valid
    for t in range(T - 1, -1, -1):
        if not np.isfinite(lm_filled[t]).all():
            for t2 in range(t + 1, T):
                if valid[t2]:
                    lm_filled[t] = lm_filled[t2]
                    break
            else: lm_filled[t] = 0
    # Per-frame-pair, per-finger Kabsch
    qs = np.zeros((T - 1, 5, 4), dtype=np.float32)
    for t in range(T - 1):
        for f, idxs in enumerate(FINGER_GROUPS):
            qs[t, f] = kabsch_quat(lm_filled[t, idxs], lm_filled[t+1, idxs])
    out[key] = qs
    if i % 100 == 0:
        print(f'{i+1}/{len(sk)} | {key[-40:]} | qs shape {qs.shape}', flush=True)

np.savez_compressed(OUT_PATH, **out)
print(f'\nSaved {len(out)} samples to {OUT_PATH}')
