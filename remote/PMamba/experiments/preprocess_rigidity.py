"""Precompute per-frame rigidity-residual summary features for every sample.

For each sample, runs the Hungarian correspondence chain (reuses the parity
loader) to get aligned points (T, P, 3), computes per-frame Kabsch residual
magnitudes via batched SVD, and summarizes to K = 6 scalars per frame:
  mean, std, max, p75, p90, p95.

Output: {stem}_rigidity.npy of shape (T, K=6) per sample, written next to the
depth .npy file so DepthVideoLoader can pick it up.

Run once. Processes ~1532 train+test samples in a few minutes.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

from nvidia_dataloader import NvidiaQuaternionQCCParityLoader


P_SAMPLE = 256
K_FEATS = 6


def kabsch_residuals_batch(points: np.ndarray) -> np.ndarray:
    """points (T, P, 3) -> (T, P) residual magnitudes under cyclic (t, t+1) fits."""
    T, P, _ = points.shape
    pts = torch.from_numpy(points).float()
    nxt = torch.roll(pts, shifts=-1, dims=0)
    cP = pts.mean(dim=-2, keepdim=True)
    cQ = nxt.mean(dim=-2, keepdim=True)
    Pc = pts - cP
    Qc = nxt - cQ
    H = Pc.transpose(-2, -1) @ Qc
    H = H + 1e-6 * torch.eye(3)
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)
    d = torch.linalg.det(V @ U.transpose(-2, -1))
    D = torch.eye(3).expand(T, 3, 3).clone()
    D[..., 2, 2] = d
    R = V @ D @ U.transpose(-2, -1)
    pred = (R @ Pc.transpose(-2, -1)).transpose(-2, -1)
    resid = Qc - pred
    return resid.norm(dim=-1).numpy()              # (T, P)


def frame_summary(residuals: np.ndarray) -> np.ndarray:
    """(T, P) -> (T, K=6): mean, std, max, p75, p90, p95 per frame."""
    T, _ = residuals.shape
    out = np.zeros((T, K_FEATS), dtype=np.float32)
    for t in range(T):
        r = residuals[t]
        out[t, 0] = r.mean()
        out[t, 1] = r.std()
        out[t, 2] = r.max()
        out[t, 3] = np.quantile(r, 0.75)
        out[t, 4] = np.quantile(r, 0.90)
        out[t, 5] = np.quantile(r, 0.95)
    return out


def corr_sample(points, orig_flat_idx, corr_target, corr_weight, sample_size):
    T, P, C = points.shape
    total_pts = corr_target.shape[-1]
    raw_ppf = total_pts // T
    out = np.zeros((T, sample_size, C), dtype=np.float32)
    idx = np.linspace(0, P - 1, sample_size).round().astype(int)
    out[0] = points[0, idx]
    current_prov = orig_flat_idx[0, idx].astype(int)
    for t in range(T - 1):
        next_prov = orig_flat_idx[t + 1].astype(int)
        reverse_map = -np.ones(total_pts, dtype=int)
        reverse_map[next_prov] = np.arange(P)
        tgt_flat = corr_target[current_prov]
        tgt_w = corr_weight[current_prov]
        tgt_safe = np.clip(tgt_flat, 0, None)
        tgt_frame = tgt_flat // raw_ppf
        tgt_pos = reverse_map[tgt_safe]
        valid = (tgt_flat >= 0) & (tgt_w > 0) & (tgt_frame == t + 1) & (tgt_pos >= 0)
        next_idx = np.random.randint(0, P, size=sample_size)
        next_idx[valid] = tgt_pos[valid]
        out[t + 1] = points[t + 1, next_idx]
        current_prov = orig_flat_idx[t + 1, next_idx].astype(int)
    return out


def arr(x):
    return x.numpy() if hasattr(x, 'numpy') else x


def process_phase(phase: str):
    L = NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
        assignment_mode='hungarian',
    )
    n = len(L)
    print(f"[{phase}] {n} samples")
    skipped = 0
    done = 0
    for i in range(n):
        s, _lab, line = L[i]
        rel_path = L.r.split(line)[1][2:]             # './...' -> '...'
        depth_path = f"../dataset/{rel_path}"
        out_path = depth_path.replace('.npy', '_rigidity.npy')
        if os.path.exists(out_path):
            skipped += 1
            continue
        try:
            pts = arr(s['points']).astype(np.float32)
            ofi = arr(s['orig_flat_idx'])
            cti = arr(s['corr_full_target_idx'])
            cw = arr(s['corr_full_weight'])
            if cti.ndim == 2: cti = cti[0]
            if cw.ndim == 2: cw = cw[0]
            if ofi.ndim == 3: ofi = ofi[0]
            aligned = corr_sample(pts, ofi, cti, cw, P_SAMPLE)[:, :, :3]
            resid = kabsch_residuals_batch(aligned)        # (T, P)
            summary = frame_summary(resid)                 # (T, K)
            np.save(out_path, summary)
            done += 1
        except Exception as e:
            print(f"  {i} fail: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}  new={done}  cached={skipped}")
    print(f"[{phase}] done {done} new, {skipped} cached")


if __name__ == '__main__':
    process_phase('train')
    process_phase('test')
    print("all done")
