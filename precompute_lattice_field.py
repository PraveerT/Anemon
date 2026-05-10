"""Precompute lattice arrow quaternion field for all NVGesture samples.

For each (sample, frame), compute a 3D lattice K×K×K of arrow vectors. Each
arrow starts pointing north and is deflected by nearby hand points via a
Gaussian kernel. The deflected direction is encoded as a quaternion (rotation
from north to deflected direction).

Output: (T=32, K^3, 4) per sample, saved as a single .npz keyed by sample-name.

Mathematical setup:
  Lattice: K^3 points in [-1, 1]^3
  Hand: depth point cloud H_t = {h_j} (normalized to same range)
  Kernel: gaussian K(p_i, h_j) = exp(-||p_i - h_j||^2 / sigma^2)
  Deflection at lattice point i:
    d_i = sum_j K(p_i, h_j) * (h_j - p_i)
  Direction:
    A_i = normalize(north + alpha * d_i)
  Quaternion:
    q_i = rotation taking (0,0,1) -> A_i
"""
import os, re, sys, time, glob
import numpy as np
import torch

LATTICE_K = 6           # K^3=216 lattice points — coarser → each gets stronger signal
SIGMA = 0.4             # broader kernel, more lattice points respond
ALPHA = 5.0             # moderate deflection
T_FIXED = 32

PROCESSED_ROOT = '/notebooks/PMamba/dataset/Nvidia/Processed'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
OUT_PATH = f'{PROCESSED_ROOT}/lattice_arrows_K{LATTICE_K}_v2.npz'


def make_lattice(K):
    """Create K^3 lattice points in [-1, 1]^3."""
    coords = np.linspace(-1, 1, K)
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing='ij'), axis=-1)
    return grid.reshape(-1, 3).astype(np.float32)  # (K^3, 3)


def vec_to_quat_torch(V):
    """V: (..., 3). Returns (..., 4) [w,x,y,z] = quaternion rotating (0,0,1) -> V."""
    n = V.norm(dim=-1, keepdim=True) + 1e-9
    u = V / n
    cos_h = torch.clamp((1 + u[..., 2:3]) * 0.5, 1e-9, 1.0)
    w = torch.sqrt(cos_h)
    sin_h = torch.sqrt(torch.clamp(1 - cos_h, 0, 1))
    axis = torch.zeros_like(u)
    axis[..., 0] = -u[..., 1]
    axis[..., 1] = u[..., 0]
    s = axis.norm(dim=-1, keepdim=True) + 1e-9
    axis = axis / s * sin_h
    return torch.cat([w, axis], dim=-1)


def compute_lattice_field(hand_xyz_t, lattice, sigma=SIGMA, alpha=ALPHA, device='cuda'):
    """hand_xyz_t: (T, P, 3) hand points per frame.
    lattice: (K^3, 3) fixed lattice positions.
    Returns: (T, K^3, 4) quaternion field per frame.
    """
    h = torch.from_numpy(hand_xyz_t).to(device).float()
    L = torch.from_numpy(lattice).to(device).float()
    T, P, _ = h.shape
    K3 = L.shape[0]
    # diffs[t, i, j] = h_j - p_i  → shape (T, K3, P, 3)
    diffs = h.unsqueeze(1) - L.unsqueeze(0).unsqueeze(2)  # broadcast: (T, K3, P, 3)
    dist2 = (diffs ** 2).sum(-1)  # (T, K3, P)
    weights = torch.exp(-dist2 / (sigma ** 2))  # (T, K3, P)
    deflection = (weights.unsqueeze(-1) * diffs).sum(2)  # (T, K3, 3)
    north = torch.zeros_like(deflection); north[..., 2] = 1.0
    direction = north + alpha * deflection  # additive perturbation around north
    quat = vec_to_quat_torch(direction)  # (T, K3, 4)
    return quat.cpu().numpy()


def parse_annot_paths(path):
    """Return list of sample paths."""
    paths = []
    with open(path) as f:
        for line in f:
            m = re.search(r'path:(\S+)', line)
            if m: paths.append(m.group(1))
    return paths


def find_pts_npy(sample_path):
    """Locate the _pts.npy file given a sample path like './Video_data/class_01/subject13_r0'."""
    # The processed file is at dataset/Nvidia/Processed/{train|test}/class_XX/subject*/sk_depth.avi/000_depth_label_XX_pts.npy
    m = re.search(r'class_(\d+)/(subject\S+)', sample_path)
    if not m: return None
    cls, subj = m.group(1), m.group(2)
    # Try test then train
    for split in ['test', 'train']:
        candidates = glob.glob(f'{PROCESSED_ROOT}/{split}/class_{cls}/{subj}/sk_depth.avi/*_pts.npy')
        if candidates:
            return candidates[0]
    return None


def resample_T(arr, T_target):
    if arr.shape[0] == T_target: return arr
    idx = np.linspace(0, arr.shape[0] - 1, T_target).astype(np.int64)
    return arr[idx]


def main():
    train_paths = parse_annot_paths(f'{ANNOT_ROOT}/nvgesture_train_correct_cvpr2016_v2.lst')
    test_paths = parse_annot_paths(f'{ANNOT_ROOT}/nvgesture_test_correct_cvpr2016_v2.lst')
    all_paths = train_paths + test_paths
    print(f'Total samples: {len(all_paths)}')

    lattice = make_lattice(LATTICE_K)
    print(f'Lattice: {LATTICE_K}^3 = {len(lattice)} points')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out = {}
    t0 = time.time()
    for i, sp in enumerate(all_paths):
        npy = find_pts_npy(sp)
        if npy is None:
            print(f'  [{i}] NOT FOUND: {sp}')
            continue
        try:
            data = np.load(npy)  # (T, P, C) or (T, P, 8)
        except Exception as e:
            print(f'  [{i}] load error: {e}')
            continue
        # Take xyz from first 3 channels
        if data.ndim != 3 or data.shape[-1] < 3:
            print(f'  [{i}] unexpected shape: {data.shape}')
            continue
        xyz = data[..., :3].astype(np.float32)
        # Normalize per-sample to [-1, 1]
        mins = xyz.reshape(-1, 3).min(0)
        maxs = xyz.reshape(-1, 3).max(0)
        scale = (maxs - mins).max() / 2 + 1e-9
        center = (mins + maxs) / 2
        xyz_n = (xyz - center) / scale  # in [-1, 1]
        # Resample to T_FIXED frames
        xyz_n = resample_T(xyz_n, T_FIXED)  # (T_FIXED, P, 3)
        with torch.no_grad():
            quat_field = compute_lattice_field(xyz_n, lattice, device=device)
        out[sp] = quat_field
        if (i + 1) % 50 == 0 or i == len(all_paths) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(all_paths) - i - 1)
            print(f'  {i+1}/{len(all_paths)} elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

    np.savez_compressed(OUT_PATH, **out)
    print(f'\nSaved {len(out)} samples to {OUT_PATH}')
    print(f'Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
