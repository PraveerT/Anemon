"""NVGesture loader for PST-Transformer. Each clip = our canonical 512-point sequence
(_pts.npy, channels u,v,depth) over 32 frames -> returns [32, 512, 3] standardized per clip
(centre + scalar scale: preserves shape, ~unit scale so the ball-query radius is meaningful,
and removes per-subject hand-size/distance variation -> helps cross-subject generalization).
Train-time aug: in-plane rotation about the depth axis (det +1, NOT a mirror -> chirality-safe),
random scale, small jitter. Splits + labels from the same {train,test}_depth_list.txt the
baseline uses, so test preds align with hard_idx.npy."""
import os
import re
import numpy as np
from torch.utils.data import Dataset

BASE = '/notebooks/Manta'
SUFFIX = os.environ.get('PTS', 'pts')   # 'pts' = baseline 512 | 'dense' = 2048 from the mesh
LISTS = {'train': f'{BASE}/dataset/Nvidia/Processed/train_depth_list.txt',
         'test': f'{BASE}/dataset/Nvidia/Processed/test_depth_list.txt'}
_R = re.compile(r'[ \t\n\r:]+')


class NvGesture(Dataset):
    _cache = {}
    _stats = None   # (per-axis mean, scalar scale) from TRAIN — global normalization keeps absolute coords

    def __init__(self, train=True, num_points=512, aug=True):
        split = 'train' if train else 'test'
        self.train = train
        self.aug = aug and train
        self.num_points = num_points
        if split not in NvGesture._cache:
            clips, labels = [], []
            for line in open(LISTS[split]):
                p = _R.split(line)
                if len(p) < 3:
                    continue
                stem = p[1][1:-4]
                label = int(p[-2])
                arr = np.load(f'{BASE}/dataset/{stem}_{SUFFIX}.npy').astype(np.float32)[:, :, :3]  # _pts(512)|_dense(2048)
                clips.append(arr)
                labels.append(label)
            NvGesture._cache[split] = (clips, np.array(labels, dtype=np.int64))
        self.clips, self.labels = NvGesture._cache[split]
        self.num_classes = 25
        if NvGesture._stats is None and 'train' in NvGesture._cache:
            allp = np.concatenate([c.reshape(-1, 3) for c in NvGesture._cache['train'][0]], 0)
            NvGesture._stats = (allp.mean(0).astype(np.float32), float(allp.std()))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx].copy()                      # (32,512,3)
        gmean, gscale = NvGesture._stats                    # GLOBAL norm: keep absolute position/depth, isotropic
        clip = (clip - gmean) / (gscale + 1e-6)
        if self.aug:
            th = np.random.uniform(-0.26, 0.26)             # ~+-15 deg, in-plane (depth-axis) rotation
            ct, st = np.cos(th), np.sin(th)
            R = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]], np.float32)
            clip = clip @ R.T
            clip = clip * np.random.uniform(0.85, 1.15)
            clip = clip + np.random.normal(0.0, 0.02, clip.shape).astype(np.float32)
            T, N = clip.shape[0], clip.shape[1]              # point dropout: replace ~15%/frame with copies
            nd = int(N * 0.15); rows = np.arange(T)[:, None]
            drop = np.random.randint(0, N, (T, nd)); src = np.random.randint(0, N, (T, nd))
            clip[rows, drop] = clip[rows, src]
        return clip.astype(np.float32), int(self.labels[idx]), idx
