"""Dataset loader that returns canonical AE outputs instead of raw points.

Reads the baked tensors written by bake_canonical.py:
  dataset/Nvidia/Processed/canonical_{phase}.npy           (N, T, K, 4)
  dataset/Nvidia/Processed/canonical_{phase}_labels.npy    (N,)

Drop-in replacement for NvidiaLoader: same __getitem__ contract
(tensor, label, line). Use in yaml via:
    dataloader: canonical_nvidia_loader.CanonicalNvidiaLoader
"""
import os

import numpy as np
import torch
from torch.utils import data


class CanonicalNvidiaLoader(data.Dataset):
    _preloaded = {}

    def __init__(self, framerate, valid_subject=None, phase='train',
                 datatype='depth', inputs_type='pts'):
        self.phase = phase
        self.framerate = framerate
        prefix = '../dataset/Nvidia/Processed'
        canon_path = os.path.join(prefix, f'canonical_{phase}.npy')
        label_path = os.path.join(prefix, f'canonical_{phase}_labels.npy')

        key = (phase, canon_path)
        if key not in CanonicalNvidiaLoader._preloaded:
            arr = np.load(canon_path)
            lbl = np.load(label_path)
            print(f'[canonical-loader] {phase}: {arr.shape} '
                  f'{arr.nbytes / 1024 / 1024:.1f} MB')
            CanonicalNvidiaLoader._preloaded[key] = (
                torch.from_numpy(arr), torch.from_numpy(lbl)
            )
        self.tensor, self.labels = CanonicalNvidiaLoader._preloaded[key]

    def __getitem__(self, index):
        return self.tensor[index], int(self.labels[index]), f'canonical_{self.phase}_{index}'

    def __len__(self):
        return self.tensor.shape[0]
