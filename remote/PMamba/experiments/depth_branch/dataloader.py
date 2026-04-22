"""Depth-video dataloader for CNN-LSTM branch.

Loads raw depth .npy (shape (T_raw, 240, 320, 1), uint8), key-frame-samples to
fixed T, resizes to HxW, normalizes to [0, 1].

Options:
  use_tops=True     -> compute per-pixel tops field (3-ch uvd centroid-relative
                       unit direction) and concat with depth -> 4-ch output.
  use_rigidity=True -> also return per-frame rigidity summary features loaded
                       from a precomputed {stem}_rigidity.npy file of shape
                       (T_raw, K). Key-frame sampled to T and returned as a
                       second tensor.

Returns: when use_rigidity=False:
           (depth_tensor, label, line)       where depth_tensor is (T, C, H, W)
         when use_rigidity=True:
           ((depth_tensor, rigidity_tensor), label, line)
           with rigidity_tensor of shape (T, K) float32.
"""
import os
import re
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data

sys.path.append("..")
from dataset import utils as dataset_utils


class DepthVideoLoader(data.Dataset):
    def __init__(
        self,
        framerate=32,
        valid_subject=None,
        phase="train",
        img_size=112,
        use_tops=False,
        use_rigidity=False,
        rigidity_per_point=False,           # True -> load {stem}_rigidity_pp.npy of (T, P=256)
        rigidity_norm_scale=8.0,
        hflip_prob=0.5,
        time_cutout_prob=0.5,
        time_cutout_max_ratio=0.2,
    ):
        self.phase = phase
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.img_size = img_size
        self.use_tops = use_tops
        self.use_rigidity = use_rigidity
        self.rigidity_per_point = rigidity_per_point
        self.rigidity_norm_scale = rigidity_norm_scale
        self.hflip_prob = hflip_prob
        self.time_cutout_prob = time_cutout_prob
        self.time_cutout_max_ratio = time_cutout_max_ratio

        self.r = re.compile('[ \t\n\r:]+')
        self.inputs_list = self._get_inputs_list()

        # Precompute normalized u/v grids for tops calc.
        if self.use_tops:
            u = np.arange(img_size, dtype=np.float32) / img_size  # (W,)
            v = np.arange(img_size, dtype=np.float32) / img_size  # (H,)
            self._uu, self._vv = np.meshgrid(u, v)  # (H, W)

        extra = []
        if use_tops: extra.append("tops 3ch")
        if use_rigidity: extra.append("rigidity K-scalars")
        desc = "depth 1ch" + ("" if not extra else " + " + " + ".join(extra))
        print(f"DepthVideoLoader[{phase}]: {len(self.inputs_list)} samples ({desc})")

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, index):
        line = self.inputs_list[index]
        parts = self.r.split(line)
        label = int(parts[-2])
        rel_path = parts[1]
        depth_path = f"../dataset/{rel_path[2:]}"

        depth = np.load(depth_path)                                # (T_raw, 240, 320, 1) uint8
        if depth.ndim == 4:
            depth = depth[..., 0]

        idx = dataset_utils.key_frame_sampling(depth.shape[0], self.framerate)
        depth = depth[idx].astype(np.float32)                      # (T, 240, 320)

        T = depth.shape[0]
        H = W = self.img_size
        resized = np.empty((T, H, W), dtype=np.float32)
        for t in range(T):
            resized[t] = cv2.resize(depth[t], (W, H), interpolation=cv2.INTER_AREA)
        depth = resized                                            # still [0, 255]

        if self.use_tops:
            tops = self._compute_tops(depth)                       # (T, 3, H, W)

        depth = depth / 255.0                                      # -> [0, 1]

        if self.phase == "train":
            depth, tops_flipped = self._augment(depth, tops if self.use_tops else None)
            if self.use_tops:
                tops = tops_flipped

        depth = depth[:, None, :, :]                               # (T, 1, H, W)
        if self.use_tops:
            arr = np.concatenate([depth, tops], axis=1)            # (T, 4, H, W)
        else:
            arr = depth
        depth_tensor = torch.from_numpy(arr).float()

        if self.use_rigidity:
            suffix = '_rigidity_pp.npy' if self.rigidity_per_point else '_rigidity.npy'
            rig_path = depth_path.replace('.npy', suffix)
            raw_rig = np.load(rig_path).astype(np.float32)
            if raw_rig.shape[0] == depth.shape[0]:
                rig = raw_rig
            else:
                rig = raw_rig[idx]
            # When per-point, sort each frame's residuals (descending) so that
            # column k = the k-th-largest residual. Makes the vector order
            # invariant to point-index mapping across samples.
            if self.rigidity_per_point:
                rig = np.sort(rig, axis=-1)[:, ::-1].copy()
            rig = rig * self.rigidity_norm_scale
            rig_tensor = torch.from_numpy(rig).float()
            return (depth_tensor, rig_tensor), label, line

        return depth_tensor, label, line

    def _compute_tops(self, depth):
        """Per-pixel centroid-relative unit direction in normalized uvd space.

        depth: (T, H, W) float32 in [0, 255]; tops is computed on mask = depth > 0.
        Output: (T, 3, H, W) float32 in [-1, 1]; zero outside mask.
        """
        T, H, W = depth.shape
        uu, vv = self._uu, self._vv                                # (H, W)
        d_norm = depth / 255.0                                     # (T, H, W)
        out = np.zeros((T, 3, H, W), dtype=np.float32)
        for t in range(T):
            m = depth[t] > 0
            if not m.any():
                continue
            ys, xs = np.where(m)
            u_c = xs.mean() / W
            v_c = ys.mean() / H
            d_c = d_norm[t][m].mean()

            ru = uu - u_c
            rv = vv - v_c
            rd = d_norm[t] - d_c
            mag = np.sqrt(ru * ru + rv * rv + rd * rd).clip(min=1e-6)
            out[t, 0] = (ru / mag) * m
            out[t, 1] = (rv / mag) * m
            out[t, 2] = (rd / mag) * m
        return out

    def _augment(self, depth, tops):
        # Horizontal flip
        if np.random.rand() < self.hflip_prob:
            depth = depth[:, :, ::-1].copy()
            if tops is not None:
                tops = tops[:, :, :, ::-1].copy()
                tops[:, 0] = -tops[:, 0]                           # u direction negates

        # Temporal cutout
        if np.random.rand() < self.time_cutout_prob:
            T = depth.shape[0]
            max_len = max(1, int(T * self.time_cutout_max_ratio))
            cut_len = np.random.randint(1, max_len + 1)
            start = np.random.randint(0, T - cut_len + 1)
            depth[start:start + cut_len] = 0.0
            if tops is not None:
                tops[start:start + cut_len] = 0.0

        return depth, tops

    def _get_inputs_list(self):
        prefix = "../dataset/Nvidia/Processed"
        if self.phase == "train":
            inputs_path = prefix + "/train_depth_list.txt"
            lines = open(inputs_path).readlines()
            out = []
            for line in lines:
                if self.valid_subject is not None and f"subject{self.valid_subject}_" in line:
                    continue
                out.append(line)
            return out
        elif self.phase == "valid":
            inputs_path = prefix + "/train_depth_list.txt"
            lines = open(inputs_path).readlines()
            out = []
            for line in lines:
                if self.valid_subject is not None and f"subject{self.valid_subject}_" in line:
                    out.append(line)
            return out
        elif self.phase == "test":
            inputs_path = prefix + "/test_depth_list.txt"
            return open(inputs_path).readlines()
        else:
            raise ValueError(f"unknown phase {self.phase}")
