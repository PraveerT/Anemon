"""Depth-video dataloader for CNN-LSTM branch.

Loads raw depth .npy (shape (T_raw, 240, 320, 1), uint8), key-frame-samples to
fixed T, resizes to HxW, normalizes to [0, 1].

Returns: tensor (T, 1, H, W) float32, label int, filename str.
Subject split matches NvidiaLoader.
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
        hflip_prob=0.5,
        time_cutout_prob=0.5,
        time_cutout_max_ratio=0.2,
    ):
        self.phase = phase
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.img_size = img_size
        self.hflip_prob = hflip_prob
        self.time_cutout_prob = time_cutout_prob
        self.time_cutout_max_ratio = time_cutout_max_ratio

        self.r = re.compile('[ \t\n\r:]+')
        self.inputs_list = self._get_inputs_list()
        print(f"DepthVideoLoader[{phase}]: {len(self.inputs_list)} samples")

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, index):
        line = self.inputs_list[index]
        parts = self.r.split(line)
        label = int(parts[-2])
        rel_path = parts[1]  # './Nvidia/Processed/train/class_01/subjectX_rY/sk_depth.avi/NNNN_depth_label_00.npy'
        depth_path = f"../dataset/{rel_path[2:]}"  # strip leading './'

        depth = np.load(depth_path)  # (T_raw, 240, 320, 1) uint8
        if depth.ndim == 4:
            depth = depth[..., 0]  # (T_raw, H, W)

        # Key-frame sample to self.framerate
        idx = dataset_utils.key_frame_sampling(depth.shape[0], self.framerate)
        depth = depth[idx].astype(np.float32)  # (T, 240, 320)

        # Resize each frame to (img_size, img_size)
        T, H, W = depth.shape
        resized = np.empty((T, self.img_size, self.img_size), dtype=np.float32)
        for t in range(T):
            resized[t] = cv2.resize(depth[t], (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        depth = resized

        # Normalize [0, 255] -> [0, 1]
        depth = depth / 255.0

        if self.phase == "train":
            depth = self._augment(depth)

        # Add channel dim: (T, 1, H, W)
        depth = depth[:, None, :, :]
        tensor = torch.from_numpy(depth).float()
        return tensor, label, line

    def _augment(self, depth):
        # Horizontal flip
        if np.random.rand() < self.hflip_prob:
            depth = depth[:, :, ::-1].copy()

        # Temporal cutout (zero out a contiguous block of frames)
        if np.random.rand() < self.time_cutout_prob:
            T = depth.shape[0]
            max_len = max(1, int(T * self.time_cutout_max_ratio))
            cut_len = np.random.randint(1, max_len + 1)
            start = np.random.randint(0, T - cut_len + 1)
            depth[start:start + cut_len] = 0.0

        return depth

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
