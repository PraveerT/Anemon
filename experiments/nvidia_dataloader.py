import re
import sys
import os
import numpy as np
import torch

sys.path.append("..")

from utils import *
import torch.utils.data as data


class NvidiaLoader(data.Dataset):
    # Preload all samples once per (phase, datatype, valid_subject), share
    # across DataLoader instances. Keyed dict is class-level so re-init in
    # the same process hits cache.
    _preloaded = {}

    def __init__(self, framerate, valid_subject=None, phase="train",
                 datatype="depth", inputs_type="pts"):
        self.phase = phase
        self.datatype = datatype
        self.inputs_type = inputs_type
        self.framerate = framerate
        self.valid_subject = valid_subject
        self.inputs_list = self.get_inputs_list()
        self.r = re.compile(r'[ \t\n\r:]+')

        try:
            self.dataset_stats = np.load(
                'nvidia_dataset_stats.npy', allow_pickle=True
            ).item()
            print("Loaded global dataset statistics for consistent normalization")
        except FileNotFoundError:
            print("Warning: Global dataset statistics not found, using default")
            self.dataset_stats = None

        print(len(self.inputs_list))

        key = (phase, datatype, valid_subject)
        if key not in NvidiaLoader._preloaded:
            self._preload(key)
        self.tensor, self.labels_tensor = NvidiaLoader._preloaded[key]

    def _preload(self, key):
        stats = self.dataset_stats
        samples, labels = [], []
        for line in self.inputs_list:
            label = int(self.r.split(line)[-2])
            path = f"../dataset/{self.r.split(line)[1][1:-4]}_pts.npy"
            arr = np.load(path).astype(np.float32)[:, :, :4]   # (T, P, 4)
            T, P, C = arr.shape
            flat = arr.reshape(-1, C)
            flat[:, 0] = (flat[:, 0] - stats['x_mean']) / stats['x_std']
            flat[:, 1] = (flat[:, 1] - stats['y_mean']) / stats['y_std']
            flat[:, 2] = (flat[:, 2] - stats['z_mean']) / stats['z_std']
            flat[:, 3] = (flat[:, 3] - stats['t_mean']) / stats['t_std']
            samples.append(flat.reshape(T, P, C))
            labels.append(label)
        arr_all = np.stack(samples).astype(np.float32)
        labels_arr = np.array(labels, dtype=np.int64)
        size_mb = arr_all.nbytes / 1e6
        print(f"Preloaded {key[0]}: {arr_all.shape}, {size_mb:.1f} MB")
        tensor = torch.from_numpy(arr_all).pin_memory()
        labels_t = torch.from_numpy(labels_arr).pin_memory()
        NvidiaLoader._preloaded[key] = (tensor, labels_t)

    def __getitem__(self, index):
        # Returns (sample tensor, label int, line). label as Python int to
        # match prior interface (test_loader_args path string is also used).
        return self.tensor[index], int(self.labels_tensor[index]), self.inputs_list[index]

    def __len__(self):
        return len(self.inputs_list)

    def get_inputs_list(self):
        prefix = "../dataset/Nvidia/Processed"
        if self.phase == "train":
            if self.datatype == "depth":
                inputs_path = prefix + "/train_depth_list.txt"
            elif self.datatype == "rgb":
                inputs_path = prefix + "/train_color_list.txt"
            lines = open(inputs_path).readlines()
            return [
                line for line in lines
                if "subject" + str(self.valid_subject) + "_" not in line
            ]
        if self.phase == "valid":
            if self.datatype == "depth":
                inputs_path = prefix + "/train_depth_list.txt"
            elif self.datatype == "rgb":
                inputs_path = prefix + "/train_color_list.txt"
            lines = open(inputs_path).readlines()
            return [
                line for line in lines
                if "subject" + str(self.valid_subject) + "_" in line
            ]
        if self.phase == "test":
            if self.datatype == "depth":
                inputs_path = prefix + "/test_depth_list.txt"
            elif self.datatype == "rgb":
                inputs_path = prefix + "/test_color_list.txt"
            return open(inputs_path).readlines()
        raise AssertionError("Phase error.")

    @staticmethod
    def key_frame_sampling(key_cnt, frame_size):
        factor = frame_size * 1.0 / key_cnt
        return [int(j / factor) for j in range(frame_size)]
