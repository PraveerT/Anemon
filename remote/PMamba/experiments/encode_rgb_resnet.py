"""Encode each NVGesture clip's sk_color frames via frozen ImageNet ResNet18.

Reads:
  /notebooks/PMamba/dataset/Nvidia/Processed/{train,test}_rgbd_list.txt
  Plus the corresponding sk_color.avi files (same parent dir as sk_depth.avi
  in the v2-trimmed Video_data symlink).

Writes:
  /notebooks/PMamba/dataset/Nvidia/Processed/rgb_resnet18_train.npz
  /notebooks/PMamba/dataset/Nvidia/Processed/rgb_resnet18_test.npz
  arrays:
    feats: (N, T, 512) float32
    labels: (N,) int64

Uses 16 frames per clip uniformly sampled across the v2 gesture window.
"""
import os
import re
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PREFIX = "/notebooks/PMamba/dataset/Nvidia"
T_OUT = 16
H, W = 112, 112  # input size to ResNet


def parse_v2(line):
    sp = re.split(r"[ \t\n\r]+", line.strip())
    rel = sp[0].split(":", 1)[1].lstrip("./").rstrip("/")
    f = sp[2].split(":")
    return rel, int(f[2]), int(f[3]), int(sp[-1].split(":")[-1]) - 1


def load_color_window(path, start, end):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (W, H))
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(fr)
    cap.release()
    return np.stack(frames, axis=0) if frames else None


def main():
    splits = {
        "train": f"{PREFIX}/nvgesture_train_correct_cvpr2016_v2.lst",
        "test":  f"{PREFIX}/nvgesture_test_correct_cvpr2016_v2.lst",
    }

    # frozen ResNet18 — strip final fc, keep avg-pool output (512-d)
    import torchvision.models as M
    backbone = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone = backbone.cuda().eval()
    for p in backbone.parameters():
        p.requires_grad = False

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    @torch.no_grad()
    def encode_clip(rgb_uint8):
        # rgb_uint8: (T, H, W, 3) numpy uint8
        x = torch.from_numpy(rgb_uint8).permute(0, 3, 1, 2).float().cuda() / 255.0  # (T, 3, H, W)
        x = (x - mean) / std
        return backbone(x).cpu().numpy()  # (T, 512)

    for split_name, lst_path in splits.items():
        lines = open(lst_path).readlines()
        print(f"{split_name}: {len(lines)} clips")
        feats_all = np.zeros((len(lines), T_OUT, 512), dtype=np.float32)
        labels_all = np.zeros((len(lines),), dtype=np.int64)
        bad = 0
        for idx, line in enumerate(tqdm(lines)):
            rel, s, e, label = parse_v2(line)
            color_path = os.path.join(PREFIX, rel, "sk_color.avi")
            arr = load_color_window(color_path, s, e)
            if arr is None or len(arr) < 4:
                bad += 1
                continue
            fidx = np.linspace(0, len(arr) - 1, T_OUT).round().astype(int)
            arr = arr[fidx]
            feats = encode_clip(arr)
            feats_all[idx] = feats
            labels_all[idx] = label
        out_path = f"{PREFIX}/Processed/rgb_resnet18_{split_name}.npz"
        np.savez_compressed(out_path, feats=feats_all, labels=labels_all)
        print(f"wrote {out_path}, bad={bad}")


if __name__ == "__main__":
    main()
