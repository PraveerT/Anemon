"""Encode sk_depth.avi via frozen ImageNet ResNet18 (depth replicated to 3 channels).

Output: depth_resnet18_{train,test}.npz with feats (N, T=16, 512), labels (N,).
"""
import os, re, cv2, numpy as np, torch
import torch.nn as nn
import torchvision.models as M
from tqdm import tqdm

PREFIX = "/notebooks/PMamba/dataset/Nvidia"
T_OUT, H, W = 16, 112, 112


def parse_v2(line):
    sp = re.split(r"[ \t\n\r]+", line.strip())
    rel = sp[0].split(":", 1)[1].lstrip("./").rstrip("/")
    f = sp[2].split(":")
    return rel, int(f[2]), int(f[3]), int(sp[-1].split(":")[-1]) - 1


def load_window(path, start, end):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ok, fr = cap.read()
        if not ok:
            break
        # sk_depth has R=G=B (grayscale); take ch0 then resize
        fr = fr[..., 0]
        fr = cv2.resize(fr, (W, H))
        frames.append(fr)
    cap.release()
    return np.stack(frames, axis=0) if frames else None


def main():
    backbone = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone = backbone.cuda().eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # ImageNet normalization (depth replicated 3x; this is approximate but works for
    # transfer because ResNet's early conv only cares about local structure)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    @torch.no_grad()
    def encode(arr_gray):
        # (T, H, W) -> (T, 3, H, W)
        x = torch.from_numpy(arr_gray).float().cuda() / 255.0
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)
        x = (x - mean) / std
        return backbone(x).cpu().numpy()

    for split in ("train", "test"):
        lst = open(f"{PREFIX}/nvgesture_{split}_correct_cvpr2016_v2.lst").readlines()
        feats = np.zeros((len(lst), T_OUT, 512), dtype=np.float32)
        labels = np.zeros((len(lst),), dtype=np.int64)
        bad = 0
        for idx, line in enumerate(tqdm(lst, desc=split)):
            rel, s, e, label = parse_v2(line)
            arr = load_window(os.path.join(PREFIX, rel, "sk_depth.avi"), s, e)
            if arr is None or len(arr) < 4:
                bad += 1
                continue
            fidx = np.linspace(0, len(arr) - 1, T_OUT).round().astype(int)
            feats[idx] = encode(arr[fidx])
            labels[idx] = label
        np.savez_compressed(f"{PREFIX}/Processed/depth_resnet18_{split}.npz",
                            feats=feats, labels=labels)
        print(f"{split}: wrote, bad={bad}")


if __name__ == "__main__":
    main()
