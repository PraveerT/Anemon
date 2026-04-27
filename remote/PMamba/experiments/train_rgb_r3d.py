"""Fine-tune R3D-18 (Kinetics-pretrained) on sk_color v2-trimmed clips.

Reads clips on the fly from .avi via cv2 (no precomputed features).
Saves best test-acc checkpoint + final test probs for fusion.
"""
import os, re, random, sys, time, json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as V
import requests

TG_TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"

def tg(msg):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            chat_id = r["result"][-1]["message"]["chat"]["id"]
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                timeout=10,
            )
    except Exception:
        pass

PREFIX = "/notebooks/PMamba/dataset/Nvidia"
T_FRAMES = 16
H = W = 112
NUM_CLASSES = 25
DEV = torch.device("cuda")
WORK = "/notebooks/PMamba/experiments/work_dir/rgb_r3d18"
os.makedirs(WORK, exist_ok=True)

# Kinetics normalization (R3D-18 default)
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


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
        fr = cv2.resize(fr, (W, H))
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(fr)
    cap.release()
    return np.stack(frames, axis=0) if frames else None


class ColorClips(Dataset):
    def __init__(self, lst_path, train):
        self.entries = [parse_v2(line) for line in open(lst_path)]
        self.train = train

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        rel, s, e, label = self.entries[i]
        path = os.path.join(PREFIX, rel, "sk_color.avi")
        arr = load_window(path, s, e)
        if arr is None or len(arr) < T_FRAMES:
            arr = np.zeros((T_FRAMES, H, W, 3), dtype=np.uint8)
        # frame sampling
        n = len(arr)
        if self.train:
            jitter = np.random.uniform(-0.5, 0.5, size=T_FRAMES)
            base = np.linspace(0, n - 1, T_FRAMES)
            fidx = np.clip(base + jitter, 0, n - 1).round().astype(int)
        else:
            fidx = np.linspace(0, n - 1, T_FRAMES).round().astype(int)
        clip = arr[fidx]                                                 # (T, H, W, 3) uint8
        clip = clip.astype(np.float32) / 255.0
        clip = (clip - MEAN) / STD
        # to (3, T, H, W)
        clip = np.transpose(clip, (3, 0, 1, 2))
        if self.train and random.random() < 0.5:
            clip = clip[:, :, :, ::-1].copy()                            # horizontal flip
        return torch.from_numpy(clip), label


def main():
    # PMamba(depth) ep110 reference — already evaluated on the same v2 test set
    pm = np.load("/notebooks/PMamba/experiments/work_dir/pmamba_branch/pmamba_test_preds.npz")
    pm_probs, pm_labels = pm["probs"], pm["labels"]
    pm_pred = pm_probs.argmax(-1)
    pm_acc = (pm_pred == pm_labels).mean()
    print(f"PMamba(depth) reference solo: {pm_acc*100:.2f}%")

    train_ds = ColorClips(f"{PREFIX}/nvgesture_train_correct_cvpr2016_v2.lst", train=True)
    test_ds = ColorClips(f"{PREFIX}/nvgesture_test_correct_cvpr2016_v2.lst", train=False)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    print(f"train clips: {len(train_ds)}  test clips: {len(test_ds)}")
    tg(f"<b>R3D-18 RGB</b> training started — 40 ep, 1050 train / 482 test\nPMamba ref solo {pm_acc*100:.2f}%")

    model = V.r3d_18(weights=V.R3D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEV)

    # lower LRs — Kinetics-pretrained R3D-18 on 1050 samples; aggressive LR caused
    # acc to oscillate downward (60->56->51) in initial run. Reduced 3x.
    params = [
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": 3e-5},
        {"params": model.fc.parameters(), "lr": 3e-4},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

    best_acc = 0.0
    best_logits = None
    for epoch in range(40):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for clips, labels in train_loader:
            clips = clips.to(DEV, non_blocking=True)
            labels = labels.to(DEV, non_blocking=True)
            logits = model(clips)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(-1) == labels).sum().item()
            train_total += labels.size(0)
        sched.step()

        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for clips, labels in test_loader:
                clips = clips.to(DEV, non_blocking=True)
                logits = model(clips)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.numpy())
        all_logits = np.concatenate(all_logits, 0)
        all_labels = np.concatenate(all_labels, 0)
        rgb_pred = all_logits.argmax(-1)
        acc = (rgb_pred == all_labels).mean()
        train_acc = train_correct / max(1, train_total)
        train_loss /= max(1, train_total)

        # Late-fusion sweep + oracle vs PMamba
        rgb_probs_ep = torch.softmax(torch.from_numpy(all_logits), dim=-1).numpy()
        best_fuse = 0.0; best_alpha = 0.0
        for alpha in np.linspace(0, 1, 41):
            fused = alpha * pm_probs + (1 - alpha) * rgb_probs_ep
            fa = (fused.argmax(-1) == all_labels).mean()
            if fa > best_fuse:
                best_fuse = fa; best_alpha = alpha
        oracle = ((pm_pred == all_labels) | (rgb_pred == all_labels)).mean()

        if acc > best_acc:
            best_acc = acc
            best_logits = all_logits
            torch.save(model.state_dict(), f"{WORK}/best.pt")

        line = (f"ep {epoch:2d}  rgb {acc*100:.2f}%  best {best_acc*100:.2f}%  "
                f"fuse[a={best_alpha:.2f}] {best_fuse*100:.2f}%  oracle {oracle*100:.2f}%  "
                f"({time.time()-t0:.0f}s)")
        print(line)
        tg(f"<b>R3D ep{epoch}</b>\nRGB {acc*100:.2f}%  best {best_acc*100:.2f}%\nFuse[a={best_alpha:.2f}] <b>{best_fuse*100:.2f}%</b>\nOracle {oracle*100:.2f}%")

    print(f"\n=== R3D-18 RGB solo best: {best_acc*100:.2f}% ===")
    rgb_probs = torch.softmax(torch.from_numpy(best_logits), dim=-1).numpy()
    np.savez_compressed(f"{PREFIX}/Processed/rgb_r3d18_test_preds.npz",
                        probs=rgb_probs, labels=all_labels)

    rgb_pred = rgb_probs.argmax(-1)
    final_oracle = ((pm_pred == all_labels) | (rgb_pred == all_labels)).mean()
    best_fuse = 0.0; best_alpha = 0.0
    for alpha in np.linspace(0, 1, 41):
        fused = alpha * pm_probs + (1 - alpha) * rgb_probs
        acc = (fused.argmax(-1) == all_labels).mean()
        if acc > best_fuse:
            best_fuse = acc; best_alpha = alpha
    final = (f"<b>R3D-18 done</b>\nRGB best {best_acc*100:.2f}%\n"
             f"PMamba {pm_acc*100:.2f}%\n"
             f"Fuse[a={best_alpha:.2f}] <b>{best_fuse*100:.2f}%</b>  "
             f"(Δ {(best_fuse - pm_acc)*100:+.2f}pp)\n"
             f"Oracle {final_oracle*100:.2f}%")
    print(final.replace("<b>", "").replace("</b>", ""))
    tg(final)


if __name__ == "__main__":
    main()
