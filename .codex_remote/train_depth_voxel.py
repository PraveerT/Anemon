"""Train a compact 3D CNN on depth-derived voxel envelopes.

Input is dataset/Nvidia/Processed/_envelopes/{train,test}_voxels_32.npy,
not pointcloud samples. The script writes Anemon-compatible logs and
test_logits.npz under experiments/work_dir/depth_small.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


def read_split(path):
    labels, sigs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rel, _n, lab = line.strip().split()[:3]
            labels.append(int(lab))
            sigs.append(rel.strip("/"))
    return np.array(labels, dtype=np.int64), np.array(sigs, dtype=object)


class VoxelDataset(Dataset):
    def __init__(self, voxels, labels, sigs, train=False):
        self.voxels = voxels
        self.labels = labels
        self.sigs = sigs
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = np.asarray(self.voxels[idx], dtype=np.uint8)
        # (D,H,W,C) -> (C,D,H,W)
        x = torch.from_numpy(x.copy()).permute(3, 0, 1, 2).float() / 255.0
        if self.train:
            if random.random() < 0.25:
                x = torch.flip(x, dims=[2])
            if random.random() < 0.25:
                x = torch.flip(x, dims=[3])
            if random.random() < 0.20:
                z = random.randint(0, x.shape[1] - 4)
                x[:, z:z + 3] = 0
        x = (x - 0.5) / 0.5
        return x, int(self.labels[idx]), str(self.sigs[idx])


class ResBlock3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.drop = nn.Dropout3d(drop)
        if stride != 1 or in_c != out_c:
            self.short = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_c),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x):
        y = F.gelu(self.bn1(self.conv1(x)))
        y = self.drop(self.bn2(self.conv2(y)))
        return F.gelu(y + self.short(x))


class VoxelNet(nn.Module):
    def __init__(self, num_classes=25, width=48, dropout=0.35):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(4, width, 3, padding=1, bias=False),
            nn.BatchNorm3d(width),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            ResBlock3D(width, width, drop=0.05),
            ResBlock3D(width, width * 2, stride=2, drop=0.08),
            ResBlock3D(width * 2, width * 2, drop=0.08),
            ResBlock3D(width * 2, width * 4, stride=2, drop=0.10),
            ResBlock3D(width * 4, width * 4, drop=0.10),
            ResBlock3D(width * 4, width * 6, stride=2, drop=0.12),
        )
        dim = width * 6
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


def set_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr


def lr_for_epoch(ep, epochs, lr, min_lr, warmup):
    if ep <= warmup:
        return lr * ep / max(1, warmup)
    p = (ep - warmup) / max(1, epochs - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + np.cos(np.pi * p))


def topk(logits, labels):
    pred = logits.topk(5, dim=1).indices
    p1 = (pred[:, :1] == labels[:, None]).any(1).float().mean().item() * 100
    p5 = (pred[:, :5] == labels[:, None]).any(1).float().mean().item() * 100
    return p1, p5


@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    logits_all, labels_all, sigs_all = [], [], []
    for x, y, sig in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(enabled=amp):
            logits = model(x)
        logits_all.append(logits.float().cpu())
        labels_all.append(y.cpu())
        sigs_all.extend(sig)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    p1, p5 = topk(logits, labels)
    return p1, p5, logits.numpy(), labels.numpy(), np.array(sigs_all, dtype=object)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/notebooks/Anemon")
    p.add_argument("--workdir", default="/notebooks/Anemon/experiments/work_dir/depth_small")
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min-lr", type=float, default=8e-6)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--wd", type=float, default=0.03)
    p.add_argument("--width", type=int, default=56)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--stop-at", type=float, default=89.0)
    p.add_argument("--stop-patience", type=int, default=8)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    env_dir = os.path.join(args.root, "dataset/Nvidia/Processed/_envelopes")
    train_x = np.load(os.path.join(env_dir, "train_voxels_32.npy"), mmap_mode="r")
    test_x = np.load(os.path.join(env_dir, "test_voxels_32.npy"), mmap_mode="r")
    train_y, train_s = read_split("/notebooks/cvpr_data/dataset_splits/train.txt")
    test_y, test_s = read_split("/notebooks/cvpr_data/dataset_splits/valid.txt")
    train_ds = VoxelDataset(train_x, train_y, train_s, train=True)
    test_ds = VoxelDataset(test_x, test_y, test_s, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             persistent_workers=args.workers > 0)

    device = torch.device("cuda")
    model = VoxelNet(width=args.width).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(enabled=not args.no_amp)
    amp = not args.no_amp
    log = open(os.path.join(args.workdir, "log.txt"), "a", encoding="utf-8")

    def write(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    write(f"VoxelNet params: {params_m:.3f}M width={args.width}")
    write(f"train clips: {len(train_ds)}  valid clips: {len(test_ds)}")
    write(f"batch={args.batch_size} lr={args.lr} wd={args.wd}")

    best, best_ep, best_payload = 0.0, 0, None
    for ep in range(1, args.epochs + 1):
        lr = lr_for_epoch(ep, args.epochs, args.lr, args.min_lr, args.warmup)
        set_lr(opt, lr)
        write(f"Training epoch: {ep}")
        model.train()
        t0 = time.time()
        total_loss = total_ok = total_n = 0
        for bi, (x, y, _sig) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * y.numel()
            total_ok += (logits.argmax(1) == y).sum().item()
            total_n += y.numel()
            if bi == len(train_loader) or bi % 50 == 0:
                write(f"\tBatch({bi}/{len(train_loader)}) done. Loss: {loss.item():.6f}  lr:{lr:.6f}")
        tr_loss = total_loss / total_n
        tr_acc = total_ok / total_n * 100
        write(f"\tMean training loss: {tr_loss:.10f}.")
        write(f"\tMean training acc: {tr_acc:.4f}")
        p1, p5, logits_np, labels_np, sigs_np = evaluate(model, test_loader, device, amp=amp)
        write(f"Epoch {ep}, Test, Evaluation: prec1 {p1:.4f}, prec5 {p5:.4f}")
        np.savez(os.path.join(args.workdir, "test_logits.npz"),
                 logits=logits_np, labels=labels_np, sigs=sigs_np,
                 epoch=np.array([ep], dtype=np.int64))
        if p1 > best:
            best, best_ep = p1, ep
            best_payload = (logits_np, labels_np, sigs_np)
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "best_acc": best,
                        "params_m": params_m}, os.path.join(args.workdir, "best_model.pt"))
            np.savez(os.path.join(args.workdir, "best_logits.npz"),
                     logits=logits_np, labels=labels_np, sigs=sigs_np,
                     epoch=np.array([ep], dtype=np.int64))
        write(f"ep{ep:3d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.2f}%  te_acc={p1:.2f}%  te_p5={p5:.2f}%  best={best:.2f}% @ ep{best_ep}  dt={time.time()-t0:.1f}s")
        if best >= args.stop_at and ep - best_ep >= args.stop_patience:
            write(f"EARLY_STOP best={best:.2f}% @ ep{best_ep}")
            break
    if best_payload is not None:
        logits_np, labels_np, sigs_np = best_payload
        np.savez(os.path.join(args.workdir, "test_logits.npz"),
                 logits=logits_np, labels=labels_np, sigs=sigs_np,
                 epoch=np.array([best_ep], dtype=np.int64))
    write(f"FINAL best={best:.2f}% @ ep{best_ep}")
    log.close()


if __name__ == "__main__":
    main()
