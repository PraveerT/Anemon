"""Fine-tune a compact S3D video model on NvGesture depth frames.

This intentionally avoids DSN checkpoints. The only pretraining used is the
standard torchvision S3D Kinetics-400 initialization, then the model is trained
on depth files through the MotionRGBD NvData depth loader.
"""
import argparse
import importlib
import json
import math
import os
import random
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models.video import S3D_Weights, s3d


MOTION_ROOT = "/notebooks/MotionRGBD"
LIB_ROOT = os.path.join(MOTION_ROOT, "lib")
DATASETS_ROOT = os.path.join(LIB_ROOT, "datasets")

# Avoid MotionRGBD/lib/__init__.py, which imports the DSN stack and optional
# DSN-only dependencies. We only need the dataset package here.
lib_pkg = types.ModuleType("lib")
lib_pkg.__path__ = [LIB_ROOT]
sys.modules.setdefault("lib", lib_pkg)
datasets_pkg = types.ModuleType("lib.datasets")
datasets_pkg.__path__ = [DATASETS_ROOT]
sys.modules.setdefault("lib.datasets", datasets_pkg)
NvData = importlib.import_module("lib.datasets.NvGesture").NvData


KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/notebooks/cvpr_data")
    p.add_argument("--splits", default="/notebooks/cvpr_data/dataset_splits")
    p.add_argument("--workdir", default="/notebooks/Anemon/experiments/work_dir/depth_small")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--head-lr", type=float, default=6.0e-4)
    p.add_argument("--min-lr", type=float, default=2.0e-6)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.45)
    p.add_argument("--use-garr", action="store_true", help="append MotionRGBD dynamic depth map as a 4th input channel")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--stop-at", type=float, default=89.0)
    p.add_argument("--stop-patience", type=int, default=5)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def make_dataset(args, phase):
    ns = SimpleNamespace(
        data=args.data,
        splits=args.splits,
        dataset="NvGesture",
        type="K",
        Network="S3DDepth",
        num_classes=25,
        sample_duration=args.frames,
        sample_size=args.size,
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        num_workers=args.workers,
        nprocs=1,
        local_rank=0,
        dist=False,
        flip=0.0,
        rotated=0.5 if phase == "train" else 0.0,
        angle="(-8, 8)",
        Blur=False,
        resize="(256, 256)",
        crop_size=args.size,
        low_frames=max(8, args.frames // 4),
        media_frames=max(16, args.frames // 2),
        high_frames=max(24, int(args.frames * 0.75)),
        w=4,
        temper=0.4,
        recoupling=False,
        knn_attention=0.7,
        sharpness=False,
        temp=[0.04, 0.07],
        frp=False,
        SEHeads=1,
        N=6,
        grad_clip=5.0,
        SYNC_BN=0,
        epoch=0,
        epochs=args.epochs,
        init_epochs=0,
        DEBUG=False,
        MultiLoss=False,
        pretrained=False,
        phase=phase,
    )
    split = "train.txt" if phase == "train" else "valid.txt"
    return NvData(ns, ground_truth=os.path.join(args.splits, split), modality="depth", phase=phase)


def build_model(args):
    weights = None if args.no_pretrained else S3D_Weights.KINETICS400_V1
    model = s3d(weights=weights)
    model.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    if args.use_garr:
        old = model.features[0][0][0]
        new = nn.Conv3d(
            4,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
            padding_mode=old.padding_mode,
        )
        with torch.no_grad():
            new.weight[:, :3].copy_(old.weight)
            new.weight[:, 3:4].copy_(old.weight.mean(dim=1, keepdim=True))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.features[0][0][0] = new
    model.classifier[0].p = args.dropout
    model.classifier[1] = nn.Conv3d(1024, 25, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    return model


def normalize_clip(clip, garr=None, use_garr=False):
    # NvData returns MotionRGBD-normalized frames in [-1, 1], shaped B,C,T,H,W.
    clip = clip.float().mul(0.5).add(0.5).clamp_(0.0, 1.0)
    mean = KINETICS_MEAN.to(clip.device, clip.dtype)
    std = KINETICS_STD.to(clip.device, clip.dtype)
    clip = (clip - mean) / std
    if use_garr:
        if garr is None:
            raise ValueError("use_garr=True requires garr tensor")
        garr = garr.to(clip.device, non_blocking=True).float().clamp_(0.0, 1.0)
        garr = (garr - 0.5) / 0.25
        clip = torch.cat([clip, garr], dim=1)
    return clip


def set_trainable(model, train_features):
    for name, p in model.named_parameters():
        p.requires_grad = train_features or name.startswith("classifier.")


def set_lrs(opt, scale, args):
    for group in opt.param_groups:
        base = group["base_lr"]
        group["lr"] = args.min_lr + (base - args.min_lr) * scale


def lr_scale(epoch, epochs, warmup):
    if epoch <= warmup:
        return epoch / max(1, warmup)
    p = (epoch - warmup) / max(1, epochs - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def topk(logits, labels):
    pred = logits.topk(5, dim=1).indices
    p1 = (pred[:, :1] == labels[:, None]).any(1).float().mean().item() * 100.0
    p5 = (pred[:, :5] == labels[:, None]).any(1).float().mean().item() * 100.0
    return p1, p5


@torch.no_grad()
def evaluate(model, loader, device, amp=True, use_garr=False):
    model.eval()
    logits_all, labels_all, paths_all = [], [], []
    for clip, garr, label, path in loader:
        clip = normalize_clip(clip.to(device, non_blocking=True), garr=garr, use_garr=use_garr)
        label = label.to(device, non_blocking=True).long()
        with autocast(enabled=amp):
            logits = model(clip)
        logits_all.append(logits.float().cpu())
        labels_all.append(label.cpu())
        paths_all.extend(path if isinstance(path, list) else list(path))
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    p1, p5 = topk(logits, labels)
    return p1, p5, logits.numpy(), labels.numpy(), np.array(paths_all, dtype=object)


def main():
    args = build_args()
    os.makedirs(args.workdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_ds = make_dataset(args, "train")
    valid_ds = make_dataset(args, "valid")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda")
    model = build_model(args).to(device)
    set_trainable(model, train_features=args.freeze_epochs <= 0)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    opt = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "base_lr": args.lr},
            {"params": model.classifier.parameters(), "base_lr": args.head_lr},
        ],
        lr=args.lr,
        weight_decay=args.wd,
    )
    scaler = GradScaler(enabled=not args.no_amp)
    amp = not args.no_amp
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    log_path = os.path.join(args.workdir, "log.txt")
    log = open(log_path, "a", encoding="utf-8")

    def write(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    write(f"S3DDepth params={params_m:.3f}M trainable={trainable_m:.3f}M pretrained={not args.no_pretrained} use_garr={args.use_garr}")
    write(f"train clips: {len(train_ds)}  valid clips: {len(valid_ds)}")
    write(f"batch={args.batch_size} frames={args.frames} size={args.size} lr={args.lr} head_lr={args.head_lr} wd={args.wd}")
    with open(os.path.join(args.workdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"params_m": params_m}, f, indent=2)

    best, best_ep, best_payload = 0.0, 0, None
    for ep in range(1, args.epochs + 1):
        if ep == args.freeze_epochs + 1:
            set_trainable(model, train_features=True)
            trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            write(f"unfroze_features trainable={trainable_m:.3f}M")
        scale = lr_scale(ep, args.epochs, args.warmup)
        set_lrs(opt, scale, args)

        model.train()
        t0 = time.time()
        total_loss = total_ok = total_n = 0
        write(f"Training epoch: {ep}")
        for bi, (clip, garr, label, _path) in enumerate(train_loader, 1):
            clip = normalize_clip(clip.to(device, non_blocking=True), garr=garr, use_garr=args.use_garr)
            label = label.to(device, non_blocking=True).long()
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logits = model(clip)
                loss = loss_fn(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * label.numel()
            total_ok += (logits.argmax(1) == label).sum().item()
            total_n += label.numel()
            if bi == len(train_loader) or bi % 50 == 0:
                write(f"\tBatch({bi}/{len(train_loader)}) done. Loss: {loss.item():.6f}  lr:{opt.param_groups[0]['lr']:.7f} head_lr:{opt.param_groups[1]['lr']:.7f}")

        tr_loss = total_loss / max(1, total_n)
        tr_acc = total_ok / max(1, total_n) * 100.0
        write(f"\tMean training loss: {tr_loss:.10f}.")
        write(f"\tMean training acc: {tr_acc:.4f}")
        p1, p5, logits_np, labels_np, paths_np = evaluate(model, valid_loader, device, amp=amp, use_garr=args.use_garr)
        write(f"Epoch {ep}, Test, Evaluation: prec1 {p1:.4f}, prec5 {p5:.4f}")
        np.savez(
            os.path.join(args.workdir, "test_logits.npz"),
            logits=logits_np,
            labels=labels_np,
            paths=paths_np,
            sigs=paths_np,
            epoch=np.array([ep], dtype=np.int64),
        )
        if p1 > best:
            best, best_ep = p1, ep
            best_payload = (logits_np, labels_np, paths_np)
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "best_acc": best,
                    "params_m": params_m,
                    "args": vars(args),
                },
                os.path.join(args.workdir, "best_model.pt"),
            )
            np.savez(
                os.path.join(args.workdir, "best_logits.npz"),
                logits=logits_np,
                labels=labels_np,
                paths=paths_np,
                sigs=paths_np,
                epoch=np.array([ep], dtype=np.int64),
            )
        write(f"ep{ep:3d}  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.2f}%  te_acc={p1:.2f}%  te_p5={p5:.2f}%  best={best:.2f}% @ ep{best_ep}  dt={time.time()-t0:.1f}s")
        if best >= args.stop_at and ep - best_ep >= args.stop_patience:
            write(f"EARLY_STOP best={best:.2f}% @ ep{best_ep}")
            break

    if best_payload is not None:
        logits_np, labels_np, paths_np = best_payload
        np.savez(
            os.path.join(args.workdir, "test_logits.npz"),
            logits=logits_np,
            labels=labels_np,
            paths=paths_np,
            sigs=paths_np,
            epoch=np.array([best_ep], dtype=np.int64),
        )
    write(f"FINAL best={best:.2f}% @ ep{best_ep}")
    log.close()


if __name__ == "__main__":
    main()
