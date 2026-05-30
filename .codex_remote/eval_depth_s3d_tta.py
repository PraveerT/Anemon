"""Evaluate a trained S3D depth checkpoint with spatial/temporal TTA."""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, "/notebooks/Anemon")
import train_depth_s3d as td  # noqa: E402


class NvDataTTA(td.NvData):
    def __init__(self, *args, crop_rect=None, flip=False, temporal_start=None, **kwargs):
        self._crop_rect = crop_rect
        self._flip = flip
        self._temporal_start = temporal_start
        super().__init__(*args, **kwargs)

    def transform_params(self, resize=(320, 240), crop_size=224, flip=0.0):
        if self._crop_rect is None:
            left = (resize[0] - crop_size) // 2
            top = (resize[1] - crop_size) // 2
            rect = (left, top, left + crop_size, top + crop_size)
        else:
            rect = self._crop_rect
        return rect, self._flip

    def get_sl(self, clip):
        n = int(clip)
        sn = self.sample_duration
        if self._temporal_start is None:
            return super().get_sl(clip)
        span = min(n, sn)
        max_start = max(0, n - span)
        start = min(max_start, int(round(self._temporal_start * max_start)))
        return np.linspace(start, start + span - 1, sn).round().astype(np.int64)


def args_from_ckpt(ckpt_args, cli):
    d = dict(ckpt_args or {})
    d.setdefault("data", cli.data)
    d.setdefault("splits", cli.splits)
    d.setdefault("workdir", cli.workdir)
    d.setdefault("batch_size", cli.batch_size)
    d.setdefault("workers", cli.workers)
    d.setdefault("frames", cli.frames)
    d.setdefault("size", cli.size)
    d.setdefault("epochs", 1)
    d.setdefault("freeze_epochs", 0)
    d.setdefault("no_pretrained", False)
    d.setdefault("dropout", 0.45)
    d.setdefault("use_garr", False)
    return SimpleNamespace(**d)


def make_loader(args, crop_rect, flip, temporal_start):
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
        rotated=0.0,
        angle="(0, 0)",
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
        epochs=1,
        init_epochs=0,
        DEBUG=False,
        MultiLoss=False,
        pretrained=False,
        phase="valid",
    )
    ds = NvDataTTA(
        ns,
        ground_truth=os.path.join(args.splits, "valid.txt"),
        modality="depth",
        phase="valid",
        crop_rect=crop_rect,
        flip=flip,
        temporal_start=temporal_start,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=True,
                      persistent_workers=args.workers > 0)


def topk(logits, labels):
    pred = logits.topk(5, dim=1).indices
    p1 = (pred[:, :1] == labels[:, None]).any(1).float().mean().item() * 100.0
    p5 = (pred[:, :5] == labels[:, None]).any(1).float().mean().item() * 100.0
    return p1, p5


@torch.no_grad()
def eval_view(model, loader, device, use_garr=False):
    logits_all, labels_all, paths_all = [], [], []
    model.eval()
    for clip, garr, label, path in loader:
        clip = td.normalize_clip(clip.to(device, non_blocking=True), garr=garr, use_garr=use_garr)
        with autocast():
            logits = model(clip)
        logits_all.append(logits.float().cpu())
        labels_all.append(label.long())
        paths_all.extend(path if isinstance(path, list) else list(path))
    return torch.cat(logits_all), torch.cat(labels_all), np.array(paths_all, dtype=object)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="/notebooks/Anemon/experiments/work_dir/depth_small")
    p.add_argument("--data", default="/notebooks/cvpr_data")
    p.add_argument("--splits", default="/notebooks/cvpr_data/dataset_splits")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--temporal", choices=["center", "three"], default="center")
    p.add_argument("--flips", action="store_true")
    p.add_argument("--promote", action="store_true")
    return p.parse_args()


def main():
    cli = parse_args()
    ckpt = torch.load(os.path.join(cli.workdir, "best_model.pt"), map_location="cpu")
    args = args_from_ckpt(ckpt.get("args", {}), cli)
    args.batch_size = cli.batch_size
    args.workers = cli.workers
    args.frames = cli.frames
    args.size = cli.size
    model = td.build_model(args).cuda()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    lim = 256 - args.size
    crops = [
        (32, 32),
        (0, 0),
        (lim, 0),
        (0, lim),
        (lim, lim),
        (lim // 2, lim // 2),
    ]
    # Preserve the original validation crop first; remove duplicates.
    seen, crop_rects = set(), []
    for left, top in crops:
        left = max(0, min(lim, left))
        top = max(0, min(lim, top))
        key = (left, top)
        if key not in seen:
            seen.add(key)
            crop_rects.append((left, top, left + args.size, top + args.size))
    temporal = [None] if cli.temporal == "center" else [0.0, 0.5, 1.0]
    flips = [False, True] if cli.flips else [False]

    sum_logits, labels, paths = None, None, None
    n = 0
    for crop in crop_rects:
        for ts in temporal:
            for flip in flips:
                loader = make_loader(args, crop, flip, ts)
                logits, labels, paths = eval_view(model, loader, "cuda", use_garr=getattr(args, "use_garr", False))
                p1, p5 = topk(logits, labels)
                print(f"view crop={crop} temporal={ts} flip={flip} p1={p1:.4f} p5={p5:.4f}", flush=True)
                sum_logits = logits if sum_logits is None else sum_logits + logits
                n += 1
    avg_logits = sum_logits / n
    p1, p5 = topk(avg_logits, labels)
    print(f"TTA views={n} prec1={p1:.4f} prec5={p5:.4f}", flush=True)
    out = os.path.join(cli.workdir, "tta_logits.npz")
    np.savez(out, logits=avg_logits.numpy(), labels=labels.numpy(), paths=paths, sigs=paths,
             views=np.array([n], dtype=np.int64), prec1=np.array([p1], dtype=np.float32))

    base = np.load(os.path.join(cli.workdir, "best_logits.npz"), allow_pickle=True)
    base_p1 = (base["logits"].argmax(1) == base["labels"]).mean() * 100.0
    if cli.promote and p1 > base_p1:
        np.savez(os.path.join(cli.workdir, "test_logits.npz"),
                 logits=avg_logits.numpy(), labels=labels.numpy(), paths=paths, sigs=paths,
                 views=np.array([n], dtype=np.int64), prec1=np.array([p1], dtype=np.float32))
        with open(os.path.join(cli.workdir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(f"TTA promoted: prec1 {p1:.4f}, prec5 {p5:.4f}, views {n}\n")
        print(f"promoted over base {base_p1:.4f}", flush=True)
    else:
        print(f"not promoted base={base_p1:.4f}", flush=True)


if __name__ == "__main__":
    main()
