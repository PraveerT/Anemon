"""Train a compact depth-only video model on NVGesture depth JPEG clips.

This branch intentionally does not use the pretrained DSN model or the
point-cloud files. It reads /notebooks/cvpr_data/depth frames directly, builds a
small cache, and trains a lightweight 2D frame encoder plus temporal
Transformer. The log format matches Anemon's sidepanel parser.
"""
import argparse
import copy
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


@dataclass
class ClipRec:
    rel: str
    nframes: int
    label: int

    @property
    def sig(self):
        return self.rel.strip("/")


def read_split(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            out.append(ClipRec(parts[0], int(parts[1]), int(parts[2])))
    return out


def resize_frame(path, size, bbox=None):
    img = Image.open(path).convert("L")
    if bbox is not None:
        img = img.crop(bbox)
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _foreground_bbox(frames, pad_frac=0.18):
    mask = np.zeros(frames[0].shape, dtype=bool)
    for a in frames:
        mask |= a > 0
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        h, w = frames[0].shape
        return (0, 0, w, h)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = frames[0].shape
    pad = int(max(y1 - y0, x1 - x0) * pad_frac) + 4
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    return (x0, y0, x1, y1)


def build_cache(split_path, phase, data_root, cache_dir, cache_size, fg_crop=False):
    recs = read_split(split_path)
    os.makedirs(cache_dir, exist_ok=True)
    crop_tag = "fg" if fg_crop else "all"
    base = os.path.join(cache_dir, f"{phase}_{crop_tag}_s{cache_size}")
    clip_path = base + ".npy"
    label_path = base + "_labels.npy"
    sig_path = base + "_sigs.npy"
    if os.path.isfile(clip_path) and os.path.isfile(label_path) and os.path.isfile(sig_path):
        clips = np.load(clip_path, mmap_mode="r")
        labels = np.load(label_path)
        sigs = np.load(sig_path, allow_pickle=True)
        return clips, labels, sigs

    max_frames = max(r.nframes for r in recs)
    clips = np.zeros((len(recs), max_frames, cache_size, cache_size), dtype=np.uint8)
    labels = np.array([r.label for r in recs], dtype=np.int64)
    sigs = np.array([r.sig for r in recs], dtype=object)
    t0 = time.time()
    for i, rec in enumerate(recs):
        d = os.path.join(data_root, rec.rel)
        if fg_crop:
            raw = [
                np.asarray(Image.open(os.path.join(d, f"{t:06d}.jpg")).convert("L"), dtype=np.uint8)
                for t in range(rec.nframes)
            ]
            bbox = _foreground_bbox(raw)
            for t, frame in enumerate(raw):
                img = Image.fromarray(frame).crop(bbox).resize((cache_size, cache_size), Image.BILINEAR)
                clips[i, t] = np.asarray(img, dtype=np.uint8)
        else:
            for t in range(rec.nframes):
                clips[i, t] = resize_frame(os.path.join(d, f"{t:06d}.jpg"), cache_size)
        if (i + 1) % 100 == 0 or i + 1 == len(recs):
            print(f"[cache] {phase}: {i + 1}/{len(recs)} clips in {time.time() - t0:.1f}s", flush=True)
    np.save(clip_path, clips)
    np.save(label_path, labels)
    np.save(sig_path, sigs)
    clips = np.load(clip_path, mmap_mode="r")
    return clips, labels, sigs


def _processed_pts_path(corr_root, corr_phase, sig):
    base = os.path.join(corr_root, corr_phase, sig, "sk_depth.avi")
    if not os.path.isdir(base):
        return None
    matches = [p for p in os.listdir(base) if p.endswith("_pts.npy")]
    if not matches:
        return None
    return os.path.join(base, sorted(matches)[0])


def _corr_target_from_pts(path, corr_frames, corr_points):
    arr = np.load(path, mmap_mode="r")
    xyz = np.asarray(arr[..., 4:7], dtype=np.float32)
    t_idx = np.linspace(0, xyz.shape[0] - 1, corr_frames).round().astype(np.int64)
    p_idx = np.linspace(0, xyz.shape[1] - 1, corr_points).round().astype(np.int64)
    out = xyz[t_idx][:, p_idx]
    center = out.reshape(-1, 3).mean(axis=0, keepdims=True)
    scale = np.sqrt(((out.reshape(-1, 3) - center) ** 2).sum(axis=1).mean())
    out = (out - center.reshape(1, 1, 3)) / max(float(scale), 1.0)
    return np.clip(out, -3.0, 3.0).astype(np.float32)


def build_corr_cache(split_path, phase, corr_root, cache_dir, corr_frames, corr_points):
    if not corr_root:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, f"{phase}_corr_f{corr_frames}_p{corr_points}")
    corr_path = base + ".npy"
    if os.path.isfile(corr_path):
        return np.load(corr_path, mmap_mode="r")

    recs = read_split(split_path)
    corr_phase = "test" if phase in ("valid", "val", "test") else "train"
    targets = np.zeros((len(recs), corr_frames, corr_points, 3), dtype=np.float32)
    missing = 0
    t0 = time.time()
    for i, rec in enumerate(recs):
        pts_path = _processed_pts_path(corr_root, corr_phase, rec.sig)
        if pts_path is None:
            missing += 1
        else:
            targets[i] = _corr_target_from_pts(pts_path, corr_frames, corr_points)
        if (i + 1) % 200 == 0 or i + 1 == len(recs):
            print(
                f"[corr-cache] {phase}: {i + 1}/{len(recs)} clips "
                f"missing={missing} in {time.time() - t0:.1f}s",
                flush=True,
            )
    np.save(corr_path, targets)
    return np.load(corr_path, mmap_mode="r")


class DepthClipDataset(Dataset):
    def __init__(self, clips, labels, sigs, frames=32, crop_size=128, train=False,
                 input_mode="raw3", hflip_prob=0.0, cutout_prob=0.0,
                 corr_targets=None):
        self.clips = clips
        self.labels = labels
        self.sigs = np.asarray(sigs)
        self.corr_targets = corr_targets
        self.frames = int(frames)
        self.crop_size = int(crop_size)
        self.train = bool(train)
        self.input_mode = input_mode
        self.hflip_prob = float(hflip_prob)
        self.cutout_prob = float(cutout_prob)
        self.cache_size = int(clips.shape[-1])
        self.max_frames = int(clips.shape[1])

    def __len__(self):
        return len(self.labels)

    def _time_indices(self):
        if self.train:
            span = random.randint(max(self.frames, int(self.max_frames * 0.65)), self.max_frames)
            start = random.randint(0, self.max_frames - span)
            idx = np.linspace(start, start + span - 1, self.frames)
            jitter = np.random.uniform(-0.45, 0.45, size=self.frames)
            idx = np.clip(np.rint(idx + jitter), 0, self.max_frames - 1).astype(np.int64)
            idx.sort()
            return idx
        return np.linspace(0, self.max_frames - 1, self.frames).round().astype(np.int64)

    def _spatial_crop(self, x):
        if self.cache_size == self.crop_size:
            return x
        lim = self.cache_size - self.crop_size
        if self.train:
            y = random.randint(0, lim)
            z = random.randint(0, lim)
        else:
            y = lim // 2
            z = lim // 2
        return x[:, y:y + self.crop_size, z:z + self.crop_size]

    def __getitem__(self, idx):
        inds = self._time_indices()
        x = np.asarray(self.clips[idx, inds], dtype=np.uint8)
        x = self._spatial_crop(x)
        if self.train and random.random() < self.hflip_prob:
            x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x.copy()).float() / 255.0
        if self.input_mode in ("raw3", "raw3_kinetics"):
            clip = x.unsqueeze(0).repeat(3, 1, 1, 1)
            if self.train and self.cutout_prob > 0 and random.random() < self.cutout_prob:
                span = random.randint(2, 5)
                st = random.randint(0, max(0, self.frames - span))
                clip[:, st:st + span] = 0
            if self.input_mode == "raw3_kinetics":
                mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=clip.dtype).view(3, 1, 1, 1)
                std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=clip.dtype).view(3, 1, 1, 1)
            else:
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=clip.dtype).view(3, 1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], dtype=clip.dtype).view(3, 1, 1, 1)
            out = ((clip - mean) / std, int(self.labels[idx]), str(self.sigs[idx]))
            if self.corr_targets is not None:
                corr = torch.from_numpy(np.asarray(self.corr_targets[idx]).copy()).float()
                return out + (corr,)
            return out

        mask = (x > 0.01).float()
        denom = mask.sum().clamp_min(1.0)
        mean = (x * mask).sum() / denom
        var = (((x - mean) * mask) ** 2).sum() / denom
        centered = ((x - mean) / torch.sqrt(var + 1e-4)).clamp(-3.0, 3.0) / 3.0
        centered = centered * mask
        diff = torch.zeros_like(x)
        diff[1:] = (x[1:] - x[:-1]).abs()
        if self.train and self.cutout_prob > 0 and random.random() < self.cutout_prob:
            # Temporal cutout: drop a short span in all channels.
            span = random.randint(2, 5)
            st = random.randint(0, max(0, self.frames - span))
            centered[st:st + span] = 0
            mask[st:st + span] = 0
            diff[st:st + span] = 0
        clip = torch.stack([centered, mask, diff * 2.0], dim=0)  # C,T,H,W
        out = (clip, int(self.labels[idx]), str(self.sigs[idx]))
        if self.corr_targets is not None:
            corr = torch.from_numpy(np.asarray(self.corr_targets[idx]).copy()).float()
            return out + (corr,)
        return out


class SepBlock(nn.Module):
    def __init__(self, channels, expansion=2, drop=0.0):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Dropout2d(drop),
        )

    def forward(self, x):
        return x + self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            SepBlock(out_c),
        )

    def forward(self, x):
        return self.net(x)


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim) * 0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        score = torch.matmul(torch.tanh(self.proj(x)), self.q)
        w = score.softmax(dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)


class DepthTinyTemporal(nn.Module):
    def __init__(self, num_classes=25, dim=256, frames=32, dropout=0.25):
        super().__init__()
        self.frame = nn.Sequential(
            nn.Conv2d(3, 48, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            SepBlock(48),
            DownBlock(48, 96),
            DownBlock(96, 160),
            SepBlock(160, drop=0.05),
            DownBlock(160, dim),
            SepBlock(dim, drop=0.05),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pos = nn.Parameter(torch.zeros(1, frames, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 3,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.pool = AttnPool(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        # x: B,C,T,H,W
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = self.frame(y).flatten(1).reshape(b, t, -1)
        y = y + self.pos[:, :t]
        y = self.temporal(y)
        y = self.pool(y)
        return self.head(y)


class ResNet18Temporal(nn.Module):
    def __init__(self, num_classes=25, frames=32, hidden=256, dropout=0.35, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)
        self.frame = nn.Sequential(*list(base.children())[:-1])
        self.gru = nn.GRU(512, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.pool = AttnPool(hidden * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def backbone_parameters(self):
        return self.frame.parameters()

    def head_parameters(self):
        for module in (self.gru, self.pool, self.head):
            yield from module.parameters()

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = self.frame(y).flatten(1).reshape(b, t, 512)
        y, _ = self.gru(y)
        y = self.pool(y)
        return self.head(y)


class MC3Depth(nn.Module):
    def __init__(self, num_classes=25, pretrained=True, kind="mc3_18", head_hidden=0, head_dropout=0.0):
        super().__init__()
        if kind == "r2plus1d_18":
            from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
            weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
            self.net = r2plus1d_18(weights=weights)
        else:
            from torchvision.models.video import MC3_18_Weights, mc3_18
            weights = MC3_18_Weights.KINETICS400_V1 if pretrained else None
            self.net = mc3_18(weights=weights)
        in_features = self.net.fc.in_features
        self.feature_dim = in_features
        if head_hidden > 0:
            self.net.fc = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(in_features, head_hidden),
                nn.GELU(),
                nn.LayerNorm(head_hidden),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.net.fc = nn.Linear(in_features, num_classes)

    def backbone_parameters(self):
        for name, p in self.net.named_parameters():
            if not name.startswith("fc."):
                yield p

    def head_parameters(self):
        yield from self.net.fc.parameters()

    def features(self, x):
        y = self.net.stem(x)
        y = self.net.layer1(y)
        y = self.net.layer2(y)
        y = self.net.layer3(y)
        y = self.net.layer4(y)
        y = self.net.avgpool(y)
        return torch.flatten(y, 1)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        logits = self.net.fc(feat)
        if return_features:
            return logits, feat
        return logits


class CorrQCCDecoder(nn.Module):
    def __init__(self, feature_dim, frames=8, points=64, hidden=512, dropout=0.10):
        super().__init__()
        self.frames = int(frames)
        self.points = int(points)
        self.point_decoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.frames * self.points * 3),
        )
        self.step_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, max(1, self.frames - 1) * 3),
        )
        self.skip_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, max(1, self.frames - 2) * 3),
        )

    def forward(self, feat):
        pts = self.point_decoder(feat).view(feat.size(0), self.frames, self.points, 3)
        step = axis_angle_to_quat(self.step_head(feat).view(feat.size(0), self.frames - 1, 3))
        skip = axis_angle_to_quat(self.skip_head(feat).view(feat.size(0), self.frames - 2, 3))
        return pts, step, skip


class QCCResidualHead(nn.Module):
    def __init__(self, feature_dim, num_classes=25, frames=8, hidden=256, dropout=0.10, scale_init=0.03):
        super().__init__()
        self.frames = int(frames)
        self.shared = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.logit_head = nn.Linear(hidden, num_classes)
        self.step_head = nn.Linear(hidden, max(1, self.frames - 1) * 3)
        self.skip_head = nn.Linear(hidden, max(1, self.frames - 2) * 3)
        self.logit_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, feat):
        h = self.shared(feat)
        residual = self.logit_scale * self.logit_head(h)
        step = axis_angle_to_quat(self.step_head(h).view(feat.size(0), self.frames - 1, 3))
        skip = axis_angle_to_quat(self.skip_head(h).view(feat.size(0), self.frames - 2, 3))
        return residual, step, skip


def update_ema(model, ema, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(msd[k], alpha=1.0 - decay)
            else:
                v.copy_(msd[k])


def update_ema_module(model, ema, decay):
    if model is None or ema is None:
        return
    update_ema(model, ema, decay)


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def lr_for_epoch(epoch, args):
    if epoch <= args.warmup_epochs:
        return args.lr * epoch / max(1, args.warmup_epochs)
    p = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * p))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)


def apply_mixup(clip, label, alpha, prob):
    if alpha <= 0.0 or prob <= 0.0 or random.random() >= prob or clip.size(0) < 2:
        return clip, label, None, 1.0, None
    lam = float(np.random.beta(alpha, alpha))
    order = torch.randperm(clip.size(0), device=clip.device)
    mixed = clip.mul(lam).add_(clip[order], alpha=1.0 - lam)
    return mixed, label, label[order], lam, order


def axis_angle_to_quat(v):
    angle = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = v / angle
    half = 0.5 * angle
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_distance_loss(a, b):
    a = quat_normalize(a.float())
    b = quat_normalize(b.float())
    dot = (a * b).sum(dim=-1).abs().clamp(0.0, 1.0)
    return (1.0 - dot).mean()


def _rotmat_to_quat(R):
    # Stable batched conversion for Kabsch rotations.
    R = R.float()
    qw = torch.sqrt((1.0 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp_min(1e-8)) * 0.5
    qx = torch.sign(R[..., 2, 1] - R[..., 1, 2]) * torch.sqrt(
        (1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]).clamp_min(1e-8)
    ) * 0.5
    qy = torch.sign(R[..., 0, 2] - R[..., 2, 0]) * torch.sqrt(
        (1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2]).clamp_min(1e-8)
    ) * 0.5
    qz = torch.sign(R[..., 1, 0] - R[..., 0, 1]) * torch.sqrt(
        (1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]).clamp_min(1e-8)
    ) * 0.5
    return quat_normalize(torch.stack([qw, qx, qy, qz], dim=-1))


def _kabsch_quat(src, dst):
    src = src.float()
    dst = dst.float()
    a = src - src.mean(dim=-2, keepdim=True)
    b = dst - dst.mean(dim=-2, keepdim=True)
    h = a.transpose(-1, -2).matmul(b) / max(1, src.shape[-2])
    u, _s, vh = torch.linalg.svd(h)
    v = vh.transpose(-1, -2)
    ut = u.transpose(-1, -2)
    det = torch.det(v.matmul(ut))
    fix = torch.ones((*det.shape, 3), device=src.device, dtype=src.dtype)
    fix[..., 2] = torch.where(det < 0, -1.0, 1.0)
    r = v.matmul(torch.diag_embed(fix)).matmul(ut)
    return _rotmat_to_quat(r)


def target_corr_quats(corr):
    with torch.no_grad(), autocast(enabled=False):
        corr = corr.float()
        step = _kabsch_quat(corr[:, :-1], corr[:, 1:])
        skip = _kabsch_quat(corr[:, :-2], corr[:, 2:])
    return step, skip


def corr_qcc_loss(pred_pts, pred_step, pred_skip, target_pts):
    point_loss = F.smooth_l1_loss(pred_pts.float(), target_pts.float())
    tgt_step, tgt_skip = target_corr_quats(target_pts)
    step_loss = quat_distance_loss(pred_step, tgt_step)
    skip_loss = quat_distance_loss(pred_skip, tgt_skip)
    cyc = quat_mul(pred_step[:, 1:], pred_step[:, :-1])
    cycle_loss = quat_distance_loss(cyc, pred_skip)
    return point_loss, step_loss + skip_loss, cycle_loss


def qcc_residual_loss(pred_step, pred_skip, target_pts):
    tgt_step, tgt_skip = target_corr_quats(target_pts)
    step_loss = quat_distance_loss(pred_step, tgt_step)
    skip_loss = quat_distance_loss(pred_skip, tgt_skip)
    cyc = quat_mul(pred_step[:, 1:], pred_step[:, :-1])
    cycle_loss = quat_distance_loss(cyc, pred_skip)
    return step_loss + skip_loss, cycle_loss


def accuracy_topk(logits, labels, topk=(1, 5)):
    maxk = max(topk)
    pred = logits.topk(maxk, dim=1).indices
    out = []
    for k in topk:
        out.append((pred[:, :k] == labels[:, None]).any(dim=1).float().mean().item() * 100.0)
    return out


@torch.no_grad()
def evaluate(model, loader, device, amp=True, tta_flip=True, qcc_head=None):
    model.eval()
    if qcc_head is not None:
        qcc_head.eval()
    all_logits, all_labels, all_sigs = [], [], []
    for batch in loader:
        clip, label, sig = batch[:3]
        clip = clip.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with autocast(enabled=amp):
            if qcc_head is None:
                logits = model(clip)
            else:
                logits, feat = model(clip, return_features=True)
                logits = logits + qcc_head(feat.float())[0]
            if tta_flip:
                flip = torch.flip(clip, dims=[-1])
                if qcc_head is None:
                    flip_logits = model(flip)
                else:
                    flip_logits, flip_feat = model(flip, return_features=True)
                    flip_logits = flip_logits + qcc_head(flip_feat.float())[0]
                logits = 0.5 * (logits + flip_logits)
        all_logits.append(logits.float().cpu())
        all_labels.append(label.cpu())
        all_sigs.extend(sig)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    p1, p5 = accuracy_topk(logits, labels)
    return p1, p5, logits.numpy(), labels.numpy(), np.array(all_sigs, dtype=object)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/notebooks/cvpr_data/depth")
    ap.add_argument("--split-root", default="/notebooks/cvpr_data/dataset_splits")
    ap.add_argument("--workdir", default="/notebooks/Anemon/experiments/work_dir/depth_small")
    ap.add_argument("--cache-dir", default="/notebooks/Anemon/dataset/Nvidia/Processed/depth_small_cache")
    ap.add_argument("--corr-root", default="/notebooks/Anemon/dataset/Nvidia/Processed")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--cache-size", type=int, default=128)
    ap.add_argument("--crop-size", type=int, default=112)
    ap.add_argument("--fg-crop", action="store_true")
    ap.add_argument("--arch", choices=["tiny", "resnet18_temporal", "mc3_18", "r2plus1d_18"], default="resnet18_temporal")
    ap.add_argument("--input-mode", choices=["raw3", "raw3_kinetics", "motion3"], default="raw3")
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--warmstart", default="")
    ap.add_argument("--head-hidden", type=int, default=0)
    ap.add_argument("--head-dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr", type=float, default=2e-5)
    ap.add_argument("--min-lr", type=float, default=8e-6)
    ap.add_argument("--warmup-epochs", type=int, default=8)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup-alpha", type=float, default=0.0)
    ap.add_argument("--mixup-prob", type=float, default=0.0)
    ap.add_argument("--corr-aux", action="store_true")
    ap.add_argument("--corr-frames", type=int, default=8)
    ap.add_argument("--corr-points", type=int, default=64)
    ap.add_argument("--corr-hidden", type=int, default=512)
    ap.add_argument("--corr-dropout", type=float, default=0.10)
    ap.add_argument("--corr-weight", type=float, default=0.02)
    ap.add_argument("--qcc-weight", type=float, default=0.01)
    ap.add_argument("--qcc-cycle-weight", type=float, default=0.01)
    ap.add_argument("--aux-start-epoch", type=int, default=1)
    ap.add_argument("--aux-ramp-epochs", type=int, default=1)
    ap.add_argument("--qcc-residual", action="store_true")
    ap.add_argument("--qcc-hidden", type=int, default=256)
    ap.add_argument("--qcc-dropout", type=float, default=0.10)
    ap.add_argument("--qcc-logit-scale-init", type=float, default=0.03)
    ap.add_argument("--freeze-main-epochs", type=int, default=0)
    ap.add_argument("--hflip-prob", type=float, default=0.0)
    ap.add_argument("--cutout-prob", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stop-at", type=float, default=89.0)
    ap.add_argument("--stop-patience", type=int, default=5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_cache = build_cache(
        os.path.join(args.split_root, "train.txt"), "train",
        args.data_root, args.cache_dir, args.cache_size, fg_crop=args.fg_crop,
    )
    valid_cache = build_cache(
        os.path.join(args.split_root, "valid.txt"), "valid",
        args.data_root, args.cache_dir, args.cache_size, fg_crop=args.fg_crop,
    )
    train_corr = valid_corr = None
    if args.corr_aux or args.qcc_residual:
        train_corr = build_corr_cache(
            os.path.join(args.split_root, "train.txt"), "train",
            args.corr_root, args.cache_dir, args.corr_frames, args.corr_points,
        )
        valid_corr = build_corr_cache(
            os.path.join(args.split_root, "valid.txt"), "valid",
            args.corr_root, args.cache_dir, args.corr_frames, args.corr_points,
        )
    train_ds = DepthClipDataset(
        *train_cache, frames=args.frames, crop_size=args.crop_size, train=True,
        input_mode=args.input_mode, hflip_prob=args.hflip_prob, cutout_prob=args.cutout_prob,
        corr_targets=train_corr,
    )
    valid_ds = DepthClipDataset(
        *valid_cache, frames=args.frames, crop_size=args.crop_size, train=False,
        input_mode=args.input_mode, corr_targets=valid_corr,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arch == "tiny":
        model = DepthTinyTemporal(frames=args.frames).to(device)
    elif args.arch in ("mc3_18", "r2plus1d_18"):
        model = MC3Depth(
            pretrained=args.pretrained,
            kind=args.arch,
            head_hidden=args.head_hidden,
            head_dropout=args.head_dropout,
        ).to(device)
    else:
        model = ResNet18Temporal(frames=args.frames, pretrained=args.pretrained).to(device)
    if args.warmstart:
        ckpt = torch.load(args.warmstart, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
    corr_decoder = None
    qcc_head = None
    if args.corr_aux:
        if not hasattr(model, "feature_dim"):
            raise RuntimeError("--corr-aux currently requires mc3_18 or r2plus1d_18")
        corr_decoder = CorrQCCDecoder(
            model.feature_dim,
            frames=args.corr_frames,
            points=args.corr_points,
            hidden=args.corr_hidden,
            dropout=args.corr_dropout,
        ).to(device)
    if args.qcc_residual:
        if not hasattr(model, "feature_dim"):
            raise RuntimeError("--qcc-residual currently requires mc3_18 or r2plus1d_18")
        qcc_head = QCCResidualHead(
            model.feature_dim,
            frames=args.corr_frames,
            hidden=args.qcc_hidden,
            dropout=args.qcc_dropout,
            scale_init=args.qcc_logit_scale_init,
        ).to(device)
    ema = copy.deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)
    qcc_ema = copy.deepcopy(qcc_head).to(device) if qcc_head is not None else None
    if qcc_ema is not None:
        for p in qcc_ema.parameters():
            p.requires_grad_(False)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    corr_params_m = 0.0 if corr_decoder is None else sum(p.numel() for p in corr_decoder.parameters()) / 1e6
    qcc_params_m = 0.0 if qcc_head is None else sum(p.numel() for p in qcc_head.parameters()) / 1e6
    if hasattr(model, "backbone_parameters"):
        opt = torch.optim.AdamW(
            [
                {"params": list(model.backbone_parameters()), "lr": args.backbone_lr},
                {"params": list(model.head_parameters()), "lr": args.lr},
            ],
            weight_decay=args.wd,
        )
        opt.param_groups[0]["lr_scale"] = args.backbone_lr / args.lr
        opt.param_groups[1]["lr_scale"] = 1.0
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if corr_decoder is not None:
        opt.add_param_group({
            "params": list(corr_decoder.parameters()),
            "lr": args.lr,
            "lr_scale": 1.0,
            "weight_decay": args.wd,
        })
    if qcc_head is not None:
        opt.add_param_group({
            "params": list(qcc_head.parameters()),
            "lr": args.lr,
            "lr_scale": 1.0,
            "weight_decay": args.wd,
        })
    scaler = GradScaler(enabled=not args.no_amp)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp = not args.no_amp
    log_path = os.path.join(args.workdir, "log.txt")
    log = open(log_path, "a", encoding="utf-8")

    def write(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    write(f"{args.arch} params: {params_m:.3f}M pretrained={args.pretrained} corr_params={corr_params_m:.3f}M qcc_params={qcc_params_m:.3f}M total_params={params_m + corr_params_m + qcc_params_m:.3f}M")
    write(f"train clips: {len(train_ds)}  valid clips: {len(valid_ds)}")
    write(f"frames={args.frames} cache_size={args.cache_size} crop_size={args.crop_size} batch={args.batch_size} input={args.input_mode} fg_crop={args.fg_crop}")
    write(f"head_hidden={args.head_hidden} head_dropout={args.head_dropout:g}")
    write(f"lr={args.lr:g} backbone_lr={args.backbone_lr:g} min_lr={args.min_lr:g} warmup={args.warmup_epochs} wd={args.wd:g} ema={args.ema_decay:g} label_smoothing={args.label_smoothing:g} mixup_alpha={args.mixup_alpha:g} mixup_prob={args.mixup_prob:g}")
    write(f"corr_aux={args.corr_aux} corr_frames={args.corr_frames} corr_points={args.corr_points} corr_hidden={args.corr_hidden} corr_weight={args.corr_weight:g} qcc_weight={args.qcc_weight:g} qcc_cycle_weight={args.qcc_cycle_weight:g} aux_start={args.aux_start_epoch} aux_ramp={args.aux_ramp_epochs}")
    write(f"warmstart={args.warmstart or 'none'} qcc_residual={args.qcc_residual} qcc_hidden={args.qcc_hidden} qcc_dropout={args.qcc_dropout:g} qcc_logit_scale_init={args.qcc_logit_scale_init:g} freeze_main_epochs={args.freeze_main_epochs}")

    best_acc = 0.0
    best_ep = 0
    best_payload = None
    for ep in range(1, args.epochs + 1):
        lr = lr_for_epoch(ep, args)
        set_lr(opt, lr)
        if corr_decoder is not None and ep >= args.aux_start_epoch:
            aux_scale = min(1.0, (ep - args.aux_start_epoch + 1) / max(1, args.aux_ramp_epochs))
        elif qcc_head is not None and ep >= args.aux_start_epoch:
            aux_scale = min(1.0, (ep - args.aux_start_epoch + 1) / max(1, args.aux_ramp_epochs))
        else:
            aux_scale = 0.0
        freeze_main = qcc_head is not None and ep <= args.freeze_main_epochs
        if qcc_head is not None:
            set_requires_grad(model, not freeze_main)
        write(f"Training epoch: {ep}")
        model.eval() if freeze_main else model.train()
        if corr_decoder is not None:
            corr_decoder.train()
        if qcc_head is not None:
            qcc_head.train()
        t0 = time.time()
        total_loss = 0.0
        total_aux = 0.0
        total_correct = 0
        total_seen = 0
        total_batches = len(train_loader)
        for bi, batch in enumerate(train_loader, 1):
            clip, label = batch[0], batch[1]
            corr = batch[3] if len(batch) > 3 else None
            clip = clip.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if corr is not None:
                corr = corr.to(device, non_blocking=True)
            clip, label_a, label_b, lam, order = apply_mixup(clip, label, args.mixup_alpha, args.mixup_prob)
            if corr is not None and order is not None and corr_decoder is not None:
                corr = corr.mul(lam).add_(corr[order], alpha=1.0 - lam)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                need_feat = (
                    (corr_decoder is not None and corr is not None and aux_scale > 0.0)
                    or qcc_head is not None
                )
                if need_feat:
                    logits, feat = model(clip, return_features=True)
                    if freeze_main:
                        logits = logits.detach()
                        feat = feat.detach()
                else:
                    logits = model(clip)
                    feat = None
                qcc_step = qcc_skip = None
                if qcc_head is not None:
                    residual, qcc_step, qcc_skip = qcc_head(feat.float())
                    logits = logits + residual
                if label_b is None:
                    loss = loss_fn(logits, label_a)
                else:
                    loss = lam * loss_fn(logits, label_a) + (1.0 - lam) * loss_fn(logits, label_b)
                aux_loss = torch.zeros((), device=device, dtype=loss.dtype)
                if corr_decoder is not None and corr is not None and aux_scale > 0.0:
                    pred_pts, pred_step, pred_skip = corr_decoder(feat.float())
                    point_loss, quat_loss, cycle_loss = corr_qcc_loss(pred_pts, pred_step, pred_skip, corr)
                    aux_loss = (
                        args.corr_weight * point_loss
                        + args.qcc_weight * quat_loss
                        + args.qcc_cycle_weight * cycle_loss
                    )
                    loss = loss + aux_scale * aux_loss
                if qcc_head is not None and corr is not None and aux_scale > 0.0 and order is None:
                    quat_loss, cycle_loss = qcc_residual_loss(qcc_step, qcc_skip, corr)
                    aux_loss = args.qcc_weight * quat_loss + args.qcc_cycle_weight * cycle_loss
                    loss = loss + aux_scale * aux_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            clip_params = list(model.parameters())
            if corr_decoder is not None:
                clip_params += list(corr_decoder.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, 5.0)
            scaler.step(opt)
            scaler.update()
            update_ema(model, ema, args.ema_decay)
            update_ema_module(qcc_head, qcc_ema, args.ema_decay)
            bs = label.numel()
            total_loss += loss.item() * bs
            total_aux += float((aux_scale * aux_loss).detach().float().item()) * bs
            total_correct += (logits.argmax(1) == label).sum().item()
            total_seen += bs
            if bi == total_batches or bi % 50 == 0:
                write(f"\tBatch({bi}/{total_batches}) done. Loss: {loss.item():.6f}  lr:{lr:.6f}")

        tr_loss = total_loss / max(1, total_seen)
        tr_aux = total_aux / max(1, total_seen)
        tr_acc = total_correct / max(1, total_seen) * 100.0
        write(f"\tMean training loss: {tr_loss:.10f}.")
        if corr_decoder is not None or qcc_head is not None:
            write(f"\tMean auxiliary loss: {tr_aux:.10f}.")
        write(f"\tMean training acc: {tr_acc:.4f}")

        p1, p5, logits_np, labels_np, sigs_np = evaluate(
            ema, valid_loader, device, amp=amp, tta_flip=not args.no_tta, qcc_head=qcc_ema,
        )
        write(f"Epoch {ep}, Test, Evaluation: prec1 {p1:.4f}, prec5 {p5:.4f}")
        np.savez(
            os.path.join(args.workdir, "test_logits.npz"),
            logits=logits_np, labels=labels_np, sigs=sigs_np,
            epoch=np.array([ep], dtype=np.int64),
        )
        if p1 > best_acc:
            best_acc = p1
            best_ep = ep
            best_payload = (logits_np, labels_np, sigs_np)
            state = {"epoch": ep, "model_state_dict": ema.state_dict(), "best_acc": best_acc, "params_m": params_m}
            if corr_decoder is not None:
                state["corr_state_dict"] = corr_decoder.state_dict()
                state["corr_params_m"] = corr_params_m
            if qcc_ema is not None:
                state["qcc_state_dict"] = qcc_ema.state_dict()
                state["qcc_params_m"] = qcc_params_m
            torch.save(state, os.path.join(args.workdir, "best_model.pt"))
            np.savez(
                os.path.join(args.workdir, "best_logits.npz"),
                logits=logits_np, labels=labels_np, sigs=sigs_np,
                epoch=np.array([ep], dtype=np.int64),
            )
        aux_txt = f"  aux_loss={tr_aux:.4f}" if (corr_decoder is not None or qcc_head is not None) else ""
        write(f"ep{ep:3d}  tr_loss={tr_loss:.4f}{aux_txt}  tr_acc={tr_acc:.2f}%  te_acc={p1:.2f}%  te_p5={p5:.2f}%  best={best_acc:.2f}% @ ep{best_ep}  dt={time.time() - t0:.1f}s")
        if best_acc >= args.stop_at:
            write(f"TARGET_REACHED best={best_acc:.2f}% @ ep{best_ep}")
            if ep - best_ep >= args.stop_patience:
                write(f"EARLY_STOP best={best_acc:.2f}% @ ep{best_ep}")
                break

    if best_payload is not None:
        logits_np, labels_np, sigs_np = best_payload
        np.savez(
            os.path.join(args.workdir, "test_logits.npz"),
            logits=logits_np, labels=labels_np, sigs=sigs_np,
            epoch=np.array([best_ep], dtype=np.int64),
        )
    write(f"FINAL best={best_acc:.2f}% @ ep{best_ep}")
    log.close()


if __name__ == "__main__":
    main()
