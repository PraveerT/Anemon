import argparse
import copy
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from train_depth_small import (
    accuracy_topk,
    quat_distance_loss,
    quat_mul,
    target_corr_quats,
    axis_angle_to_quat,
)


class CorrDataset(Dataset):
    def __init__(self, corr, labels, sigs, train=False, jitter=0.0, point_drop=0.0):
        self.corr = torch.from_numpy(np.asarray(corr, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.sigs = np.asarray(sigs, dtype=object)
        self.train = bool(train)
        self.jitter = float(jitter)
        self.point_drop = float(point_drop)

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, idx):
        x = self.corr[idx].clone()
        if self.train and self.jitter > 0:
            x = x + torch.randn_like(x) * self.jitter
        if self.train and self.point_drop > 0 and random.random() < self.point_drop:
            keep = torch.rand(x.shape[1]) > self.point_drop
            if keep.sum() >= 8:
                mean = x[:, keep].mean(dim=1, keepdim=True)
                x[:, ~keep] = mean
        return x, self.labels[idx], self.sigs[idx]


def normalize_corr(x):
    x = x.float()
    centered = x - x.mean(dim=2, keepdim=True)
    scale = centered.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-4)
    return centered / scale


def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def random_unit_quat(batch, device, dtype):
    u1 = torch.rand(batch, device=device, dtype=dtype)
    u2 = torch.rand(batch, device=device, dtype=dtype)
    u3 = torch.rand(batch, device=device, dtype=dtype)
    two_pi = 2.0 * math.pi
    q = torch.stack([
        torch.sqrt(1.0 - u1) * torch.sin(two_pi * u2),
        torch.sqrt(1.0 - u1) * torch.cos(two_pi * u2),
        torch.sqrt(u1) * torch.sin(two_pi * u3),
        torch.sqrt(u1) * torch.cos(two_pi * u3),
    ], dim=-1)
    # Return scalar-first convention used by the rest of this file.
    return torch.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], dim=-1)


def sample_rotation_quat(batch, device, dtype, mode="uniform", max_angle_deg=20.0):
    mode = str(mode).lower()
    if mode == "uniform":
        return random_unit_quat(batch, device, dtype)
    angle = (torch.rand(batch, device=device, dtype=dtype) * 2.0 - 1.0)
    angle = angle * math.radians(float(max_angle_deg))
    if mode == "z":
        axis = torch.zeros(batch, 3, device=device, dtype=dtype)
        axis[:, 2] = 1.0
    elif mode == "y":
        axis = torch.zeros(batch, 3, device=device, dtype=dtype)
        axis[:, 1] = 1.0
    elif mode == "x":
        axis = torch.zeros(batch, 3, device=device, dtype=dtype)
        axis[:, 0] = 1.0
    elif mode == "small-so3":
        axis = torch.randn(batch, 3, device=device, dtype=dtype)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        raise ValueError(f"unknown rot mode: {mode}")
    return axis_angle_to_quat(axis * angle.unsqueeze(-1))


def quat_to_rotmat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack([
        ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz,
    ], dim=-1).view(q.size(0), 3, 3)


def rotate_corr(corr, q):
    rot = quat_to_rotmat(q).to(dtype=corr.dtype)
    return torch.einsum("btpc,bcd->btpd", corr, rot)


def consistency_kl(student_logits, teacher_logits, conf_thresh=0.0, conf_power=0.0):
    teacher = F.softmax(teacher_logits.detach().float(), dim=1)
    log_student = F.log_softmax(student_logits.float(), dim=1)
    per_sample = F.kl_div(log_student, teacher, reduction="none").sum(dim=1)
    weight = None
    conf = teacher.max(dim=1).values
    if conf_thresh > 0.0:
        weight = (conf >= float(conf_thresh)).float()
    if conf_power > 0.0:
        conf_weight = conf.clamp_min(1e-6).pow(float(conf_power))
        weight = conf_weight if weight is None else weight * conf_weight
    if weight is not None:
        return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)
    return per_sample.mean()



class CorrQCCNet(nn.Module):
    def __init__(
        self,
        frames=8,
        num_classes=25,
        point_hidden=128,
        temporal_hidden=192,
        layers=2,
        dropout=0.25,
        quat_inject=False,
    ):
        super().__init__()
        self.frames = int(frames)
        self.quat_inject = bool(quat_inject)
        self.point_mlp = nn.Sequential(
            nn.Linear(9, point_hidden),
            nn.GELU(),
            nn.LayerNorm(point_hidden),
            nn.Dropout(dropout),
            nn.Linear(point_hidden, point_hidden),
            nn.GELU(),
        )
        self.frame_proj = nn.Sequential(
            nn.Linear(point_hidden * 2, temporal_hidden),
            nn.GELU(),
            nn.LayerNorm(temporal_hidden),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(
            temporal_hidden,
            temporal_hidden // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(temporal_hidden, temporal_hidden // 2),
            nn.Tanh(),
            nn.Linear(temporal_hidden // 2, 1),
        )
        if self.quat_inject:
            qdim = (self.frames - 1) * 4 + (self.frames - 2) * 4 + (self.frames - 2) * 4 + (self.frames - 2)
            self.quat_proj = nn.Sequential(
                nn.Linear(qdim, temporal_hidden),
                nn.GELU(),
                nn.LayerNorm(temporal_hidden),
                nn.Dropout(dropout),
            )
        else:
            self.quat_proj = None
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_hidden),
            nn.Linear(temporal_hidden, temporal_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden, num_classes),
        )
        pair_dim = temporal_hidden * 3
        self.step_head = nn.Linear(pair_dim, 3)
        self.skip_head = nn.Linear(pair_dim, 3)

    def forward(self, x):
        x = normalize_corr(x)
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]
        disp = x - x[:, :1]
        point_in = torch.cat([x, vel, disp], dim=-1)
        p = self.point_mlp(point_in)
        frame = torch.cat([p.mean(dim=2), p.amax(dim=2)], dim=-1)
        frame = self.frame_proj(frame)
        h, _ = self.temporal(frame)
        attn = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        pooled = (h * attn.unsqueeze(-1)).sum(dim=1)

        if self.quat_proj is not None:
            with torch.no_grad():
                tgt_step, tgt_skip = target_corr_quats(x)
                cyc = quat_mul(tgt_step[:, 1:], tgt_step[:, :-1])
                cyc_err = 1.0 - (cyc * tgt_skip).sum(dim=-1).abs().clamp(0.0, 1.0)
                qvec = torch.cat([
                    tgt_step.reshape(x.size(0), -1),
                    tgt_skip.reshape(x.size(0), -1),
                    cyc.reshape(x.size(0), -1),
                    cyc_err.reshape(x.size(0), -1),
                ], dim=1)
            pooled = pooled + self.quat_proj(qvec)
        logits = self.classifier(pooled)

        step_pair = torch.cat([h[:, :-1], h[:, 1:], h[:, 1:] - h[:, :-1]], dim=-1)
        skip_pair = torch.cat([h[:, :-2], h[:, 2:], h[:, 2:] - h[:, :-2]], dim=-1)
        step = axis_angle_to_quat(self.step_head(step_pair).view(x.size(0), self.frames - 1, 3))
        skip = axis_angle_to_quat(self.skip_head(skip_pair).view(x.size(0), self.frames - 2, 3))
        return logits, step, skip


def qcc_loss(pred_step, pred_skip, corr):
    corr = normalize_corr(corr)
    tgt_step, tgt_skip = target_corr_quats(corr)
    quat_loss = quat_distance_loss(pred_step, tgt_step) + quat_distance_loss(pred_skip, tgt_skip)
    cycle = quat_mul(pred_step[:, 1:], pred_step[:, :-1])
    cycle_loss = quat_distance_loss(cycle, pred_skip)
    return quat_loss, cycle_loss


def update_ema(model, ema, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(msd[k], alpha=1.0 - decay)
            else:
                v.copy_(msd[k])


def lr_for_epoch(epoch, epochs, lr, min_lr, warmup):
    if epoch <= warmup:
        return lr * epoch / max(1, warmup)
    p = (epoch - warmup) / max(1, epochs - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    logits_all, labels_all, sigs_all = [], [], []
    for corr, labels, sigs in loader:
        corr = corr.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=amp):
            logits = model(corr)[0]
        logits_all.append(logits.float().cpu())
        labels_all.append(labels.cpu())
        sigs_all.extend(sigs)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    p1, p5 = accuracy_topk(logits, labels)
    return p1, p5, logits.numpy(), labels.numpy(), np.asarray(sigs_all, dtype=object)


def log_prob_fusion(base_logits, branch_logits, labels):
    device = torch.device("cpu")
    base = torch.from_numpy(base_logits).float().to(device)
    branch = torch.from_numpy(branch_logits).float().to(device)
    y = torch.from_numpy(labels).long().to(device)
    best = None
    temps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    weights = [i / 20.0 for i in range(0, 61)]
    for tb in temps:
        bp = F.log_softmax(base / tb, dim=1)
        for tc in temps:
            cp = F.log_softmax(branch / tc, dim=1)
            for w in weights:
                fused = bp + w * cp
                p1, p5 = accuracy_topk(fused, y)
                if best is None or p1 > best["top1"] or (p1 == best["top1"] and p5 > best["top5"]):
                    best = {
                        "top1": p1,
                        "top5": p5,
                        "tb": tb,
                        "tc": tc,
                        "w": w,
                        "logits": fused.numpy(),
                    }
    return best


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/notebooks/Anemon/dataset/Nvidia/Processed/depth_small_cache")
    ap.add_argument("--workdir", default="/notebooks/Anemon/experiments/work_dir/depth_corr_qcc_fusion")
    ap.add_argument("--base-logits", default="/notebooks/Anemon/experiments/work_dir/depth_small_r2_fg83_restored_20260528_033028/best_logits.npz")
    ap.add_argument("--active-depth-workdir", default="/notebooks/Anemon/experiments/work_dir/depth_small")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--points", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--warmup-epochs", type=int, default=8)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--label-smoothing", type=float, default=0.08)
    ap.add_argument("--qcc-weight", type=float, default=0.0)
    ap.add_argument("--cycle-weight", type=float, default=0.0)
    ap.add_argument("--rot-cycle-weight", type=float, default=0.0)
    ap.add_argument("--rot-aug-ce-weight", type=float, default=0.0)
    ap.add_argument("--rot-cycle-prob", type=float, default=1.0)
    ap.add_argument("--rot-mode", choices=["uniform", "small-so3", "x", "y", "z"], default="uniform")
    ap.add_argument("--rot-max-angle-deg", type=float, default=20.0)
    ap.add_argument("--rot-conf-thresh", type=float, default=0.0)
    ap.add_argument("--rot-conf-power", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--point-hidden", type=int, default=128)
    ap.add_argument("--temporal-hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--jitter", type=float, default=0.006)
    ap.add_argument("--point-drop", type=float, default=0.08)
    ap.add_argument("--quat-inject", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--publish-active", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache_dir)
    train_corr = np.load(cache / f"train_corr_f{args.frames}_p{args.points}.npy")
    valid_corr = np.load(cache / f"valid_corr_f{args.frames}_p{args.points}.npy")
    train_labels = np.load(cache / "train_fg_s128_labels.npy")
    valid_labels = np.load(cache / "valid_fg_s128_labels.npy")
    train_sigs = np.load(cache / "train_fg_s128_sigs.npy", allow_pickle=True)
    valid_sigs = np.load(cache / "valid_fg_s128_sigs.npy", allow_pickle=True)

    train_ds = CorrDataset(train_corr, train_labels, train_sigs, train=True, jitter=args.jitter, point_drop=args.point_drop)
    valid_ds = CorrDataset(valid_corr, valid_labels, valid_sigs, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
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

    base_npz = np.load(args.base_logits, allow_pickle=True)
    base_logits = base_npz["logits"]
    base_labels = base_npz["labels"]
    base_sigs = base_npz["sigs"]
    if not np.array_equal(base_labels, valid_labels) or not np.array_equal(base_sigs, valid_sigs):
        raise RuntimeError("base logits are not aligned with valid correspondence cache")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorrQCCNet(
        frames=args.frames,
        point_hidden=args.point_hidden,
        temporal_hidden=args.temporal_hidden,
        layers=args.layers,
        dropout=args.dropout,
        quat_inject=args.quat_inject,
    ).to(device)
    ema = copy.deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=not args.no_amp)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp = not args.no_amp

    log = open(workdir / "log.txt", "a", encoding="utf-8")

    def write(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    write(f"CorrQCCNet params={params_m:.3f}M train={len(train_ds)} valid={len(valid_ds)} frames={args.frames} points={args.points}")
    write(f"lr={args.lr:g} min_lr={args.min_lr:g} warmup={args.warmup_epochs} wd={args.wd:g} ema={args.ema_decay:g} qcc={args.qcc_weight:g} cycle={args.cycle_weight:g} rot_cycle={args.rot_cycle_weight:g} rot_aug_ce={args.rot_aug_ce_weight:g} rot_prob={args.rot_cycle_prob:g}")
    write(f"rot_mode={args.rot_mode} rot_max_angle_deg={args.rot_max_angle_deg:g} rot_conf_thresh={args.rot_conf_thresh:g} rot_conf_power={args.rot_conf_power:g}")
    write(f"dropout={args.dropout:g} point_hidden={args.point_hidden} temporal_hidden={args.temporal_hidden} layers={args.layers} jitter={args.jitter:g} point_drop={args.point_drop:g} quat_inject={args.quat_inject}")

    best_branch = 0.0
    best_fused = log_prob_fusion(base_logits, np.zeros_like(base_logits), valid_labels)
    best_fused["logits"] = base_logits
    best_ep = 0
    best_payload = None
    total_batches = len(train_loader)
    for ep in range(1, args.epochs + 1):
        lr = lr_for_epoch(ep, args.epochs, args.lr, args.min_lr, args.warmup_epochs)
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        t0 = time.time()
        total_loss = total_ce = total_q = total_c = total_rot = 0.0
        correct = seen = 0
        for bi, (corr, labels, _sigs) in enumerate(train_loader, 1):
            corr = corr.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logits, step, skip = model(corr)
                ce = loss_fn(logits, labels)
                qloss, closs = qcc_loss(step, skip, corr)
                loss = ce + args.qcc_weight * qloss + args.cycle_weight * closs
                rot_loss = torch.zeros((), device=device, dtype=loss.dtype)
                if (
                    (args.rot_cycle_weight > 0.0 or args.rot_aug_ce_weight > 0.0)
                    and random.random() < args.rot_cycle_prob
                ):
                    qrot = sample_rotation_quat(
                        corr.size(0),
                        corr.device,
                        corr.dtype,
                        mode=args.rot_mode,
                        max_angle_deg=args.rot_max_angle_deg,
                    )
                    corr_rot = rotate_corr(corr, qrot)
                    corr_cycle = rotate_corr(corr_rot, quat_conj(qrot))
                    rot_logits = model(corr_rot)[0]
                    cycle_logits = model(corr_cycle)[0]
                    if args.rot_aug_ce_weight > 0.0:
                        rot_loss = rot_loss + args.rot_aug_ce_weight * loss_fn(rot_logits, labels)
                    if args.rot_cycle_weight > 0.0:
                        rot_loss = rot_loss + args.rot_cycle_weight * (
                            consistency_kl(rot_logits, logits, args.rot_conf_thresh, args.rot_conf_power)
                            + consistency_kl(cycle_logits, logits, args.rot_conf_thresh, args.rot_conf_power)
                            + consistency_kl(cycle_logits, rot_logits, args.rot_conf_thresh, args.rot_conf_power)
                        ) / 3.0
                    loss = loss + rot_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            update_ema(model, ema, args.ema_decay)
            bs = labels.numel()
            total_loss += loss.detach().float().item() * bs
            total_ce += ce.detach().float().item() * bs
            total_q += qloss.detach().float().item() * bs
            total_c += closs.detach().float().item() * bs
            total_rot += rot_loss.detach().float().item() * bs
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += bs
            if bi == total_batches or bi % 10 == 0:
                write(f"\tBatch({bi}/{total_batches}) loss={loss.detach().float().item():.5f} lr={lr:.6f}")

        branch_p1, branch_p5, branch_logits, labels_np, sigs_np = evaluate(ema, valid_loader, device, amp=amp)
        fused = log_prob_fusion(base_logits, branch_logits, labels_np)
        train_acc = correct / max(1, seen) * 100.0
        write(
            f"ep{ep:3d} tr_loss={total_loss/max(1,seen):.4f} ce={total_ce/max(1,seen):.4f} "
            f"q={total_q/max(1,seen):.4f} cyc={total_c/max(1,seen):.4f} rot={total_rot/max(1,seen):.4f} tr_acc={train_acc:.2f}% "
            f"branch={branch_p1:.2f}%/{branch_p5:.2f}% fused={fused['top1']:.2f}%/{fused['top5']:.2f}% "
            f"w={fused['w']:.2f} tb={fused['tb']:.2f} tc={fused['tc']:.2f} "
            f"best_fused={best_fused['top1']:.2f}% @ ep{best_ep} dt={time.time()-t0:.1f}s"
        )

        np.savez(
            workdir / "branch_logits.npz",
            logits=branch_logits,
            labels=labels_np,
            sigs=sigs_np,
            epoch=np.array([ep], dtype=np.int64),
        )
        np.savez(
            workdir / "fused_logits.npz",
            logits=fused["logits"],
            labels=labels_np,
            sigs=sigs_np,
            epoch=np.array([ep], dtype=np.int64),
            branch_top1=np.array([branch_p1], dtype=np.float32),
            fused_top1=np.array([fused["top1"]], dtype=np.float32),
            weight=np.array([fused["w"]], dtype=np.float32),
            temp_base=np.array([fused["tb"]], dtype=np.float32),
            temp_branch=np.array([fused["tc"]], dtype=np.float32),
        )
        if branch_p1 > best_branch or fused["top1"] > best_fused["top1"]:
            if branch_p1 > best_branch:
                best_branch = branch_p1
            if fused["top1"] >= best_fused["top1"]:
                best_fused = fused
                best_ep = ep
                best_payload = (branch_logits, labels_np, sigs_np, fused)
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": ema.state_dict(),
                        "branch_top1": branch_p1,
                        "branch_top5": branch_p5,
                        "fused_top1": fused["top1"],
                        "fused_top5": fused["top5"],
                        "fusion_weight": fused["w"],
                        "temp_base": fused["tb"],
                        "temp_branch": fused["tc"],
                        "params_m": params_m,
                    },
                    workdir / "best_model.pt",
                )
                np.savez(
                    workdir / "best_branch_logits.npz",
                    logits=branch_logits,
                    labels=labels_np,
                    sigs=sigs_np,
                    epoch=np.array([ep], dtype=np.int64),
                )
                np.savez(
                    workdir / "best_fused_logits.npz",
                    logits=fused["logits"],
                    labels=labels_np,
                    sigs=sigs_np,
                    epoch=np.array([ep], dtype=np.int64),
                    branch_top1=np.array([branch_p1], dtype=np.float32),
                    fused_top1=np.array([fused["top1"]], dtype=np.float32),
                    weight=np.array([fused["w"]], dtype=np.float32),
                    temp_base=np.array([fused["tb"]], dtype=np.float32),
                    temp_branch=np.array([fused["tc"]], dtype=np.float32),
                )
                if args.publish_active:
                    active = Path(args.active_depth_workdir)
                    active.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        active / "test_logits.npz",
                        logits=fused["logits"],
                        labels=labels_np,
                        sigs=sigs_np,
                        epoch=np.array([ep], dtype=np.int64),
                    )
                    np.savez(
                        active / "best_logits.npz",
                        logits=fused["logits"],
                        labels=labels_np,
                        sigs=sigs_np,
                        epoch=np.array([ep], dtype=np.int64),
                    )
        if best_payload is not None and ep - best_ep >= 50:
            write(f"EARLY_STOP no fused improvement for 50 epochs; best={best_fused['top1']:.2f}% @ ep{best_ep}")
            break

    if best_payload is not None:
        branch_logits, labels_np, sigs_np, fused = best_payload
        np.savez(workdir / "branch_logits.npz", logits=branch_logits, labels=labels_np, sigs=sigs_np, epoch=np.array([best_ep], dtype=np.int64))
        np.savez(workdir / "fused_logits.npz", logits=fused["logits"], labels=labels_np, sigs=sigs_np, epoch=np.array([best_ep], dtype=np.int64))
    write(f"FINAL branch_best={best_branch:.2f}% fused_best={best_fused['top1']:.2f}% @ ep{best_ep}")
    log.close()


if __name__ == "__main__":
    main()
