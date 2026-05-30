"""Quaternion rotation-cycle fine-tune for cnxxlquat.

This is the CNXXL-side counterpart to the depth correspondence q-rotation-cycle
run. It keeps the canonical classifier as the teacher, applies a true 3D
quaternion rotation to the depth point tensor, rotates the transformed tensor
back by the inverse quaternion, and penalizes prediction drift.

The point transform is not a raw normalized-coordinate rotation. It
un-normalizes row/col/depth, unprojects to camera 3D, rotates around the
per-sample 3D centroid, reprojects to row/col/depth, and re-normalizes.
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, "/notebooks/Anemon/experiments")
os.chdir("/notebooks/Anemon/experiments")

from nvidia_dataloader import NvidiaLoader
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead


_FX = _FY = 463.889
_CX, _CY = 320.0, 240.0
_X_MEAN, _X_STD = 143.5320921018914, 37.762996875345834
_Y_MEAN, _Y_STD = 197.01543121736293, 52.412147141177215
_Z_MEAN, _Z_STD = 131.22534211559645, 34.754814250125044


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1.0e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-interval", type=int, default=2)
    ap.add_argument("--lambda-cycle", type=float, default=0.05)
    ap.add_argument("--rot-aug-ce-weight", type=float, default=0.0)
    ap.add_argument("--cycle-prob", type=float, default=1.0)
    ap.add_argument("--rot-mode", choices=["z", "small-so3", "x", "y"], default="z")
    ap.add_argument("--max-angle-deg", type=float, default=10.0)
    ap.add_argument("--conf-thresh", type=float, default=0.0)
    ap.add_argument("--conf-power", type=float, default=0.0)
    ap.add_argument("--freeze-bn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-scope", choices=["all", "head"], default="all")
    ap.add_argument("--diagnostic-only", action="store_true")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--workdir", default="./work_dir/cn_xxl_quat_head_qrotcycle_z10_w005")
    ap.add_argument("--warmstart", default="./work_dir/cn_xxl_quat_head/best_model.pt")
    ap.add_argument("--cfg", default="./cn_xxl_quat_head.yaml")
    return ap.parse_args()


def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def axis_angle_to_quat(v):
    angle = v.norm(dim=-1, keepdim=True)
    axis = v / angle.clamp_min(1e-8)
    half = 0.5 * angle
    return torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)


def sample_quat(batch, device, dtype, mode, max_angle_deg):
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


def fixed_axis_quat(batch, device, dtype, axis_name, angle_deg):
    axis = torch.zeros(batch, 3, device=device, dtype=dtype)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name]
    axis[:, axis_idx] = 1.0
    angle = torch.full((batch,), math.radians(float(angle_deg)), device=device, dtype=dtype)
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


def rotate_points_quat(points, q):
    """Rotate full NvidiaLoader points: (B, T, N, C), C >= 3."""
    coords = points[..., :3]
    row = coords[..., 0] * _X_STD + _X_MEAN
    col = coords[..., 1] * _Y_STD + _Y_MEAN
    dep = coords[..., 2] * _Z_STD + _Z_MEAN

    x3 = (col - _CX) * dep / _FX
    y3 = (row - _CY) * dep / _FY
    z3 = dep
    xyz = torch.stack([x3, y3, z3], dim=-1)
    center = xyz.mean(dim=(1, 2), keepdim=True)
    rot = quat_to_rotmat(q).to(dtype=points.dtype)
    xyz_rot = torch.einsum("btpc,bcd->btpd", xyz - center, rot) + center

    x_new, y_new, z_new = xyz_rot.unbind(dim=-1)
    z_new = z_new.clamp_min(1.0)
    col_new = x_new * _FX / z_new + _CX
    row_new = y_new * _FY / z_new + _CY

    out = points.clone()
    out[..., 0] = (row_new - _X_MEAN) / _X_STD
    out[..., 1] = (col_new - _Y_MEAN) / _Y_STD
    out[..., 2] = (z_new - _Z_MEAN) / _Z_STD
    return out


def consistency_kl(student_logits, teacher_logits, conf_thresh=0.0, conf_power=0.0):
    teacher = F.softmax(teacher_logits.detach().float(), dim=1)
    log_student = F.log_softmax(student_logits.float(), dim=1)
    per_sample = F.kl_div(log_student, teacher, reduction="none").sum(dim=1)
    conf = teacher.max(dim=1).values
    weight = None
    if conf_thresh > 0.0:
        weight = (conf >= float(conf_thresh)).float()
    if conf_power > 0.0:
        conf_weight = conf.clamp_min(1e-6).pow(float(conf_power))
        weight = conf_weight if weight is None else weight * conf_weight
    if weight is not None:
        return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)
    return per_sample.mean()


def topk_acc(logits, labels, k=1):
    pred = logits.topk(k, dim=1).indices
    return pred.eq(labels.view(-1, 1)).any(dim=1).float().mean().item() * 100.0


def set_bn_eval(model):
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                               torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
            module.eval()


def configure_train_scope(model, scope):
    if scope == "all":
        return
    train_names = ("stage6", "quat_head", "quat_head_scale")
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(train_names))


def load_model(args, device):
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    model = MotionCleanestLinXLQuatHead(**cfg["model_args"]).to(device)
    sd = torch.load(args.warmstart, map_location="cpu")
    sd = sd.get("model_state_dict", sd.get("model", sd))
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    configure_train_scope(model, args.train_scope)
    return model, res


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for pts, labels, _sig in loader:
        pts = pts.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        logits = model(pts)
        logits_all.append(logits.float().cpu())
        labels_all.append(labels.cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return topk_acc(logits, labels, 1), topk_acc(logits, labels, 5), logits.numpy(), labels.numpy()


@torch.no_grad()
def evaluate_axis_angle(model, loader, device, axis_name, angle_deg):
    model.eval()
    logits_all, labels_all = [], []
    for pts, labels, _sig in loader:
        pts = pts.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        q = fixed_axis_quat(pts.size(0), pts.device, pts.dtype, axis_name, angle_deg)
        logits = model(rotate_points_quat(pts, q))
        logits_all.append(logits.float().cpu())
        labels_all.append(labels.cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return topk_acc(logits, labels, 1), logits.numpy(), labels.numpy()


def fixed_broken_report(base_logits, base_labels, logits, labels):
    base_pred = base_logits.argmax(1)
    pred = logits.argmax(1)
    base_err = base_pred != base_labels
    err = pred != labels
    fixed = int(np.logical_and(base_err, ~err).sum())
    broken = int(np.logical_and(~base_err, err).sum())
    return fixed, broken


def run_diagnostic(model, loader, device, baseline, write):
    base_logits, base_labels = baseline["logits"], baseline["labels"]
    angles = [-15, -10, -5, 0, 5, 10, 15]
    angle_logits = []
    write("diagnostic true-3D z-axis quaternion TTA:")
    for angle in angles:
        if angle == 0:
            acc, _p5, logits, labels = evaluate(model, loader, device)
        else:
            acc, logits, labels = evaluate_axis_angle(model, loader, device, "z", angle)
        fixed, broken = fixed_broken_report(base_logits, base_labels, logits, labels)
        write(f"  z{angle:+d}: acc={acc:.3f}% fixed={fixed} broken={broken}")
        angle_logits.append(logits)
    avg_logits = np.mean(angle_logits, axis=0)
    avg_acc = (avg_logits.argmax(1) == labels).mean() * 100.0
    fixed, broken = fixed_broken_report(base_logits, base_labels, avg_logits, labels)
    write(f"  avg_logits: acc={avg_acc:.3f}% fixed={fixed} broken={broken}")


def main():
    args = parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_f = open(os.path.join(args.workdir, "log.txt"), "a", encoding="utf-8")

    def write(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    model, load_res = load_model(args, device)
    write(f"warmstart={args.warmstart} missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}")
    write(f"epochs={args.epochs} lr={args.lr:g} wd={args.wd:g} lambda_cycle={args.lambda_cycle:g} rot_aug_ce={args.rot_aug_ce_weight:g} cycle_prob={args.cycle_prob:g}")
    write(f"rot_mode={args.rot_mode} max_angle_deg={args.max_angle_deg:g} conf_thresh={args.conf_thresh:g} conf_power={args.conf_power:g} freeze_bn={args.freeze_bn} train_scope={args.train_scope}")

    train_ds = NvidiaLoader(phase="train", framerate=32)
    test_ds = NvidiaLoader(phase="test", framerate=32)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    baseline = np.load("./work_dir/cn_xxl_quat_head/test_logits.npz", allow_pickle=True)
    base_logits = baseline["logits"]
    base_labels = baseline["labels"]
    sigs = baseline["sigs"] if "sigs" in baseline.files else np.arange(len(base_labels))

    base_acc, base_p5, logits, labels = evaluate(model, test_loader, device)
    write(f"pretrain canonical acc={base_acc:.3f}% top5={base_p5:.3f}%")
    if args.diagnostic_only:
        run_diagnostic(model, test_loader, device, baseline, write)
        log_f.close()
        return

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05
    )

    best_acc = base_acc
    best_ep = 0
    best_logits = logits
    total_batches = len(train_loader)
    for ep in range(1, args.epochs + 1):
        write(f"Training epoch: {ep}")
        model.train()
        if args.freeze_bn:
            set_bn_eval(model)
        t0 = time.time()
        total_loss = total_ce = total_cyc = total_rot_ce = 0.0
        correct = seen = 0
        for pts, labels_t, _sig in train_loader:
            pts = pts.to(device, non_blocking=True).float()
            labels_t = labels_t.to(device, non_blocking=True).long()
            logits_c = model(pts)
            ce = F.cross_entropy(logits_c, labels_t)
            cyc_loss = torch.zeros((), device=device)
            rot_ce = torch.zeros((), device=device)
            if random.random() < args.cycle_prob and (args.lambda_cycle > 0.0 or args.rot_aug_ce_weight > 0.0):
                q = sample_quat(pts.size(0), pts.device, pts.dtype, args.rot_mode, args.max_angle_deg)
                pts_rot = rotate_points_quat(pts, q)
                pts_cycle = rotate_points_quat(pts_rot, quat_conj(q))
                logits_r = model(pts_rot)
                logits_cyc = model(pts_cycle)
                cyc_loss = (
                    consistency_kl(logits_r, logits_c, args.conf_thresh, args.conf_power)
                    + consistency_kl(logits_cyc, logits_c, args.conf_thresh, args.conf_power)
                    + consistency_kl(logits_cyc, logits_r, args.conf_thresh, args.conf_power)
                ) / 3.0
                if args.rot_aug_ce_weight > 0.0:
                    rot_ce = F.cross_entropy(logits_r, labels_t)
            loss = ce + args.lambda_cycle * cyc_loss + args.rot_aug_ce_weight * rot_ce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = labels_t.numel()
            total_loss += loss.detach().item() * bs
            total_ce += ce.detach().item() * bs
            total_cyc += cyc_loss.detach().item() * bs
            total_rot_ce += rot_ce.detach().item() * bs
            correct += (logits_c.argmax(1) == labels_t).sum().item()
            seen += bs
        sched.step()
        lr = opt.param_groups[0]["lr"]
        train_acc = correct / max(1, seen) * 100.0
        write(f"\tMean training loss: {total_loss/max(1, seen):.10f}.")
        write(f"\tMean training acc: {train_acc:.4f}")
        write(f"\tBatch({total_batches}/{total_batches}) done. Loss: {total_loss/max(1, seen):.6f}  lr:{lr:.6f}")

        if ep == 1 or ep % args.eval_interval == 0:
            acc, p5, logits_np, labels_np = evaluate(model, test_loader, device)
            fixed, broken = fixed_broken_report(base_logits, base_labels, logits_np, labels_np)
            write(f"Epoch {ep}, Test, Evaluation: prec1 {acc:.4f}, prec5 {p5:.4f}")
            write(
                f"ep{ep:3d} ce={total_ce/max(1, seen):.4f} cyc={total_cyc/max(1, seen):.5f} "
                f"rot_ce={total_rot_ce/max(1, seen):.4f} canon={acc:.3f}% best={best_acc:.3f}% "
                f"fixed={fixed} broken={broken} dt={time.time()-t0:.1f}s"
            )
            np.savez(os.path.join(args.workdir, "test_logits.npz"),
                     logits=logits_np, labels=labels_np, sigs=sigs,
                     epoch=np.array([ep], dtype=np.int64))
            if acc > best_acc:
                best_acc = acc
                best_ep = ep
                best_logits = logits_np
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args),
                }, os.path.join(args.workdir, "best_model.pt"))
                np.savez(os.path.join(args.workdir, "best_logits.npz"),
                         logits=logits_np, labels=labels_np, sigs=sigs,
                         epoch=np.array([ep], dtype=np.int64))
        if ep - best_ep >= 20 and ep >= 30:
            write(f"EARLY_STOP no canonical improvement for 20 epochs; best={best_acc:.3f}% @ ep{best_ep}")
            break

    fixed, broken = fixed_broken_report(base_logits, base_labels, best_logits, base_labels)
    write(f"FINAL best={best_acc:.3f}% @ ep{best_ep} fixed={fixed} broken={broken}")
    log_f.close()


if __name__ == "__main__":
    main()
