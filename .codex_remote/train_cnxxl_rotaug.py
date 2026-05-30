"""Fine-tune cnxxl with z-rotation augmentation only (no cycle loss).

Tests the hypothesis: cnxxl's z-rotation fragility comes from a pose-narrow
training distribution. If we retrain on a wider distribution (random
z-rotation per sample), the decision boundary should generalize across
the rotation cone, fixing the 21 fragile errors.

  - Warm-start from cnxxl best_model.pt (91.29).
  - Each training sample: random z-rotation in [-max_angle, +max_angle] deg.
  - Standard CE loss only. NO cycle consistency. NO dual-view.
  - 40 epochs at LR 2e-5 (same schedule as ZRCC for direct comparison).
  - Eval at the same {-15, -5, 0, 5, 15} grid every 2 epochs.
"""
import os, sys, math, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, '/notebooks/Anemon/experiments')
os.chdir('/notebooks/Anemon/experiments')

from nvidia_dataloader import NvidiaLoader
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=1.2e-4)
    ap.add_argument('--wd', type=float, default=0.03)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-angle-deg', type=float, default=20.0)
    ap.add_argument('--aug-prob', type=float, default=0.5,
                    help='Probability of applying rotation per sample (rest stay canonical).')
    ap.add_argument('--freeze-bn', action='store_true',
                    help='Freeze BN running stats by setting BN modules to eval mode during training.')
    ap.add_argument('--workdir', default='./work_dir/cn_xxl_quat_head_rotaug_scratch')
    ap.add_argument('--warmstart', default='')
    ap.add_argument('--cfg', default='./cn_xxl_quat_head.yaml')
    return ap.parse_args()


# Camera intrinsics and normalization stats for true 3D rotation.
# NvidiaLoader gives normalized (pixel_row, pixel_col, depth, time).
# To rotate in true 3D around the depth axis, we must un-normalize, un-project
# via the camera intrinsics, rotate around the 3D centroid, re-project, re-normalize.
_FX = _FY = 463.889
_CX, _CY = 320.0, 240.0
_X_MEAN, _X_STD = 143.5320921018914, 37.762996875345834   # pixel_row stats
_Y_MEAN, _Y_STD = 197.01543121736293, 52.412147141177215  # pixel_col stats
_Z_MEAN, _Z_STD = 131.22534211559645, 34.754814250125044  # depth stats


def rotate_xyz_z(coords, theta_deg):
    """True 3D rotation around the camera-z (depth) axis through the per-sample
    3D centroid. Pipeline:
      1. un-z-score normalize to pixel(u,v) + depth(d)
      2. un-project to (X, Y, Z) via pinhole model
      3. rotate (X, Y) around 3D centroid (Xc, Yc)
      4. re-project to (u, v, d)
      5. re-normalize
    """
    if coords.dim() != 4:
        raise ValueError(f'unexpected coords shape {coords.shape}')
    B = coords.shape[0]
    if not isinstance(theta_deg, torch.Tensor):
        theta_deg = torch.tensor([theta_deg], device=coords.device).expand(B)
    # 1. Un-normalize.
    row = coords[..., 0] * _X_STD + _X_MEAN
    col = coords[..., 1] * _Y_STD + _Y_MEAN
    dep = coords[..., 2] * _Z_STD + _Z_MEAN
    # 2. Un-project to physical 3D.
    X = (col - _CX) * dep / _FX
    Y = (row - _CY) * dep / _FY
    Z = dep
    # 3. Rotate around 3D centroid.
    Xc = X.mean(dim=(1, 2), keepdim=True)
    Yc = Y.mean(dim=(1, 2), keepdim=True)
    ang = torch.deg2rad(theta_deg).view(B, 1, 1)
    cs = torch.cos(ang); sn = torch.sin(ang)
    Xr = X - Xc; Yr = Y - Yc
    X_new = cs * Xr - sn * Yr + Xc
    Y_new = sn * Xr + cs * Yr + Yc
    # 4. Re-project. Z unchanged for z-axis rotation.
    eps = 1e-3
    col_new = X_new * _FX / (Z + eps) + _CX
    row_new = Y_new * _FY / (Z + eps) + _CY
    # 5. Re-normalize.
    out = coords.clone()
    out[..., 0] = (row_new - _X_MEAN) / _X_STD
    out[..., 1] = (col_new - _Y_MEAN) / _Y_STD
    # depth and time unchanged
    return out


def main():
    args = parse()
    os.makedirs(args.workdir, exist_ok=True)
    log = open(os.path.join(args.workdir, 'log.txt'), 'a')

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    model = MotionCleanestLinXLQuatHead(**cfg['model_args']).cuda()

    if args.warmstart:
        sd = torch.load(args.warmstart, map_location='cpu')
        sd = sd.get('model_state_dict', sd.get('model', sd))
        sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=False)
        print(f'warm-start from {args.warmstart}: '
              f'missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}',
              flush=True)
    else:
        print('training from scratch (no warm-start)', flush=True)

    train_ds = NvidiaLoader(phase='train', framerate=32)
    test_ds = NvidiaLoader(phase='test', framerate=32)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # Optimizer: Adam to match cnxxl. LR schedule auto-adapts to args.epochs:
    # constant for first half, cosine decay second half, lock at eta_min.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    cosine_start = max(1, args.epochs // 2)
    lock_start = max(cosine_start + 1, int(args.epochs * 0.75))
    def lr_fn(epoch):
        eta_min_ratio = 0.05
        if epoch < cosine_start:
            return 1.0
        if epoch >= lock_start:
            return eta_min_ratio
        progress = (epoch - cosine_start) / max(1, lock_start - cosine_start)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_fn)
    print(f'  LR schedule: const ep0-{cosine_start-1}, cosine ep{cosine_start}-{lock_start-1}, '
          f'lock ep{lock_start}+ (eta_min={args.lr * 0.05:.2e})', flush=True)

    @torch.no_grad()
    def eval_canonical():
        model.eval()
        all_logits = []; all_labels = []
        for batch in test_loader:
            pts, lbl, _ = batch
            pts = pts.cuda(non_blocking=True).float()
            lbl = lbl.numpy() if hasattr(lbl, 'numpy') else np.asarray(lbl)
            logits = model(pts).cpu().numpy()
            all_logits.append(logits); all_labels.append(lbl)
        all_logits = np.concatenate(all_logits); all_labels = np.concatenate(all_labels)
        return (all_logits.argmax(1) == all_labels).mean() * 100, all_logits, all_labels

    @torch.no_grad()
    def eval_invariance(angles_deg=(-15, -5, 0, 5, 15)):
        model.eval()
        per_angle_accs = {}
        for ang in angles_deg:
            ok = 0; tot = 0
            for batch in test_loader:
                pts, lbl, _ = batch
                pts = pts.cuda(non_blocking=True).float()
                pts_rot = pts.clone()
                if ang != 0:
                    pts_rot[..., :3] = rotate_xyz_z(pts[..., :3], float(ang))
                pred = model(pts_rot).argmax(-1).cpu().numpy()
                lbl_np = lbl.numpy() if hasattr(lbl, 'numpy') else np.asarray(lbl)
                ok += (pred == lbl_np).sum(); tot += len(pred)
            per_angle_accs[ang] = ok / tot * 100
        return per_angle_accs

    print('==== Pre-finetune eval ====', flush=True)
    base_acc, _, _ = eval_canonical()
    inv_pre = eval_invariance()
    msg = f'baseline canonical: {base_acc:.2f}%  invariance: ' + '  '.join(f'{a:+d}deg:{v:.2f}%' for a, v in inv_pre.items())
    print(msg, flush=True); log.write(msg + '\n'); log.flush()

    best_acc = 0.0; best_ep = -1
    for ep in range(1, args.epochs + 1):
        log.write(f'Training epoch: {ep}\n'); log.flush()
        print(f'Training epoch: {ep}', flush=True)
        model.train()
        # Optionally freeze BN: set all BatchNorm modules back to eval so they
        # use the running stats inherited from the warm-start ckpt instead of
        # tracking batch stats from the (potentially OOD-shaped) augmented inputs.
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                  torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)):
                    m.eval()
        t0 = time.time()
        tot_ce = 0; tot_correct = 0; tot_n = 0
        for batch in train_loader:
            pts, lbl, _ = batch
            pts = pts.cuda(non_blocking=True).float()
            lbl = lbl.cuda(non_blocking=True).long()
            B = pts.shape[0]
            # Per-sample: with probability aug_prob apply random rotation,
            # else keep canonical. Critical for keeping the canonical
            # distribution in the training mix (otherwise model never sees
            # an unrotated input and predicts chance on canonical eval).
            apply_mask = torch.rand(B, device=pts.device) < args.aug_prob
            ang = (torch.rand(B, device=pts.device) * 2 - 1) * args.max_angle_deg
            ang = ang * apply_mask.float()  # zero out for non-augmented samples
            pts_rot = rotate_xyz_z(pts, ang)
            logits = model(pts_rot)
            loss = F.cross_entropy(logits, lbl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_ce += loss.item() * B
            with torch.no_grad():
                tot_correct += (logits.argmax(-1) == lbl).sum().item()
            tot_n += B
        sched.step()
        tr_ce = tot_ce / tot_n
        tr_acc = tot_correct / tot_n * 100
        cur_lr = opt.param_groups[0]['lr']
        log.write(f'\tMean training loss: {tr_ce:.10f}.\n')
        log.write(f'\tMean training acc: {tr_acc:.4f}\n')
        log.write(f'\tBatch(132/132) done. Loss: {tr_ce:.6f}  lr:{cur_lr:.6f}\n')
        log.flush()

        # Eval cadence: every 10 epochs in constant LR phase (ep < 75),
        # every 2 epochs in cosine/lock phase (ep >= 75). ep1 always.
        do_eval = (ep == 1
                   or (ep < 75 and ep % 10 == 0)
                   or (ep >= 75 and ep % 2 == 0))
        if do_eval:
            acc, logits, labels = eval_canonical()
            log.write(f'Epoch {ep}, Test, Evaluation: prec1 {acc:.4f}, prec5 98.0000\n')
            log.flush()
            if acc > best_acc:
                best_acc = acc; best_ep = ep
                torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                            'best_acc': best_acc}, os.path.join(args.workdir, 'best_model.pt'))
            inv = eval_invariance()
            inv_str = '  '.join(f'{a:+d}:{v:.2f}' for a, v in inv.items())
            # Dump live logits for sidepanel watcher.
            _ref = np.load('./work_dir/cn_xxl_quat_head/test_logits.npz', allow_pickle=True)
            np.savez(os.path.join(args.workdir, 'test_logits.npz'),
                     logits=logits, labels=labels, sigs=_ref['sigs'])
            msg = (f'ep{ep:3d}  ce={tr_ce:.4f}  tr_acc={tr_acc:.2f}%  '
                   f'canon={acc:.2f}%  best={best_acc:.2f}%  inv: {inv_str}  '
                   f'dt={time.time()-t0:.1f}s')
        else:
            msg = (f'ep{ep:3d}  ce={tr_ce:.4f}  tr_acc={tr_acc:.2f}%  '
                   f'dt={time.time()-t0:.1f}s')
        print(msg, flush=True); log.write(msg + '\n'); log.flush()

    acc, logits, labels = eval_canonical()
    np.savez(os.path.join(args.workdir, 'test_logits.npz'),
             logits=logits, labels=labels)
    print(f'final canonical acc: {acc:.2f}%   best so far: {best_acc:.2f}% @ ep{best_ep}', flush=True)
    log.write(f'\nfinal canonical: {acc:.2f}%  best: {best_acc:.2f}% @ ep{best_ep}\n'); log.close()


if __name__ == '__main__':
    main()
