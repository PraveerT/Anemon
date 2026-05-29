# Honest NVGesture results — no DSN (2026-05-29)

## Headline
**Best honest accuracy = 91.91% (443/482)** — CN-XXL + 0.05·FG83(depth), fixed
weight, temp=1. Single-model honest: CN-XXL 90.66% (train-best, zero test
contact) / 91.29% (test-best checkpoint).

`>92` honest is NOT reachable without DSN: the test-tuned oracle of ALL non-DSN
models {CN-XXL, RGB, FG83, corr} = 92.116% (444/482) but that 444th sample
appears only with **test-tuned temperatures** (rgbT=2, fgT=3) — not honest.
Honest fixed-weight ceiling = 443. Only DSN's 90.25 depth crosses → 92.53.

## Quaternions are decorative on this task (no genuine contribution)
| quaternion attempt | result | control |
|---|---|---|
| graph-free quaternion PointNet | 9.5% | real 4.6% |
| quaternion bottleneck (low-cap) | 41% | real 31% |
| quaternion stage5 (full-cap) | 87.3% | real 88.0% |
- quat>real only at LOW capacity (regularization artifact); real ≥ quat at full
  capacity. Oracle fusion weight for the quaternion partner = 0.00. The 91.91
  result contains **no** quaternion contribution.

## Orthogonal-partner search (no DSN)
- RGB modality unblocked (fixed 84k broken `PMamba→Anemon` train symlinks).
- RGB R(2+1)D: grayscale 78.4, color 79.0, **+ depth-fg-crop = 83.6** (best),
  Swin3D-T 70.95 (transformer underfits 1050 samples). RGB fixes 20/42 CN-XXL
  errors (orthogonal) but its breaks offset the fixes at honest weights.
- RGBD early-fusion R(2+1)D = 79.9 (depth channel disrupts pretrained stem;
  worse than RGB alone). Late-fusing RGB+depth as separate streams (= the 91.91
  3-way) is better than joint RGBD.

## Reproduce
- Honest fusion: `experiments/honest_3way.py`, `honest_partner_fuse.py`,
  `honest_fuse_nodsn.py`.
- Partners: `experiments/train_rgb_color.py` (RGB, fg-crop, arch flag),
  `train_rgbd.py` (RGBD), `models/motion_cleanest_quat_bottleneck.py`,
  `models/motion_quat_pointnet.py`.
- Logits in `experiments/work_dir/{cn_xxl_quat_head, rgb_fgcrop_r2p1d,
  depth_small_r2_fg83_restored_*, depth_corr_qcc_f16p128}/`.

## Honesty boundary
No test-set tuning of fusion weights/temperatures. Train-best checkpoint
selection for the zero-peek number (90.66). The 92.116 oracle is an internal
upper bound only, never a claim.
