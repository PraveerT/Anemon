# Depth Correspondence Quaternion-Cycle Status

Date: 2026-05-28

## Current State

The quaternion rotation-cycle correspondence run is now the active depth-side
result. It is the first quaternion/cycle mechanism in this line of experiments
that beats the clean correspondence control.

- Active app logits:
  - `/notebooks/Anemon/experiments/work_dir/depth_small/best_logits.npz`
  - `/notebooks/Anemon/experiments/work_dir/depth_small/test_logits.npz`
- Source active run:
  - `/notebooks/Anemon/experiments/work_dir/depth_corr_qrotcycle_f16p128_w002_ce0/best_fused_logits.npz`
  - `/notebooks/Anemon/experiments/work_dir/depth_corr_qrotcycle_f16p128_w002_ce0/best_model.pt`
  - `/notebooks/Anemon/experiments/work_dir/depth_corr_qrotcycle_f16p128_w002_ce0/log.txt`
- Public status endpoint:
  - `https://viz-qcc-production.up.railway.app/api/anemon-status`

Current active result:

- Depth fg83 baseline: `83.61` top-1, `96.47` top-5.
- Quaternion-cycle branch alone: `64.73` top-1, `87.97` top-5.
- fg83 depth + quaternion-cycle fusion: `85.89` top-1, `96.06` top-5.
- Best fused epoch: `117`.
- Branch size: `0.814M` parameters.

The clean `85.48` correspondence branch remains the control baseline. The
current active model adds a small quaternion rotation-cycle consistency loss
over classifier outputs and improves fused accuracy by `+0.41pp`.

## What The Controls Showed

The originally committed QCC/cycle path did not survive ablation.

| Run | QCC loss | Cycle loss | Kabsch-quat injection | Branch solo | Fused |
| --- | ---: | ---: | --- | ---: | ---: |
| `depth_corr_qcc_f16p128` | `0.04` | `0.04` | on | `61.83` | `84.65` |
| `depth_corr_qcc_f16p128_ceonly` | `0` | `0` | on | `61.00` | `85.06` |
| `depth_corr_qcc_f16p128_ceonly_noqinject` | `0` | `0` | off | `67.01` | `85.48` |

Conclusion:

- The QCC and cycle losses are not the source of the gain in this branch.
- Removing QCC/cycle improved fused accuracy from `84.65` to `85.06`.
- Removing Kabsch-quat feature injection improved further to `85.48`.
- The useful signal is currently the correspondence sequence classifier itself:
  normalized xyz, velocity, displacement, point MLP pooling, temporal GRU, and
  attention pooling.

Loss behavior:

- With cycle weight, `cycle_loss` is driven to about `0.009`.
- Without cycle weight, `cycle_loss` stays near its natural value (`0.10` to
  `0.22`, depending on injection).
- The Kabsch geodesic loss remains high in all cases, around `1.08` to `1.20`.
- This indicates the predicted quaternions were not matching Kabsch targets;
  the cycle term was mostly enforcing an internally consistent but non-useful
  quaternion solution.

## Quaternion-Cycle Improvement

The successful follow-up is `depth_corr_qrotcycle_f16p128_w002_ce0`.

| Run | Rot-cycle weight | Rotated CE weight | Branch solo | Fused | Delta vs clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean control | `0` | `0` | `67.01` | `85.48` | `0.00` |
| Heavy q-rot cycle | `0.05` | `0.25` | `63.69` | `85.48` | `0.00` |
| Light q-rot cycle | `0.02` | `0` | `64.73` | `85.89` | `+0.41` |

Mechanism:

1. Sample a random unit quaternion.
2. Rotate the 3D correspondence cloud by that quaternion.
3. Rotate the transformed cloud back by the inverse quaternion, forming a
   rotation cycle.
4. Penalize KL prediction drift across original, rotated, and inverse-cycled
   views.
5. Do not add rotated-view cross entropy for the best run; the heavy rotated CE
   variant hurt branch accuracy and only tied the clean fused result.

This mechanism is classifier-tied. Unlike the earlier auxiliary QCC head, the
cycle loss directly constrains the class evidence under quaternion rotation
cycles, so it cannot improve unless the classifier benefits.

## Files

- `train_corr_qcc_fusion.py`
  - Current clean branch trainer.
  - Defaults still match the clean baseline:
    - `--qcc-weight 0`
    - `--cycle-weight 0`
    - `--no-quat-inject`
  - Quaternion rotation-cycle consistency is available through:
    - `--rot-cycle-weight`
    - `--rot-aug-ce-weight`
    - `--rot-cycle-prob`
  - The old QCC loss and Kabsch injection options remain available as explicit
    ablation flags, but they are not the default path.

- `train_depth_small.py`
  - Foreground-cropped R(2+1)D depth baseline and correspondence cache builder.
  - Contains reusable quaternion helpers:
    - `axis_angle_to_quat`
    - `quat_normalize`
    - `quat_mul`
    - `quat_distance_loss`
    - `target_corr_quats`

- `sidepanel_api/server.py`
  - Parses the correspondence branch log format for phone/Anemon status:
    branch accuracy, fused accuracy, q loss, cycle loss, fusion weight, and
    temperatures.

- `sidepanel_api/fusion_watcher.py`
  - Tracks `depth_small` as a live model slot alongside `cnxxl`, DSN, and M.

## Clean Method

Inputs:

- Cached correspondence tensors:
  - `dataset/Nvidia/Processed/depth_small_cache/train_corr_f16_p128.npy`
  - `dataset/Nvidia/Processed/depth_small_cache/valid_corr_f16_p128.npy`
- Shape: `(N, 16, 128, 3)`.
- Built from processed depth point correspondences under:
  - `dataset/Nvidia/Processed/{train,test}/class_XX/subjectY_rZ/sk_depth.avi/*_pts.npy`

Feature path:

1. Normalize each correspondence cloud by subtracting sample mean and dividing
   by RMS scale.
2. Build per-point features:
   - normalized xyz
   - temporal velocity
   - displacement from the first frame
3. Encode points with a point MLP.
4. Pool each frame by mean and max.
5. Encode the sequence with a bidirectional GRU.
6. Attention-pool the frame features.
7. Classify with an MLP head.
8. Fuse branch logits with fg83 depth logits:
   `log_softmax(depth / T_depth) + w * log_softmax(branch / T_branch)`.

For the clean best run:

- Branch run: `depth_corr_qcc_f16p128_ceonly_noqinject`
- Best fused: `85.48` top-1.
- Best branch: `67.01` top-1.
- Best epoch: `202`.

## Current Goal

The current target is to improve beyond `85.89` while keeping the clean `85.48`
control as the ablation baseline.

Next controls to run before any paper claim:

1. Repeat the light q-rot cycle run with at least one additional seed.
2. Run a matched random-rotation augmentation without the inverse-cycle KL term.
3. Run a matched consistency term without quaternion rotations, if possible.
4. Lock fusion parameters on a calibration split before reporting held-out
   accuracy.
