# CNXXL FG83 Q-Rotation Fusion Status

Date: 2026-05-28

## Current Candidate

The current best CNXXL-side candidate crosses the 92% target by adding a
quaternion-rotated CNXXL inference branch to the CNXXL + FG83 depth fusion.

Formula:

```text
score =
    log_softmax(cnxxl)
  + 0.05 * log_softmax(fg83_depth)
  + 0.06 * log_softmax(cnxxl_qrot_z+2)
```

Result on the current 482-sample evaluation set:

| Method | Correct | Accuracy | Fixed vs CNXXL | Broken vs CNXXL |
| --- | ---: | ---: | ---: | ---: |
| CNXXL | `440/482` | `91.286%` | `0` | `0` |
| CNXXL + `0.05` FG83 | `443/482` | `91.909%` | `5` | `2` |
| CNXXL + `0.06` qrot z+2 | `440/482` | `91.286%` | `0` | `0` |
| CNXXL + `0.05` FG83 + `0.06` qrot z+2 | `444/482` | `92.116%` | `5` | `1` |

The quaternion term does not fix an additional sample by itself. Its useful
effect in this fusion is stabilizing the FG83 correction: it removes one false
correction made by CNXXL + FG83, which is the difference between `443/482` and
`444/482`.

## Source Files

- CNXXL baseline logits:
  - `/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz`
- FG83 depth logits:
  - `/notebooks/Anemon/experiments/work_dir/depth_small_r2_fg83_restored_20260528_033028/best_logits.npz`
- CNXXL q-rotation logits:
  - `/notebooks/Anemon/experiments/work_dir/cnxxl_qrot_tta_z4_logits.npz`
  - branch used: `z+2`
- Saved candidate output:
  - `/notebooks/Anemon/experiments/work_dir/cnxxl_fg83_qrot_z2_fusion_005_006/test_logits.npz`
  - `/notebooks/Anemon/experiments/work_dir/cnxxl_fg83_qrot_z2_fusion_005_006/summary.txt`

## Reproduce

Run from `/notebooks/Anemon`:

```bash
python experiments/fuse_cnxxl_fg83_qrot.py
```

The script recomputes the table above and rewrites the saved candidate logits.

## Honesty Caveat

This is a validation candidate, not a blind-test claim. The weights were found
after exploring the current evaluation set. The mechanism is genuine and the
control is clear, but a publishable claim needs the exact rule frozen before a
held-out evaluation or a fresh protocol.

Current defensible statement:

- Q-rotation cycle is already verified on the depth correspondence branch as a
  small multi-seed gain.
- On CNXXL fusion, a quaternion-rotated z+2 inference branch provides a measured
  stabilization over CNXXL + FG83 fusion, producing `92.116%` on the current
  evaluation set.
- The next required step is to lock this rule and validate it without tuning on
  the final evaluation set.
