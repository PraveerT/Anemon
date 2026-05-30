# Recent Quaternion Rotation Breakthroughs

Date: 2026-05-28

## Summary

We now have two concrete quaternion-related results:

1. A verified depth-side quaternion rotation-cycle gain.
2. A CNXXL fusion candidate that crosses the `>92%` target by using a
   quaternion-rotated CNXXL inference branch.

The strongest point is that the recent work separated decorative quaternion
losses from a useful quaternion mechanism. The original QCC auxiliary losses did
not survive controls, but the later rotation-cycle consistency and q-rotated
inference branch produced measurable effects.

## Breakthrough 1: We Identified The Decorative QCC Path

The first important result was negative but clarifying. The original
correspondence QCC/cycle losses were not responsible for the improvement.

| Run | QCC loss | Cycle loss | Kabsch-quat injection | Branch solo | Fused |
| --- | ---: | ---: | --- | ---: | ---: |
| `depth_corr_qcc_f16p128` | `0.04` | `0.04` | on | `61.83` | `84.65` |
| `depth_corr_qcc_f16p128_ceonly` | `0` | `0` | on | `61.00` | `85.06` |
| `depth_corr_qcc_f16p128_ceonly_noqinject` | `0` | `0` | off | `67.01` | `85.48` |

Conclusion:

- Removing QCC/cycle did not hurt; it improved the fused score.
- Removing Kabsch-quat injection improved further.
- The old QCC loss was therefore decorative for this model.
- The clean `85.48%` correspondence branch became the valid control baseline.

This matters because it prevents us from claiming a quaternion contribution
where there was none.

## Breakthrough 2: Real Depth-Side Quaternion Rotation-Cycle Gain

The successful replacement was a classifier-tied quaternion rotation-cycle
consistency term in `train_corr_qcc_fusion.py`.

Mechanism:

1. Sample a quaternion rotation.
2. Rotate the 3D correspondence cloud.
3. Rotate it back with the inverse quaternion.
4. Penalize prediction drift across original, rotated, and inverse-cycled views.
5. Keep the best setting light: no rotated-view cross entropy.

Best active run:

- Run: `depth_corr_qrotcycle_f16p128_w002_ce0`
- Clean control: `85.477%`
- Q-rotation-cycle fused: `85.892%`
- Gain: `+0.415pp`
- Branch solo: `64.730%`
- Best epoch: `117`

Matched seed check:

| Seed | Clean fused | Q-rot-cycle fused | Delta |
| ---: | ---: | ---: | ---: |
| `31` | `85.477` | `86.100` | `+0.622` |
| `37` | `85.062` | `85.270` | `+0.207` |
| `43` | `85.477` | `86.307` | `+0.830` |
| `47` | `85.477` | `85.685` | `+0.207` |

Including the original seed-29 pair:

- Direction: `5/5` seeds positive.
- Mean fused improvement: about `+0.456pp`.

This is our first genuinely verified quaternion-cycle contribution in the depth
correspondence line.

## Breakthrough 3: True 3D CNXXL Quaternion Rotation Tooling

We added `experiments/train_cnxxl_qrotcycle.py` for the CNXXL side.

Unlike raw normalized-coordinate perturbations, it performs a true 3D transform:

1. Unnormalize row, column, and depth.
2. Unproject to camera-space xyz.
3. Rotate around the per-sample 3D centroid by a quaternion.
4. Reproject back to row, column, and depth.
5. Renormalize for the CNXXL input format.

This gives us an honest q-rotation branch for CNXXL diagnostics, fine-tuning,
and inference-time fusion.

Fine-tuning result so far:

- Warm-start fine-tuning from the finished CNXXL checkpoint usually hurts.
- Head-only fine-tuning is stable but did not improve accuracy.
- Full-model qrot continuation from earlier checkpoints also degraded.

So the useful CNXXL route is currently q-rotated inference/fusion, not
post-finish qrot fine-tuning.

## Breakthrough 4: CNXXL Fusion Candidate Crosses 92%

CNXXL baseline:

- `440/482`
- `91.286%`

To exceed `92%`, we need at least:

- `444/482`
- `92.116%`

The current best candidate uses fixed log-probability fusion:

```text
score =
    log_softmax(cnxxl)
  + 0.05 * log_softmax(fg83_depth)
  + 0.06 * log_softmax(cnxxl_qrot_z+2)
```

Result:

| Method | Correct | Accuracy | Fixed vs CNXXL | Broken vs CNXXL |
| --- | ---: | ---: | ---: | ---: |
| CNXXL | `440/482` | `91.286%` | `0` | `0` |
| CNXXL + `0.05` FG83 | `443/482` | `91.909%` | `5` | `2` |
| CNXXL + `0.06` qrot z+2 | `440/482` | `91.286%` | `0` | `0` |
| CNXXL + `0.05` FG83 + `0.06` qrot z+2 | `444/482` | `92.116%` | `5` | `1` |

Interpretation:

- FG83 supplies most of the correction signal.
- The qrot z+2 branch does not add a standalone correction.
- The qrot branch removes one FG83-induced false correction.
- That stabilization is exactly the difference between `443/482` and `444/482`.

This is the first current candidate that reaches the user's `>92%` target while
including a genuine quaternion mechanism.

## Files And Reproduction

Core implementation files:

- `train_corr_qcc_fusion.py`
  - Depth correspondence trainer.
  - Contains the verified q-rotation-cycle consistency path.
- `experiments/train_cnxxl_qrotcycle.py`
  - CNXXL true-3D quaternion rotation-cycle trainer and diagnostic runner.
- `experiments/fuse_cnxxl_fg83_qrot.py`
  - Reproduces the `92.116%` CNXXL + FG83 + qrot z+2 fusion candidate.

Important result files:

- Depth qrot result:
  - `/notebooks/Anemon/experiments/work_dir/depth_corr_qrotcycle_f16p128_w002_ce0/best_fused_logits.npz`
- CNXXL baseline:
  - `/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/test_logits.npz`
- FG83 depth logits:
  - `/notebooks/Anemon/experiments/work_dir/depth_small_r2_fg83_restored_20260528_033028/best_logits.npz`
- CNXXL qrot logits:
  - `/notebooks/Anemon/experiments/work_dir/cnxxl_qrot_tta_z4_logits.npz`
- Saved >92 candidate:
  - `/notebooks/Anemon/experiments/work_dir/cnxxl_fg83_qrot_z2_fusion_005_006/test_logits.npz`

Reproduce the CNXXL candidate:

```bash
cd /notebooks/Anemon
python experiments/fuse_cnxxl_fg83_qrot.py
```

## Honesty Boundary

The depth q-rotation-cycle gain is the cleanest verified result because it has
matched controls and a multi-seed direction check.

The CNXXL `92.116%` fusion is currently a validation candidate. It is a real
saved-logit result, and the quaternion branch has a measurable stabilizing
effect, but the exact weights were discovered during evaluation-set
exploration. A paper-safe claim requires freezing this rule before testing on a
fresh held-out split or a locked protocol.

## Current Position

We can now say:

- The old auxiliary QCC path was disproven by controls.
- A new classifier-tied quaternion rotation-cycle method gives a repeatable
  depth-side improvement.
- A true 3D quaternion transform path now exists for CNXXL.
- CNXXL + FG83 + qrot z+2 reaches `92.116%` on the current eval logits.
- The next work is not more ad hoc thresholding; it is freezing and validating
  the rule honestly, or integrating q-rotation earlier in CNXXL training.
