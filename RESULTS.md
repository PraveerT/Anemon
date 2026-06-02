# NVGesture — headline results (no-DSN, point-cloud, CN-XXL)

Two CN-XXL solo models on NVGesture (test set = 482 clips), with the code + configs to
reproduce them under `experiments/`. Checkpoints are not committed (gitignored, `*.pt`); they
live on the training box.

## Results

| model | test acc | config | model def |
|---|---|---|---|
| **decoupled Skew-TCC aux** | **91.91** | `experiments/cn_skewrawhead_skew.yaml` | `experiments/models/motion_cleanest_skew_raw_head.py` |
| inertia-quat aux head | 91.29 | `experiments/cn_xxl_quat_head.yaml` | `experiments/models/motion_cleanest_quat_head.py` |

Both attach a **backbone-decoupled auxiliary classifier head** — its input is a *fixed transform
of the raw point coordinates* (no gradient into the backbone) — logit-ensembled with the main
classifier: `out = main_logits + scale · aux_logits`.

## Mechanism: decoupled-aux regularization

- A **decoupled** aux gives **+1 to +2** over the no-aux floor. A **coupled** aux (reading
  backbone features) competes for the representation and gives ~nothing.
- The aux is a **training-time regularizer**: its learned `scale → ~0` by convergence, so the
  benefit is baked into the main weights and the aux is effectively **off at inference**.
- Skew-TCC aux input = the **antisymmetric lagged cross-covariance** of the per-frame 3×3
  point-cloud covariance (a directed temporal "swirl" / time-arrow descriptor).

### Controlled comparison (seed 0, identical arch/schedule, only the aux mode differs)

| mode | test acc |
|---|---|
| skew (antisymmetric `(C−Cᵀ)/2`) | **91.91** |
| off (no aux) | 89.83  (floor) |
| sym (symmetric `(C+Cᵀ)/2`) | 89.21  (≈ floor) |

Two independent controls (no-aux, symmetric-aux) agree at ~89.5; the antisymmetric version is
the lone +2 outlier → the lift is **antisymmetry-specific**, not just "a decoupled aux exists."

## Caveats (honest)

- **Single seed.** The +2 over floor is well above typical run noise but not multi-seed verified.
- The antisymmetric content is a **training regularizer, not an inference feature**: forcing it
  inference-active (fixed scale + a direct aux-CE loss) underperforms (~89, ≈ the floor) because
  the descriptor is a weak standalone classifier and pollutes the ensembled output.
- **Chirality is a robustness axis, not the source of the clean +2.** Reflecting the input
  collapses CN-XXL (every 3D mirror → ~20–36% acc, OOD scatter into attractor classes; *zero*
  clean chiral pairs). The aux's clean fixes are general confusions, only marginally enriched
  for mirror-fragility. Converting the chirality axis into clean accuracy (reflection-cycle
  consistency) is untested.

## Reproduce

```bash
cd experiments
PYTHONPATH=. python main.py --config cn_skewrawhead_skew.yaml   # 91.91 (decoupled skew aux)
PYTHONPATH=. python main.py --config cn_xxl_quat_head.yaml      # 91.29 (inertia-quat aux head)
```
