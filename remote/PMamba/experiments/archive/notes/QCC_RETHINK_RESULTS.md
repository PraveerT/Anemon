# QCC Rethink — Results Summary

Full ablation ladder from the 2026-04-19/20 QCC rethink work. All runs: same
dataset (Nvidia Dynamic Gesture, 25 classes), same recipe (Adam LR 5e-4,
step at 100/120/130, 140 epochs, Hungarian correspondence where applicable),
from scratch, single seed.

## Final Table

| run  | description                                    | peak acc | Δ vs v6q |
|------|------------------------------------------------|---------:|---------:|
| v6q  | xyz only, no aux (honest baseline)             | 75.93%   | 0        |
| v6p  | xyz + mutual corr + prediction aux             | 76.14%   | +0.21    |
| v15a | xyz + Hungarian corr + prediction aux          | 74.07%   | -1.86    |
| v16a | xyz + Hungarian + K=6 parts rigidity aux (A+B) | 73.03%   | -2.90    |
| v17a | xyz + Hungarian + K=6 parts cycle aux (C)      | 74.69%   | -1.24    |
| v18a | xyz + Hungarian + K=6 parts feature only       | 73.44%   | -2.49    |
| v19a | residual scalar only, simple conv              |  4.56%   | -71.37   |
| v20a | shortest-rot quat only, simple conv            |  4.56%   | -71.37   |
| v21a | tops quat only, simple conv                    |  4.56%   | -71.37   |
| v22a | local-normal only, simple conv                 |  4.56%   | -71.37   |
| v23a | local-normal + xyz, simple conv                |  4.36%   | -71.57   |
| v24a | local-normal only, FULL arch                   | 59.96%   | -15.97   |
| v25a | tops only (centroid-radial), FULL arch         | 64.73%   | -11.20   |
| **v26a** | **xyz + tops together, FULL arch, 7-ch**   | **78.01%**   | **+2.08** |

## Key Findings

### 1. Hungarian correspondence alone doesn't fix QCC
- v15a (Hungarian + pr aux) at 74.07% trailed v6p (mutual + pr aux) at
  76.14%. Doubling correspondence density didn't improve results.
- Rules out "more/better correspondence" as the lever for QCC.

### 2. K=6 parts rotation supervision hurts more than it helps
- Three different K=6 formulations (v16a rigidity, v17a cycle, v18a
  feature-only) all trailed v6q by 1–3pp.
- Aux losses compete with classification loss; derived-rotation
  features duplicate what EdgeConv already extracts.

### 3. Architecture capacity matters more than feature choice
- Five "derived-feature-only" simple-architecture tests (v19a–v23a) all
  flatlined at ~4.5% (random). We initially read this as "feature has no
  signal." Wrong.
- Same features fed through full architecture (v24a normals, v25a tops)
  learned 60–65%. The simple PointNet-style conv couldn't extract the
  signal; the full arch could.
- **Lesson:** feature-only ablations need the same backbone as the baseline
  to be fair. Don't conclude "feature is information-weak" from
  capacity-limited arch.

### 4. Tops field carries ~85% of xyz's classification signal
- v25a (tops only, full arch): 64.73% = 85% of v6q's 75.93%
- v24a (kNN-PCA normals only, full arch): 59.96%, +4.77pp worse than tops
- Why tops beats true normals: centroid-radial is globally consistent,
  kNN-PCA normals flip sign at articulated regions (fingertips, creases)
  where gesture differences live.

### 5. **Adding tops alongside xyz beats the xyz-only baseline (+2.08pp)**
- v26a: 7-channel input `[x, y, z, t, dir_x, dir_y, dir_z]` through the
  full BearingQCCFeatureMotion architecture.
- First EdgeConv conv enlarged to accept 14 inputs (vs baseline 8).
- Peak **78.01% vs v6q's 75.93%**. First positive QCC-rethink result.

## Architectural Anatomy of the Win

**What doesn't work:** adding QCC as an **auxiliary loss** or as
**multiplicative modulation** on encoded features. Both hurt 1–3pp
because:
  - Aux loss gradient competes with classification loss
  - Multiplicative modulation can destroy rather than augment signal
  - The full architecture already extracts rotation-aware features from xyz

**What works:** adding the tops direction as an **explicit input channel**
alongside xyz. The first conv layer can learn to weight xyz vs tops; later
layers see an enriched per-point representation from the start. The
explicit channel saves the network from rediscovering centroid-radial
structure via EdgeConv neighborhood ops.

## Implementation Notes

See `patch_tops_xyz_input.py` + `patch_tops_xyz_v2.py`. Class
`TopsXYZInputMotion` in `models/reqnn_motion.py`. Usage:

```yaml
model: models.reqnn_motion.TopsXYZInputMotion
model_args:
  pts_size: 96
  num_classes: 25
  hidden_dims: [64, 256]
  dropout: 0.05
  edgeconv_k: 20
  merge_eps: 1.0e-06
  bearing_knn_k: 10
  qcc_weight: 0.0
  qcc_variant: contrastive
```

The model computes tops from sampled xyz on-the-fly (no precomputed
cache required), so this change is orthogonal to correspondence-mode
choice. Paired with Hungarian correspondence in v26a for consistency
with earlier experiments.

## Caveats

- **Single seed.** The +2.08pp gain is not yet confirmed over multiple
  seeds. Typical seed noise on this model is ≈±1–2pp.
- **Not orthogonal to existing rigidity modulation.** v26a keeps the
  bearing-QCC rigidity modulation path active. Ablation (rigidity off +
  tops on) not yet run.
- **Dataset-specific.** 25-class Nvidia Dynamic Gesture at framerate 32.
  Generalization to other gesture / action datasets untested.

## Next Directions

If the +2.08pp is worth validating:
1. Re-run v26a with 3–5 seeds to confirm significance
2. Ablation: rigidity modulation off + tops input on (isolate contributions)
3. Augment tops with per-point magnitude (|p - centroid|) as 8th channel
4. Try tops + xyz on the full PMamba fusion model (current best: 90.25% with xyz)
