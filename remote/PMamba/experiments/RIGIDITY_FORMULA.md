# Per-point rigidity residual

Canonical definition. Used identically in the web viz (`viz-qcc/src/app/page.tsx`),
the QCC branch (`models/reqnn_motion.py::_kabsch_rigidity_magnitudes`), and
the depth-branch preprocess (`preprocess_rigidity.py`).

## Inputs

Two correspondence-aligned point clouds at consecutive frames:

- Frame t:   `p₁, p₂, …, p_P  ∈ ℝ³`, centroid `c  = (1/P) Σᵢ pᵢ`
- Frame t+1: `p₁', p₂', …, p_P' ∈ ℝ³`, centroid `c' = (1/P) Σᵢ pᵢ'`

`pᵢ ↔ pᵢ'` must be the **same physical point across frames** (Hungarian mapping
in the Nvidia pipeline).

## Step 1 — Center

```
uᵢ = pᵢ  − c
vᵢ = pᵢ' − c'
```

## Step 2 — Best-fit rigid rotation R_best (Kabsch)

Cross-covariance:

```
H = Σᵢ uᵢ vᵢᵀ                   (3×3)
```

SVD form:

```
U, S, Vᵀ = SVD(H)
d        = det(V · Uᵀ)          (+1 or −1)
R_best   = V · diag(1, 1, d) · Uᵀ
```

Quaternion form (Horn's method, used in the viz):

```
Σ_ij  = Hᵀ entries
N     = 4×4 symmetric matrix built from Σ
q_best = eigenvector of N with largest eigenvalue   (unit quaternion)
```

Both yield the same rotation; pick whichever is convenient.

`R_best` minimises `Σᵢ ‖R · uᵢ − vᵢ‖²` over all rotation matrices.

## Step 3 — Per-point residual

```
r̂ᵢ        = R_best · uᵢ
residualᵢ = vᵢ − r̂ᵢ
         = (pᵢ' − c') − R_best · (pᵢ − c)          (3-vector)
```

## Step 4 — Scalar magnitude

```
‖residualᵢ‖ = √( (rx)² + (ry)² + (rz)² )           (≥ 0, in world units)
```

## Interpretation

| Case | ‖residualᵢ‖ |
|------|-------------|
| Point i moves perfectly rigidly with the cloud | 0 |
| Point i articulates independently (finger curl etc.) | > 0 |
| Hungarian matched pᵢ to the wrong physical point at t+1 | > 0 (noise) |

The signal is a mix of real articulation and correspondence error; it does NOT
distinguish them on its own.

## Equivalence between implementations

| Location | R_best form | Residual line |
|----------|-------------|---------------|
| `viz-qcc/src/app/page.tsx` (SampleCard, RealSampleCard) | quaternion via Horn | `frames[fn][i] − (q_best ∘ (frames[f][i] − cP) + cQ)` |
| `models/reqnn_motion.py::_kabsch_rigidity_magnitudes` | matrix via torch.linalg.svd | `Qc − (R·Pcᵀ).T` |
| `preprocess_rigidity.py::kabsch_residuals_batch` | matrix via torch.linalg.svd | same as above |

All three compute `‖ pᵢ' − c' − R_best·(pᵢ − c) ‖` for each point.

## Frame pairing convention

Cyclic forward:

```
pair (t, t+1) for t = 0 … F-1, with (F-1, 0) closing the cycle.
```

Residual at "frame t" = residual of the forward step t → t+1.

## Notes for future tweaking

- `H` can be computed on a subset of points (e.g. only the low-residual ones
  from the previous iteration) to make Kabsch robust to articulation.
  Two-pass procedure: fit, mask high-residual outliers, refit.
- Weighted Kabsch replaces `H = Σ uᵢ vᵢᵀ` with `H = Σ wᵢ uᵢ vᵢᵀ` where `wᵢ` can
  come from correspondence confidence (already in `corr_full_weight`) — avoids
  contamination from bad matches.
- Normalising magnitudes per-sample (divide by max or by mean) makes the
  signal invariant to hand size and camera distance.
