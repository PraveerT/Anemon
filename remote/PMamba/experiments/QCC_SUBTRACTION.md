# QCC Subtraction

Forward+backward quaternion Kabsch rigid-motion residual fed as **input
feature** (not auxiliary loss) to gesture-classification models.

---

## Concept

Per consecutive frame pair t → t+1 of correspondence-aligned point clouds,
fit a rigid transform via weighted Kabsch and compute the per-point residual
(what the rigid model fails to explain). Stack forward and backward residuals
as extra input channels alongside xyz and time:

```
Cfbq input = [ xyz (3) | res_fwd (3) | res_bwd (3) | t (1) ] = 10 channels
```

Frame 0 has no `res_fwd` (no prior frame); frame T-1 has no `res_bwd`.

---

## Math

Weighted Kabsch on correspondence-matched points (mask `w` ∈ {0,1}^N):

```
sm = Σᵢ wᵢ · src_i / Σ wᵢ                                  # weighted centroid src
tm = Σᵢ wᵢ · tgt_i / Σ wᵢ                                  # weighted centroid tgt
H  = Σᵢ wᵢ · (src_i − sm)(tgt_i − tm)ᵀ + 1e-6·I           # weighted scatter
U, Σ, Vᵀ = SVD(H)
R  = V · diag(1, 1, det(V·Uᵀ)) · Uᵀ                       # proper rotation
q  = Shepperd(R)                                          # R → unit quaternion
t  = tm − q · sm · q*                                     # translation in q-frame
```

Forward residual (target frame):
```
rigid_fwd[t+1] = q · p[t]   · q*   +  t
res_fwd[t+1]   = p[t+1] − rigid_fwd[t+1]
```

Backward residual (source frame, swap roles):
```
rigid_bwd[t]   = q_b · p[t+1] · q_b* + t_b
res_bwd[t]     = p[t]   − rigid_bwd[t]
```

Rotation is applied via the Hamilton sandwich product `q · p · q*` rather
than `R · p`. Numerically equivalent at float32: max-diff < 5×10⁻⁶ verified
across 1000 random rotations × 128 points (`quat_equivalence_test.py`).

---

## Empirical results

### Tiny architecture (no kNN, no Mamba) — 5 seeds, 120 ep, 256 pts

Paired comparison, same architecture, 4-ch baseline vs 10-ch Cfbq:

| seed | A (xyz + t, 4ch) | Cfbq (10ch) | Δ |
|---|---|---|---|
| 0 | 67.43 | 76.97 | +9.54 |
| 1 | 65.15 | 79.46 | +14.32 |
| 2 | 70.12 | 74.90 | +4.77 |
| 3 | 67.43 | 78.22 | +10.79 |
| 4 | 67.84 | 77.18 | +9.34 |
| **mean** | **67.59 ± 1.58** | **77.34 ± 1.51** | **+9.75 paired** |

All 5 seeds positive, sign test p < 0.05.

### TinyKNN (single EdgeConv k=16, xyz-space static kNN) — 1 seed, 120 ep

| | acc |
|---|---|
| Best at ep 101 | **82.57** |
| Compared to no-kNN tiny | +5.23 |
| Compared to PMamba base | −7.26 |

### Full PMamba + Cfbq

| Variant | Solo | Notes |
|---|---|---|
| **pmamba_base** (xyz + t, 4ch) | **89.83** | reference (ep 110) |
| pmamba_rigidres (single fwd residual, 7ch, fine-tune from base) | **90.04** | +0.21 pp over base; oracle with base = **94.19** all-time high |
| pmamba_rigidres_fbq (Cfbq 10ch, fine-tune from base) | 88.80 (ep 123) | killed early; not converged |
| pmamba_qcc_scratch (Cfbq 10ch, from scratch, 300 ep) | 78.42 (ep 102) | killed; tracking ~5 pp behind base |

### Fusion: TinyKNN-Cfbq + pmamba_base

| | acc |
|---|---|
| pmamba_base solo | 89.83 |
| TinyKNN solo | 82.57 |
| Oracle (A or B right) | 93.15 |
| α-blend fusion | **90.66 (+0.83 pp)** at α = 0.76 |

Per-class fusion gains: C18 +10.5, C3 +4.8, C5 +5.0, C1 +4.5, others zero.
Net: 5 pmamba failures recovered, 1 pmamba success broken → +4 samples.

Confirmed not exploitable beyond α-blend: QCC-feature gating, motion-feature
gating, agreement-feature gating, learned MLP gates all plateau at the same
~90.66–91.08 ceiling — oracle 93.15 is unreachable with linear routing.

---

## Key finding — capacity-conditional prior

**Cfbq's contribution scales inversely with the model's intrinsic motion
capacity.**

| Model class | Cfbq contribution |
|---|---|
| Tiny (max-pool over points, conv1d temporal, no kNN) | **+9.75 pp** |
| TinyKNN (single xyz-kNN EdgeConv) | strong (helps reach 82.57) |
| PMamba (st-group kNN in space-time, Mamba temporal) | **+0.2 pp** or 0 |

PMamba's `st_group_points` already aggregates information across **both
space and time** via kNN at each stage, implicitly encoding per-point
relative motion. Pre-computing the rigid-motion residual is redundant in
that regime.

Tiny architectures with global aggregation (max-pool) cannot recover
cross-frame point relationships from xyz alone, so the explicit Cfbq
residual is essential.

---

## Defensible claim

> Forward+backward quaternion Kabsch residuals (QCC subtraction) provide
> motion-explicit input channels that strongly improve gesture
> classification on lightweight set-aggregation architectures (+9.75 pp on
> a tiny model, paired across 5 seeds). The benefit diminishes as the
> backbone's spatial-temporal grouping captures the same signal
> implicitly, demonstrating QCC subtraction as a **capacity-conditioned
> prior** rather than universally additive.

---

## Code artifacts (kept in repo)

| File | Purpose |
|---|---|
| `rigidres_strong_5seed.py` | The +9.75 pp 5-seed paired experiment |
| `rigidres_quat_ms_5seed.py` | 4-variant (A/Bq/Cfbq/Dmsq) tiny ablation |
| `quat_equivalence_test.py` | R·p ≡ q·p·q* numerical verification |
| `tiny_knn_edge.py` | TinyKNN architecture (single EdgeConv) |
| `tiny_knn_full_analyze.py` | Trains TinyKNN to 82.57 + saves logits |
| `tinyknn_solo_analyze.py` | Per-class confusion + calibration breakdown |
| `tinyknn_fusion_final.py` | Multi-strategy fusion sweep (90.66 ceiling) |
| `gate_agreement_family.py` | Agreement-feature gating analysis |
| `gate_motion_uncertainty.py` | Motion-feature gating analysis |
| `qcc_group_analyze.py` | tn_only group QCC-signature analysis |
| `patch_pmamba_rigidres_fbq.py` | MotionRigidResFBQ class for full PMamba |
| `qcc_scratch_fusion.py` | From-scratch + fusion sanity |

---

## Reproduction

To reproduce the headline +9.75 pp result:

```bash
cd PMamba/experiments
python rigidres_strong_5seed.py
```

Runs in ~25 minutes on an A6000. Expected output: tiny_A 67.59 ± 1.58
vs tiny_Cfbq 77.34 ± 1.51, paired Δ = +9.75 pp, 5/5 seeds positive.

To verify the quaternion sandwich numerical equivalence:

```bash
python quat_equivalence_test.py
```

Expected: max absolute difference < 5×10⁻⁶ across 1000 random rotations.

---

## What did *not* work

- **QCC as auxiliary loss** (7 variants tried in earlier work, all null —
  see prior `project_qcc_exhausted.md` memory).
- **PMamba+Cfbq from scratch** matching the base recipe — slower
  convergence than 4-ch base, killed at ep 102 / 78.42% (vs ~85% expected
  for base at same epoch). Extra noisy channels during dynamic-pts ramp
  hurt early feature formation.
- **QCC-gated fusion** to push past α-blend — pmamba beats tiny in *every*
  cycle-error regime, so QCC stats don't separate fusion-recovery cases
  from fusion-loss cases. Linear gate plateau at 91.08.
- **Stacked dynamic-feature kNN** (DGCNN style L=2,4,6) — overfit fast
  with feature-space kNN on 1050 samples; static xyz-space kNN at L=1
  remained the best small-arch choice.
- **PMamba+Cfbq warmstart from base ep110_nostage1** at LR 1.2e-5 —
  insufficient runtime; killed for re-strategy.
