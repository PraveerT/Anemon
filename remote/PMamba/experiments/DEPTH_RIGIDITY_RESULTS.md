# Depth-branch results table

All runs: depth+tops CNN-LSTM backbone, 140 epochs, same hparams as v2, Adam
lr=1e-3 step [80,110,130], batch=16. Oracle vs PMamba @ e110 (89.83%) on 482
test samples.

**Target: beat depth_v2** (93.36% oracle, +0.83pp fusion gain).

## Results

| run | idea | solo | oracle | headroom | fusion α | fusion gain | only-depth | both wrong |
|-----|------|------|--------|----------|----------|-------------|------------|------------|
| v1  | depth-only 1ch                    | 64.73 | 92.95 | +3.11 | 0.90 | +0.21 | 15 | 34 |
| **v2**  | **+ tops 4ch (baseline)**      | **66.39** | **93.36** | **+3.53** | **0.85** | **+0.83** | **17** | **32** |
| v3–5 | QCC aux losses (3 variants)      | collapse | — | — | — | — | — | — |
| v6  | rigidity stats concat to LSTM     | 58.92 | 92.95 | +3.11 | 1.00 | +0.00 | 15 | 34 |
| v7  | aux rigidity-predict head (MSE, λ=0.1) | 64.73 | 92.12 | +2.28 | 0.90 | +0.21 | 11 | 38 |
| v8  | binary articulation aux (BCE median) | 57.26 | 92.12 | +2.28 | 0.90 | +0.00 | 11 | 38 |
| v9  | per-clip CE reweight (β=2)        | 61.62 | 92.95 | +3.11 | 0.90 | +0.41 | 15 | 34 |
| v9-clean β=0.5 | batch-norm softplus weight | — | 93.57 | +3.74 | 0.90 | +0.42 | — | — |
| v9-clean β=1   | batch-norm softplus weight | 62.86 | 93.57 | +3.74 | 0.90 | +0.41 | 18 | 31 |
| **v9-clean β=1.5** | **batch-norm softplus weight** | — | **94.19** | **+4.36** | 0.90 | **+0.83** | — | — |
| v9-clean β=2   | batch-norm softplus weight | 56.85 | **94.40** | **+4.57** | 0.85 | +0.00 | **22** | **27** |
| v10 | rigidity-contrastive              | — | — | — | — | — | — | — |
| v11 | rigidity-gated readout            | — | — | — | — | — | — | — |

## Benchmarks (non-depth-branch)

| source | solo | oracle | fusion gain |
|--------|------|--------|-------------|
| PMamba (baseline) | 89.83 | — | — |
| velpolar (quat branch) | 79.67 | 91.29 | 0.00 |
| rigidity_v1 (quat + rigidity 5th ch) | 68.26 | 91.70 | +0.21 |
| rigidity_attn_v1 (quat + rigidity gate) | 67.01* | 91.91 | +0.21 |

*killed pre-convergence; best-so-far.

## Findings after v6 – v9

Four rigidity-as-supervisory-signal variants on depth_v2 backbone:

| run | oracle Δ vs v2 | fusion Δ vs v2 |
|-----|----------------|-----------------|
| v6  (concat to LSTM)        | −0.41 | −0.83 |
| v7  (MSE aux)               | −1.24 | −0.62 |
| v8  (BCE median aux)        | −1.24 | −0.83 |
| v9  (per-clip CE reweight)  | −0.41 | −0.42 |

**All four lose.** No variant reaches parity with v2 on oracle OR on fusion.

**Interpretation:** Per-frame rigidity summary features, as computed from
Hungarian-correspondence-aligned 256-point clouds with vanilla Kabsch, do NOT
carry new class-discriminative signal beyond what the depth+tops CNN-LSTM
backbone already extracts from pixels.

Plausible reasons (untested):
- Kabsch residuals are dominated by correspondence errors, not true
  articulation. Without a weighted/robust fit, the "rigidity" scalar is
  noisy-signal + noisy-noise.
- The depth CNN already sees articulation directly in pixels (finger outlines,
  contour curvature). A derived scalar is redundant.
- Aggregating per-point residuals to 6 per-frame scalars throws away spatial
  information.

Next axes (untested):
- **Weighted Kabsch** using `corr_full_weight` from the dataloader →
  cleaner q_best, less contaminated residuals.
- **Outlier-robust Kabsch** (2-pass: fit, mask high-residual outliers, refit)
  → q_best reflects the rigid majority cleanly.
- **Rasterised per-pixel rigidity map** as a 5th CNN channel (preserves
  spatial info; harder to implement because of pixel↔point mapping).
- **Abandon rigidity axis** and pursue other orthogonality sources (e.g.,
  different backbone family, ensemble).

## Workflow

1. Implement variant (patch or new file under `depth_branch/`).
2. Smoke-test via `jlab run` (model init + one forward/backward).
3. Launch training, schedule wake-up.
4. On completion, run `depth_branch/oracle_v{N}.py best` against PMamba@e110.
5. Record row here. Push commit `depth branch v{N}: <idea>`.
6. If beats v2: keep, move on to next idea and consider stacking.
   If ties v2: move on.
   If loses to v2: diagnose in notes and move on.
