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
| v7  | aux rigidity-predict head         | — | — | — | — | — | — | — |
| v8  | binary articulation classifier    | — | — | — | — | — | — | — |
| v9  | per-clip CE weighting             | — | — | — | — | — | — | — |
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

## Workflow

1. Implement variant (patch or new file under `depth_branch/`).
2. Smoke-test via `jlab run` (model init + one forward/backward).
3. Launch training, schedule wake-up.
4. On completion, run `depth_branch/oracle_v{N}.py best` against PMamba@e110.
5. Record row here. Push commit `depth branch v{N}: <idea>`.
6. If beats v2: keep, move on to next idea and consider stacking.
   If ties v2: move on.
   If loses to v2: diagnose in notes and move on.
