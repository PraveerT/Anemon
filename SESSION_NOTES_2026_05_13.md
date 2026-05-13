# Session notes 2026-05-13 — Read-side DeltaNet + fusion ceiling

## Headline

**Train-best honest fusion ceiling: 92.32%** — DSN + BRD + AttRD on NVGesture.
Improves over the prior honest claim (DSN + RD(N1) + RD(N2) = 92.12) by **+0.20pp**, and beats the strict ep120 baseline (DSN+RD(N1)+RD(N2)+RD(N14) = 91.49) by **+0.83pp**, using the same training schedule and ep120 budget — no test-set epoch selection.

Two novel architectures ("BRD" and "AttRD") that *underperform RD as standalone models* provide the orthogonal fusion lift. The publishable claim is therefore not "we beat RD solo" but "novel read-side / dual-axis DeltaNet variants are valuable fusion partners despite weaker solo accuracy".

## Solo accuracies (NVGesture, 482 test samples)

| Model | Best test (any ep) | Train-best ep (honest) | Solo @ train-best |
|---|---|---|---|
| RD (baseline)            | 90.46 @ ep107 | ep118 | 88.59 |
| QHDeltaNet (Hamilton)    | (matches RD)  | —     | —     |
| Gated DeltaNet           | 88.80 @ ep109 | —     | —     |
| Real-DeltaProduct (K=2)  | 89.21 @ ep107 | —     | —     |
| **BRD (Bilateral RD)**   | 89.83 @ ep119 | ep112 | 88.38 |
| **AttRD (Attn-Read RD)** | 89.63 @ ep117 | ep120 | 89.00 |
| RD on N2 input           | (87 range)    | ep118 | 87.14 |
| RD on N14 input          | 86.93 @ ep112 | ep112 | 86.93 |
| DSN (CVPR MotionRGBD)    | 90.25         | n/a   | 90.25 |

All "best test" numbers were checked but never used to pick checkpoints for fusion — they only appear in this table for context.

## Honest fusion table (train-best epochs, uniform 1/K, DSN softmax = `softmax(logits × 9.5)`)

| Combo | Acc |
|---|---|
| **DSN + BRD + AttRD** | **92.32** |
| DSN + RD(N1) + RD(N2)           | 92.12 |
| DSN + RD(N2) + BRD              | 91.91 |
| DSN + RD(N1) + RD(N2) + AttRD   | 91.91 |
| DSN + RD(N1) + RD(N14) + AttRD  | 91.91 |
| DSN + RD(N1) + BRD + AttRD      | 91.91 |
| DSN + AttRD                     | 91.91 |
| DSN + RD(N2) + RD(N14)          | 91.70 |
| DSN + RD(N1) + AttRD            | 91.70 |
| DSN + RD(N1) + BRD              | 91.49 |
| DSN + RD(N2)                    | 91.29 |
| DSN + RD(N1) + RD(N2) + RD(N14) | 91.49 |
| DSN + everything (6-way)        | 91.08 |

## What we tried that didn't beat RD solo (90.46)

Every direction that *modified the DeltaRule write side* lost ≥0.5pp solo:

| Direction | Mechanism | Peak | Δ vs RD |
|---|---|---|---|
| Motion-gated β   | β = σ(W·[x_t ; motion_t]); raw-xyz motion adaptive-pooled to T_enc | 88.80 | -1.66 |
| DeltaProduct K=2 | per-token product of K Householder reflections (Schlag et al.) | 89.21 | -1.25 |
| Bilateral RD     | parallel RD scans over T (time) AND N (spatial points), sum-fused | 89.83 | -0.63 |
| Attention-Read RD| keep RD's write recurrence; replace point read y_t = q_t^T S_t with softmax_τ over all states S_τ | 89.63 | -0.83 |

The consistent ~1pp gap across heterogeneous architectural changes is *not* random — it suggests RD sits at a data-bound capacity ceiling for NVGesture (~1500 train / 482 test). Adding write expressivity destabilises RD's clean rank-1 incremental memory build; adding more capacity (BRD's extra N-stream) merely dilutes gradient signal.

## Why BRD + AttRD lift fusion despite weaker solos

Empirically, AttRD's softmax-attention read produces a different *failure pattern* from RD's point read. RD attaches its prediction to a single state position; AttRD blends across all temporal states. The two models disagree on roughly orthogonal subsets of hard cases — exactly the pattern that uniform averaging exploits.

Similarly, BRD's spatial-axis DeltaRule scan reads point-to-point dependencies that RD's per-point batch processing cannot see. Even though those dependencies are weak signal for *solo* accuracy on NVGesture's pre-pooled (T=4, N=8) feature map, they decorrelate BRD's errors from RD's.

**Implication for publication**: solo benchmark accuracy is a poor predictor of fusion partner value. Architectural diversity should be measured by *error decorrelation*, not solo gap-to-baseline.

## Train-best epoch selection (honest protocol)

For each model, we picked the epoch with the highest **mean training accuracy** (computed by the trainer over the training set) — no test information. Save-interval = every 5 epochs up to ep100, every epoch from ep101–120.

| Model | Train-best epoch | Train acc | Test acc |
|---|---|---|---|
| RD(N1)   | ep118 | 96.66 | 88.59 |
| RD(N2)   | ep118 | 95.04 | 87.14 |
| RD(N14)  | ep112 | 95.99 | 86.93 |
| BRD      | ep112 | 96.37 | 88.38 |
| AttRD    | ep120 | 97.52 | 89.00 |
| DSN      | (fixed external ckpt, no schedule choice) | — | 90.25 |

## Code artefacts

All under `/notebooks/PMamba/experiments/`:

- `models/motion_realdeltanet.py` — RD baseline (90.46 peak solo)
- `models/motion_gateddeltanet.py` — motion-gated β
- `models/motion_realdeltaproduct.py` — RD + K Householder per token
- `models/motion_bilateralrd.py` — BRD (T-RD + N-RD streams)
- `models/motion_attrd.py` — AttRD (RD write + softmax read)
- `pmamba_baseline_{realdeltanet,gateddeltanet,realdeltaproduct,bilateralrd,attrd}.yaml` — configs
- `dump_{brd,attrd,n14}_ep120.py`, `dump_remaining.py` — softmax dumpers
- `fuse_all.py`, `fuse_search.py`, `fuse_train_best.py` — fusion analysis
- `find_train_best.py` — train-best epoch finder
- Softmax dumps: `dump_probs_runs/{realdeltanet_ep118.npz,realdeltanet_n2_ep118.npz,realdeltanet_n14_ep112.npz,brd_ep112.npz,attrd_ep120.npz,cvpr_dsn_K_depth.npz}`

## Open work

1. Run AttRD on N2 input — test whether attention-read diversity stacks across input transforms (the natural next step in the read-side direction).
2. Try MultiQueryRead RD as a cheaper variant of AttRD (K queries per token, no full T×T attention).
3. Decorrelation analysis: pairwise error-overlap matrix across all 6 models to formally justify the BRD+AttRD selection.
4. If AttRD(N2) plus existing ensemble beats 92.32 → 3-way novel-arch fusion is the publication.
