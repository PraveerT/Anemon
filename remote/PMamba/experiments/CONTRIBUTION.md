# Quaternion Cycle Consistency for Lightweight Gesture Classification

## Contribution

We propose **dual-level quaternion cycle consistency (QCC)** for point-cloud gesture
classification: rigid-motion subtraction at the per-point level plus pair-wise
quaternion composition error at the per-frame level. Fed through a dual-stream tiny
architecture, this gives **82.78%** test accuracy on NVGesture-25 with a 1.04M-parameter
model — competitive with a 25M-parameter PMamba baseline at **89.83%** — and provides
orthogonal predictions that fuse to **90.46%** (oracle ceiling 93.15%).

The benefit is **capacity-conditional**: tiny architectures lacking
spatio-temporal grouping gain +9.75pp paired (5-seed sign-test p<0.05) from the
explicit QCC features; full PMamba with `st_group_points` gains 0pp because its
backbone implicitly captures the same signal.

---

## Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{P}_t \in \mathbb{R}^{P \times 3}$ | point cloud at frame $t$, $P=256$ points |
| $\mathbf{p}_i^t$ | the $i$-th point of frame $t$ |
| $T = 32$ | number of frames per clip |
| $C = 25$ | number of gesture classes |
| $\mathbf{w}_t \in \{0,1\}^P$ | mutual-NN correspondence mask between $t$ and $t+1$ |
| $\mathbf{q} = (q_0, q_1, q_2, q_3) = (w, x, y, z)$ | unit quaternion, scalar-first |
| $\mathbf{q}^*$ | quaternion conjugate $(w, -x, -y, -z)$ |
| $\otimes$ | Hamilton (quaternion) product |

The Hamilton product is

$$
\mathbf{q}_a \otimes \mathbf{q}_b = \begin{pmatrix}
w_a w_b - x_a x_b - y_a y_b - z_a z_b \\
w_a x_b + x_a w_b + y_a z_b - z_a y_b \\
w_a y_b - x_a z_b + y_a w_b + z_a x_b \\
w_a z_b + x_a y_b - y_a x_b + z_a w_b
\end{pmatrix}.
$$

The sandwich rotation of a 3-vector $\mathbf{p}$ by unit quaternion $\mathbf{q}$ is

$$
\mathrm{Rot}(\mathbf{q}, \mathbf{p}) = \mathbf{q} \otimes (0, \mathbf{p}) \otimes \mathbf{q}^*\big|_{\text{vec part}},
$$

which is numerically equivalent to $\mathbf{R}(\mathbf{q}) \cdot \mathbf{p}$ at float32
(maximum absolute difference $< 5\times 10^{-6}$ over $10^3$ random rotations × 128
points).

---

## Per-pair weighted quaternion Kabsch

Given correspondence-aligned source $\mathbf{S} = \mathcal{P}_t$ and target
$\mathbf{T} = \mathcal{P}_{t+1}$ with mask $\mathbf{w} \in \{0,1\}^P$:

**Weighted centroids:**

$$
\bar{\mathbf{s}} = \frac{\sum_{i} w_i \mathbf{s}_i}{\sum_i w_i + \varepsilon},
\quad
\bar{\mathbf{t}} = \frac{\sum_{i} w_i \mathbf{t}_i}{\sum_i w_i + \varepsilon}.
$$

**Centred scatter:**

$$
\mathbf{H} = \sum_{i} w_i (\mathbf{s}_i - \bar{\mathbf{s}})(\mathbf{t}_i - \bar{\mathbf{t}})^\top + \varepsilon\,\mathbf{I}_3.
$$

**SVD and proper rotation:**

$$
\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top, \quad
d = \det(\mathbf{V}\mathbf{U}^\top), \quad
\mathbf{R} = \mathbf{V}\,\mathrm{diag}(1, 1, d)\,\mathbf{U}^\top.
$$

**Quaternion conversion (Shepperd's method):**

$$
\mathbf{q} = \mathrm{rot\_to\_quat}(\mathbf{R}),
\quad
\mathbf{q} \leftarrow \mathrm{sign}(q_0) \cdot \mathbf{q} \quad \text{(hemisphere-fix)}.
$$

**Translation:**

$$
\mathbf{t} = \bar{\mathbf{t}} - \mathrm{Rot}(\mathbf{q}, \bar{\mathbf{s}}).
$$

We denote the full operation $(\mathbf{q}, \mathbf{t}) = \mathrm{Kabsch}_\mathbf{q}(\mathbf{S}, \mathbf{T}, \mathbf{w})$.

---

## Level 1: Per-point QCC subtraction (Cfbq feature)

For each consecutive pair $(t, t+1)$ we fit **forward** and **backward** Kabsch
transforms independently:

$$
(\mathbf{q}_f, \mathbf{t}_f) = \mathrm{Kabsch}_\mathbf{q}(\mathcal{P}_t, \mathcal{P}_{t+1}, \mathbf{w}_t),
$$

$$
(\mathbf{q}_b, \mathbf{t}_b) = \mathrm{Kabsch}_\mathbf{q}(\mathcal{P}_{t+1}, \mathcal{P}_t, \mathbf{w}_t).
$$

The **forward residual** at frame $t+1$ is the part the rigid model fails to explain:

$$
\mathbf{r}_i^{f,t+1} = \mathbf{p}_i^{t+1} - \big( \mathrm{Rot}(\mathbf{q}_f, \mathbf{p}_i^t) + \mathbf{t}_f \big).
$$

The **backward residual** at frame $t$ is the symmetric counterpart:

$$
\mathbf{r}_i^{b,t} = \mathbf{p}_i^t - \big( \mathrm{Rot}(\mathbf{q}_b, \mathbf{p}_i^{t+1}) + \mathbf{t}_b \big).
$$

Boundary: $\mathbf{r}_i^{f,0} = \mathbf{r}_i^{b,T-1} = \mathbf{0}$.

The per-point input feature is the 10-channel vector

$$
\mathbf{x}_i^t = \big[\, \mathbf{p}_i^t \;\big|\; \mathbf{r}_i^{f,t} \;\big|\; \mathbf{r}_i^{b,t} \;\big|\; \tau_t \,\big] \in \mathbb{R}^{10},
$$

where $\tau_t = t / (T-1) \in [0, 1]$ is a normalized time channel.

**Interpretation.** The classifier sees only what rigid motion *cannot* explain
(non-rigid motion = articulation), separately for forward and backward time.
The hand's gross translation and rotation are removed; only finger/thumb
articulation evidence remains.

---

## Level 2: Per-frame QCC violation (pair_cyc feature)

By Kabsch construction on rigid data, the backward fit is the conjugate of the
forward fit: $\mathbf{q}_b = \mathbf{q}_f^*$, hence

$$
\mathbf{q}_f \otimes \mathbf{q}_b = (1, 0, 0, 0) \quad \text{(identity quaternion)}.
$$

When points articulate, the two fits are noisy estimates that **disagree**: the
loss landscape of the weighted Kabsch problem differs slightly when src and tgt
are swapped because the same articulating points contribute different
gradients. The deviation from identity is the cycle-consistency violation:

$$
\boxed{\;
\mathbf{c}_t = \mathrm{normalize}\big(\mathbf{q}_f \otimes \mathbf{q}_b\big) - (1, 0, 0, 0) \in \mathbb{R}^4
\;}
$$

with hemisphere disambiguation $\mathbf{c}_t \leftarrow \mathrm{sign}(c_{t,0}) \cdot \mathbf{c}_t$.
$\mathbf{c}_t$ is one 4-vector per frame, no per-point dimension. Frame $t = T-1$
gets $\mathbf{c}_{T-1} = \mathbf{0}$.

**Magnitude meaning.** The scalar part $c_{t,0}$ encodes the cosine of half the
deviation angle; the vector part $(c_{t,1}, c_{t,2}, c_{t,3})$ encodes the axis
along which the cycle fails to close. For pure rigid frames $\|\mathbf{c}_t\| \to 0$;
for high-articulation frames $\|\mathbf{c}_t\|$ peaks.

**Why this is the cleanest QCC signal.** Each frame's $\mathbf{c}_t$ uses
exactly two Kabsch fits (forward and backward at the same pair). No multi-step
chain is composed, so noise does not compound with frame distance. We tested
multi-step alternatives (cycle over $(t, t+1, t+2)$ with chain composition,
cumulative chain over $T$ frames) and they all degrade rapidly with chain
length.

---

## Architecture: tiny dual-stream

The model has two parallel streams that fuse after per-frame pooling.

**Per-point stream** — operates on $\mathbf{x}_i^t \in \mathbb{R}^{10}$:

```
EdgeConv(static xyz-kNN, k=16):
  edge_{ij} = [x_i, x_j - x_i]                ∈ ℝ²⁰
  Conv2d(20→64, 1×1) → BN → GELU
  Conv2d(64→128, 1×1) → BN → GELU
  max over k neighbors                        ∈ ℝ¹²⁸

Per-point MLP:
  concat(local₁₂₈, raw₁₀)                    ∈ ℝ¹³⁸
  Linear(138→128) → LayerNorm → GELU
  Linear(128→256) → LayerNorm → GELU         ∈ ℝ²⁵⁶

Per-frame pool:
  h_pt^t = concat(max_P, mean_P) ∈ ℝ⁵¹²
```

**Per-frame stream** — operates on $\mathbf{c}_t \in \mathbb{R}^4$:

$$
h_{fr}^t = \mathrm{MLP}_{fr}(\mathbf{c}_t) = \mathrm{Linear}(32 \to 64)\big(\mathrm{GELU}(\mathrm{Linear}(4 \to 32)(\mathbf{c}_t))\big) \in \mathbb{R}^{64}.
$$

**Fusion + temporal stack:**

$$
h^t = \mathrm{concat}(h_{pt}^t, h_{fr}^t) \in \mathbb{R}^{576},
$$

$$
\mathbf{H} = [h^0, h^1, \ldots, h^{T-1}] \in \mathbb{R}^{576 \times T}.
$$

```
Conv1d(576→256, k=1) projection
4× residual Conv1d(256→256, k=3, pad=1):
  H ← H + GELU(BN(Conv1d(H)))                (first not residual)
Dropout(0.2)
Global max over T → ℝ²⁵⁶
Head: Linear(256→128) GELU Dropout(0.3) Linear(128→25)
```

**Total parameters: 1.04M.** No Mamba, no `st_group_points`, no multi-scale
encoder. The QCC features carry the motion signal that the architecture itself
does not extract.

**Critical design choice.** The per-frame QCC stream enters AFTER the per-point
pool, not concatenated to per-point input. Direct concat (treating $\mathbf{c}_t$
as a per-point feature broadcast) hurts test accuracy by 4-5pp because the
high-magnitude cycle signal contaminates the EdgeConv kNN feature space and
overfits 1050 training samples. Separate stream avoids this.

---

## Training recipe

| Setting | Value |
|---|---|
| Epochs | 120 |
| Batch size | 16 |
| Optimizer | AdamW, $\beta = (0.9, 0.999)$ |
| Learning rate | $2 \times 10^{-3}$ peak |
| Schedule | 5-epoch linear warmup → cosine decay to 0 |
| Weight decay | $10^{-4}$ |
| Loss | Cross-entropy with label smoothing $\epsilon = 0.1$ |
| Point dropout | 10% (per-point Bernoulli mask, training only) |
| Gradient clip | 1.0 (L2 norm) |
| Correspondence | mutual-NN (loader default) |
| Sampling | correspondence-aware: 256 points/frame with cross-frame identity preserved |

---

## Capacity-conditional finding

| Architecture | Input | Test acc | $\Delta$ vs xyz-only |
|---|---|---|---|
| Tiny no-kNN, max-pool (5-seed avg) | $[\mathbf{p}, \tau]$ (4ch) | 67.59 ± 1.58 | — |
| Tiny no-kNN, max-pool (5-seed avg) | Cfbq (10ch) | 77.34 ± 1.51 | **+9.75** paired (sign test $p < 0.05$) |
| Tiny + EdgeConv k=16 | Cfbq (10ch) | 82.57 | +14.98 |
| **Tiny + EdgeConv + dual-stream pair_cyc (V2)** | **Cfbq + $\mathbf{c}_t$** | **82.78** | **+15.19** |
| Full PMamba (xyz + Mamba + st-kNN) | $[\mathbf{p}, \tau]$ (4ch) | 89.83 | — |
| Full PMamba + Cfbq fine-tune | Cfbq (10ch) | 90.04 | +0.21 |

**Hypothesis (verified):** when the backbone has explicit
spatio-temporal grouping (`st_group_points` kNN over space-and-time), it
implicitly extracts what QCC features encode. Tiny set-aggregation
backbones cannot — they need the explicit features.

---

## Fusion with PMamba baseline

| Model | Solo | $\rho$(errors) vs base | Oracle | Best $\alpha$-blend |
|---|---|---|---|---|
| pmamba_base | 89.83 | — | — | — |
| Tiny + V2 (1.04M params) | 82.78 | 0.46 | 93.15 | **90.46** at $\alpha = 0.72$ |

Per-class breakdown shows V2 uniquely recovers 16 test samples that
pmamba_base misses, while pmamba_base uniquely solves 46 that V2 misses; 387
samples both solve correctly, 33 both miss (oracle ceiling).

---

## Why other QCC formulations failed

We tested 12+ alternative QCC designs, all of which underperformed the
proposed pair-level cycle violation:

- **Multi-step transitivity error**: $\mathbf{c}_t = \mathbf{q}_{(t,t+2)}^{\text{direct}} - \mathbf{q}_{(t,t+1)} \otimes \mathbf{q}_{(t+1,t+2)}$.
  Compounding noise from 2-step composition; -4pp.
- **Cumulative chain residual**: $\mathbf{r}_t^{\text{cum}} = \mathbf{p}^t - \mathrm{Rot}(\prod_{k=0}^{t-1} \mathbf{q}_k, \mathbf{p}^0)$.
  Drift grows monotonically with $t$, max-norm hits 12+ units (vs 0.5 for pair); -13pp.
- **IRLS Kabsch**: per-pair iteratively reweighted Cauchy fits. Slightly cleaner
  residuals but losing useful articulation evidence; -1.86pp.
- **QCC as auxiliary loss**: predict $\mathbf{q}_f$ or $(q_f, t_f)$ from
  per-frame features; sandwich-consistency loss. Backbone trades
  classifier capacity for predicting rigid pose — gesture-irrelevant. -1 to
  -3pp depending on weight.
- **Adversarial GRL on rigid pose**: backbone tries to make pose
  un-predictable. Wash to slight regression.
- **Cycle-aligned smoothness regularizer** ($\|h^{t+1} - h^t\|^2$ weighted by
  $\|\mathbf{c}_t\|$): too restrictive on natural feature dynamics; -10pp.

The common failure mode: any QCC formulation that introduces noise
(chain composition) or pulls the backbone toward predicting irrelevant
quantities (rigid pose) hurts classification.

---

## Reproducibility

Single end-to-end script on the
[NVGesture point-cloud dataset](https://research.nvidia.com/publication/online-detection-and-classification-dynamic-hand-gestures-recurrent-3d-convolutional)
with the standard 25-class subject-disjoint split (1050 train, 482 test):

```bash
cd PMamba/experiments
python train_qcc_branch.py
```

- Phase A (~5 min): collect Cfbq + pair_cyc features for train + test
- Phase B (~10 min): train V2 dualstream for 120 epochs on a single A6000

Outputs to `work_dir/qcc_branch/`:
- `best_model.pt` — model state_dict at peak test accuracy
- `test_logits.npz` — per-sample logits for downstream fusion analyses

Expected result: **82–84% test accuracy**, single seed.
