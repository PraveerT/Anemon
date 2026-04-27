# Quaternion residuals and cycle consistency: method and prior art

This note documents (a) how we compute per-point residuals using
quaternion-form Kabsch fits, (b) how we use cycle consistency as a feature,
and (c) the published precedent for each step.

---

## 1. Weighted point-pair Kabsch in quaternion form

Given two correspondence-aligned point sets $\mathbf{S} \in \mathbb{R}^{P\times 3}$ (source) and
$\mathbf{T} \in \mathbb{R}^{P\times 3}$ (target), with non-negative weights $w_i \ge 0$, we want
the rigid transform $(\mathbf{R}, \mathbf{t})$ that minimises

$$
\mathcal{E}(\mathbf{R}, \mathbf{t}) = \sum_{i=1}^{P} w_i \big\| \mathbf{t}_i - (\mathbf{R}\mathbf{s}_i + \mathbf{t}) \big\|^2
\quad \text{subject to} \quad \mathbf{R} \in \mathrm{SO}(3).
$$

**Closed-form Kabsch solution** (Kabsch 1976):

1. Weighted centroids
   $$\bar{\mathbf{s}} = \frac{\sum_i w_i \mathbf{s}_i}{\sum_i w_i + \varepsilon}, \quad \bar{\mathbf{t}} = \frac{\sum_i w_i \mathbf{t}_i}{\sum_i w_i + \varepsilon}.$$
2. Cross-covariance
   $$\mathbf{H} = \sum_i w_i (\mathbf{s}_i - \bar{\mathbf{s}})(\mathbf{t}_i - \bar{\mathbf{t}})^\top + \varepsilon\, \mathbf{I}_3.$$
3. SVD: $\mathbf{H} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ and reflection-fix
   $$d = \det(\mathbf{V}\mathbf{U}^\top), \quad \mathbf{R} = \mathbf{V}\,\mathrm{diag}(1, 1, d)\,\mathbf{U}^\top.$$
4. Translation: $\mathbf{t} = \bar{\mathbf{t}} - \mathbf{R}\bar{\mathbf{s}}$.

**Conversion to unit quaternion** $\mathbf{q} = (q_0, q_1, q_2, q_3)$ via Shepperd's
branchless formula:

$$
\mathrm{tr}(\mathbf{R}) > 0: \quad
s = 2\sqrt{1 + \mathrm{tr}(\mathbf{R})}, \quad
\mathbf{q} = \begin{pmatrix} s/4 \\ (R_{32}-R_{23})/s \\ (R_{13}-R_{31})/s \\ (R_{21}-R_{12})/s \end{pmatrix}
$$

(branches for the other cases are standard; see e.g. Markley 2008). We
hemisphere-fix $\mathbf{q} \leftarrow \mathrm{sign}(q_0)\cdot\mathbf{q}$ to remove the
double-cover ambiguity.

**Equivalent quaternion-domain Kabsch.** Horn (1987) gives a closed-form
solution in quaternions: the optimal $\mathbf{q}$ is the eigenvector with the
largest eigenvalue of a $4\times 4$ matrix built from $\mathbf{H}$. We use the SVD
form (faster on GPU) followed by Shepperd conversion; both produce the
same unit quaternion up to numerical precision.

### Prior art for this step

- W. Kabsch, *Acta Crystallographica A* 1976. The original $\mathbf{R}$-form
  algorithm.
- B.K.P. Horn, *J. Opt. Soc. America A* 1987. Closed-form quaternion-form
  solution to the same problem.
- M.D. Shepperd, *J. Guidance and Control* 1978. The branchless rotation-to-
  quaternion conversion we use.
- F.L. Markley, "Unit quaternion from rotation matrix", *Journal of Guidance,
  Control, and Dynamics* 2008. Survey + numerical-stability analysis of conversion
  branches.

---

## 2. Sandwich product for quaternion rotation

To rotate a 3-vector $\mathbf{p}$ by unit quaternion $\mathbf{q}$ we use the Hamilton
sandwich product

$$
\mathrm{Rot}(\mathbf{q}, \mathbf{p}) = \mathbf{q} \otimes (0, \mathbf{p}) \otimes \mathbf{q}^*\big|_{\text{vec}},
$$

where $\otimes$ is Hamilton multiplication and $\mathbf{q}^* = (q_0, -q_1, -q_2, -q_3)$ is
the conjugate. This is mathematically equivalent to $\mathbf{R}(\mathbf{q})\mathbf{p}$, where

$$
\mathbf{R}(\mathbf{q}) = \begin{pmatrix}
1 - 2(q_2^2+q_3^2) & 2(q_1 q_2 - q_0 q_3) & 2(q_1 q_3 + q_0 q_2) \\
2(q_1 q_2 + q_0 q_3) & 1 - 2(q_1^2+q_3^2) & 2(q_2 q_3 - q_0 q_1) \\
2(q_1 q_3 - q_0 q_2) & 2(q_2 q_3 + q_0 q_1) & 1 - 2(q_1^2+q_2^2)
\end{pmatrix}.
$$

We verified numerical equivalence at float32 (max absolute difference
$<5 \times 10^{-6}$ over $10^3$ random rotations × 128 points). We use the sandwich
form because it allows downstream cycle algebra (composition, conjugation)
without ever materialising $\mathbf{R}$.

### Prior art

- W.R. Hamilton, *Lectures on Quaternions* 1853 (the original).
- J. Diebel, "Representing attitude: Euler angles, unit quaternions, and
  rotation vectors", *Stanford TR* 2006. Comprehensive comparison of
  representations.
- E.B. Dam, M. Koch, M. Lillholm, "Quaternions, interpolation and animation",
  *Univ. of Copenhagen TR* 1998. Sandwich product for rotation.

---

## 3. Per-point residual feature (Cfbq)

For each consecutive frame pair $(t, t+1)$, we run two **independent** weighted
Kabsch fits (forward and backward), using the correspondence mask
$\mathbf{w}_t \in \{0,1\}^P$ (mutual nearest-neighbour matches in 3D):

$$
(\mathbf{q}_f, \mathbf{t}_f) = \mathrm{Kabsch}_q(\mathcal{P}_t \to \mathcal{P}_{t+1}, \mathbf{w}_t),
$$
$$
(\mathbf{q}_b, \mathbf{t}_b) = \mathrm{Kabsch}_q(\mathcal{P}_{t+1} \to \mathcal{P}_t, \mathbf{w}_t).
$$

Per-point **forward residual** (stored at frame $t+1$):

$$
\boxed{\; \mathbf{r}_i^{f,t+1} = \mathbf{p}_i^{t+1} - \big( \mathrm{Rot}(\mathbf{q}_f, \mathbf{p}_i^t) + \mathbf{t}_f \big) \;}
$$

Per-point **backward residual** (stored at frame $t$):

$$
\boxed{\; \mathbf{r}_i^{b,t} = \mathbf{p}_i^t - \big( \mathrm{Rot}(\mathbf{q}_b, \mathbf{p}_i^{t+1}) + \mathbf{t}_b \big) \;}
$$

Boundary frames: $\mathbf{r}_i^{f,0} = \mathbf{r}_i^{b,T-1} = \mathbf{0}$.

The classifier input per point is the 10-channel vector

$$
\mathbf{x}_i^t = \big[\,\mathbf{p}_i^t \;\big|\; \mathbf{r}_i^{f,t} \;\big|\; \mathbf{r}_i^{b,t} \;\big|\; \tau_t\,\big],
\quad \tau_t = t / (T-1) \in [0, 1].
$$

**Interpretation.** $\mathbf{r}_i^{f,t+1}$ is what the rigid model cannot explain
when going $t \to t+1$. Stacking forward and backward gives the network two
views of the same articulation evidence, evaluated against neighbours on
either side in time.

### Prior art for "rigid + non-rigid decomposition" features

- S. Vedula, S. Baker, P. Rander, R.T. Collins, T. Kanade, "Three-dimensional
  scene flow", *IEEE TPAMI* 2005. The original rigid-vs-non-rigid scene-flow
  framework on which this builds.
- J.P. Pontes et al., "Scene flow estimation as a non-rigid structure from
  motion problem", *3DV* 2018. Decomposes 3D motion into rigid + deformable
  components.
- M. Hornáček et al., "SphereFlow: 6 DoF scene flow from RGB-D pairs", *CVPR*
  2014. Per-region rigid Kabsch fits; residual = non-rigid.
- X. Liu, C. Qi, L. Guibas, "FlowNet3D: Learning scene flow in 3D point
  clouds", *CVPR* 2019. Learned scene flow that aggregates point-pair motion
  features.
- W. Wang et al., "PointPWC-Net: Cost volume on point clouds for 3D scene
  flow", *ECCV* 2020. Cost-volume reasoning over per-point motion residuals.

These all decompose 3D motion the same way (rigid pose + non-rigid leftover)
but mostly within scene-flow estimation, not gesture classification, and
not with explicit forward+backward Kabsch as fixed input channels.

---

## 4. Pair-level quaternion cycle consistency (pair_cyc)

For pure rigid motion between frames $t$ and $t+1$, the Kabsch-optimal
backward quaternion is exactly the conjugate of the forward one,
$\mathbf{q}_b = \mathbf{q}_f^*$, so

$$
\mathbf{q}_f \otimes \mathbf{q}_b = (1, 0, 0, 0) \quad \text{(identity quaternion)}.
$$

When the motion is **not** purely rigid, the two independent fits are noisy
estimates that disagree slightly: the weighted Kabsch loss landscape is
asymmetric in the source–target swap because the same articulating points
contribute different gradient mass on each side of the swap. The deviation
from identity is therefore a per-frame measure of **non-rigidity**:

$$
\boxed{\;
\mathbf{c}_t = \mathrm{normalize}\!\big(\mathbf{q}_f \otimes \mathbf{q}_b\big) \;-\; (1, 0, 0, 0) \in \mathbb{R}^4
\;}
$$

with hemisphere disambiguation $\mathbf{c}_t \leftarrow \mathrm{sign}(c_{t,0}) \cdot \mathbf{c}_t$.

**Properties:**
- Single Kabsch fit each direction, **no chain composition**, so noise does not
  compound with frame distance (in contrast to multi-step transitivity errors
  $\mathbf{q}_{(t,t+2)}^{\text{direct}} - \mathbf{q}_{(t,t+1)} \otimes \mathbf{q}_{(t+1,t+2)}$,
  which we tested and which performed poorly).
- For rigid frames $\|\mathbf{c}_t\| \to 0$. For high-articulation frames
  $\|\mathbf{c}_t\|$ peaks. The scalar part encodes the cosine of the half-deviation
  angle; the vector part encodes the axis around which the cycle fails to close.
- $\mathbf{c}_t \in \mathbb{R}^4$ is one 4-vector per frame. No per-point dimension.

**How we use it.** $\mathbf{c}_t$ is fed to a small MLP $\mathbb{R}^4 \to \mathbb{R}^{32} \to \mathbb{R}^{64}$
that runs **after** the per-point pooling step, producing a per-frame 64-dim
vector. This is concatenated with the 512-dim per-frame point-pool feature,
then sent to the temporal Conv1d stack and classifier head. The motivation
for injecting it after pooling (not as a per-point channel broadcast) is
that the 4-vec cycle signal has a much larger magnitude than typical xyz
coordinates and would otherwise contaminate the EdgeConv kNN feature space;
direct concatenation cost us 4–5pp in ablation.

### Prior art for cycle / forward–backward consistency

**As loss / regulariser** (mature, widely used):
- T. Brox, J. Malik, "Large displacement optical flow", *IEEE TPAMI* 2011.
  Forward–backward consistency check for optical flow.
- N. Sundaram, T. Brox, K. Keutzer, "Dense point trajectories by GPU-accelerated
  large displacement optical flow", *ECCV* 2010. Forward–backward consistency
  to detect occlusions.
- S. Meister, J. Hur, S. Roth, "UnFlow: Unsupervised learning of optical flow
  with a bidirectional census loss", *AAAI* 2018. Forward–backward
  consistency loss for unsupervised flow.
- Y. Wang et al., "Occlusion aware unsupervised learning of optical flow",
  *CVPR* 2018. Bidirectional cycle consistency under occlusion.
- J.-Y. Zhu et al., "Unpaired image-to-image translation using cycle-
  consistent adversarial networks", *ICCV* 2017 (CycleGAN). Cycle consistency
  as adversarial regulariser between domains.
- L. Carlone, R. Tron, K. Daniilidis, F. Dellaert, "Initialization techniques
  for 3D SLAM: A survey on rotation averaging", *ICRA* 2015. Rotation cycle
  consistency in graph SLAM.
- A. Zhu, L. Yuan, K. Chaney, K. Daniilidis, "EV-FlowNet: Self-supervised
  optical flow estimation for event-based cameras", *RSS* 2018. Forward–
  backward cycle as self-supervision.

**As a feature** (much rarer):
- T. Zhou, Y.J. Lee, S.X. Yu, A.A. Efros, "Flowweb: Joint image set alignment
  by weaving consistent, pixel-wise correspondences", *CVPR* 2015. Cycle
  inconsistency to detect bad correspondences.
- D. Detone, T. Malisiewicz, A. Rabinovich, "SuperPoint: self-supervised
  interest point detection and description", *CVPR-W* 2018. Cycle-consistency
  signals embedded in keypoint training, but not used as input feature.

**Quaternion-domain specifically:**
- R. Hartley, J. Trumpf, Y. Dai, H. Li, "Rotation averaging", *IJCV* 2013.
  Cycle consistency in quaternion / rotation space, but for averaging, not
  as a feature.
- L. Carlone, F. Dellaert, "Duality-based verification techniques for 2D SLAM",
  *ICRA* 2015. Quaternion cycle algebra for SLAM.

To our knowledge, **using $\mathbf{q}_f \otimes \mathbf{q}_b - \mathbf{I}$ as a passive
per-frame input feature for downstream classification** has not been published.
Closest precedents use it as a loss (e.g. UnFlow, CycleGAN) or as a quality
signal for outlier rejection (FlowWeb). The framing as a stable, passive
feature for non-rigidity detection in a gesture-classification pipeline is
the engineering contribution we attempt to substantiate empirically.

---

## 5. Architecture: dual-stream injection

Per-point stream consumes $\mathbf{x}_i^t \in \mathbb{R}^{10}$ (Cfbq):

$$
\text{EdgeConv}_{k=16}(\mathbf{x}^t) \;\to\; \text{MLP}_{\text{point}} \;\to\; \mathrm{maxpool}_P \,\Vert\, \mathrm{meanpool}_P
\;\to\; h^t_{\text{pt}} \in \mathbb{R}^{512}.
$$

Per-frame stream consumes $\mathbf{c}_t \in \mathbb{R}^4$:

$$
\mathrm{MLP}_{\text{cyc}}(\mathbf{c}_t) \;\to\; h^t_{\text{cyc}} \in \mathbb{R}^{64}.
$$

Concatenation, then temporal stack:

$$
h^t = h^t_{\text{pt}} \,\Vert\, h^t_{\text{cyc}} \in \mathbb{R}^{576},
\quad
\mathbf{H} = [h^0, \ldots, h^{T-1}] \in \mathbb{R}^{576 \times T}.
$$

$$
\mathbf{H} \to \text{Conv1d}(576\to256) \to 4\times \text{ResConv1d}_{k=3} \to \text{maxpool}_T \to \text{Linear}(\to C).
$$

Total parameters: 1.04M.

### Prior art for two-stream / dual-stream architectures

- K. Simonyan, A. Zisserman, "Two-Stream ConvNets for Action Recognition",
  *NeurIPS* 2014. The original two-stream RGB + optical-flow design.
- C. Feichtenhofer et al., "Convolutional two-stream network fusion for video
  action recognition", *CVPR* 2016. Fusion variants for two-stream.
- C. Feichtenhofer, H. Fan, J. Malik, K. He, "SlowFast networks for video
  recognition", *ICCV* 2019. Two-stream with different temporal resolutions.

We follow the spirit of these designs but apply them to a point-cloud
gesture-recognition pipeline with a non-pixel motion stream: per-frame
quaternion cycle violation rather than dense optical flow.

---

## 6. Summary: which steps are routine, which are our specific contribution

| Step | Routine | Our specific framing |
|---|---|---|
| Weighted Kabsch in $\mathbf{R}$-form (Kabsch 1976) | ✓ | reused as-is |
| Quaternion-form Kabsch (Horn 1987) | ✓ | reused via SVD + Shepperd |
| Sandwich rotation $\mathbf{q} \otimes \mathbf{p} \otimes \mathbf{q}^*$ | ✓ | reused; numerical equivalence verified |
| Forward Kabsch residual as feature | ✓ (scene flow) | adapted to gesture clip input |
| **Forward + backward Kabsch residual concatenated** | partial precedent | specific 10-channel design |
| Forward–backward cycle consistency as loss | ✓ | not what we do — see next row |
| **$\mathbf{q}_f \otimes \mathbf{q}_b - \mathbf{I}$ as passive per-frame feature** | not published | this is the framing we add |
| **Dual-stream injection at per-frame post-pool stage** | partial (two-stream) | specific architectural choice |
| Per-point + per-frame fusion via Conv1d temporal stack | routine | reused |

Our contribution is therefore not in any individual mathematical step but
in the **engineering pipeline**: combining quaternion Kabsch residuals with
pair-level cycle consistency as **passive features** in a dual-stream
classifier, demonstrating that this works at small model scale on a
specific gesture-recognition task while documenting that 12 alternative
QCC-as-loss formulations do not help.

---

## 7. What still needs validation before a strong claim

- 5-seed paired test of "Cfbq alone" vs "Cfbq + pair_cyc dualstream" with
  fixed Phase A. Currently single-seed result is within seed noise.
- Multi-dataset transfer (SHREC'17, MSR-Action3D, NTU-RGBD) to show the
  capacity-conditional finding generalises beyond NVGesture.
- Comparison to published gesture baselines on NVGesture with similar
  parameter budgets (1M params), not just full PMamba (25M).
- Ablation isolating $\mathbf{c}_t$ contribution from Cfbq: train V0 (Cfbq only)
  and V2 (Cfbq + pair_cyc) under identical Phase A.
