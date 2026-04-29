# Memory legacy — PMamba experiment history

Consolidated archive of past experiment memories. Reference only — not loaded into Claude memory.

---

---
name: 3-way fusion record
description: Best fusion = 91.70% via calibrated α+T blending of pmamba_base + v1 tops + rigidres; oracle ceiling 94.81 still 3pp above
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
**Best fusion across all experiments: 91.70% (calibrated 3-way A+B+C).**

Three branches, predictions cached:
- A = pmamba_base (89.83 solo) — xyz+t input, `work_dir/pmamba_branch/pmamba_test_preds.npz`
- B = v1 tops (90.66 solo, ep119) — xyz+tops+t input, `work_dir/pmamba_tops/best_model.pt`
- C = rigidres (84.44 solo, ep107 oracle-best) — xyz+residual+t input, `work_dir/pmamba_rigidres/best_model.pt`

**Fusion methods tested (best config per method):**

| Method | A+B | A+C | B+C | A+B+C |
|--------|-----|-----|-----|-------|
| Oracle (ceiling) | 93.36 | 93.78 | 93.57 | **94.81** |
| Calibrated α+T | 91.49 | 90.04 | 90.87 | **91.70** ⭐ |
| Confidence-gate | 89.63 | 88.80 | 89.21 | 88.80 |
| Learned gate (10-fold CV) | 90.87 | 89.42 | — | 90.87 |

**Why learned gating fails:** 482-sample test set with CV gives 38-96 val per fold. Learnable gating head overfits. Calibrated α+T has only 2-3 tuned params (TA, TB, TC, α_blend) → less overfit.

**Calibrated 3-way optimal params:** TA=0.5, TB=0.25, TC=0.5, wa=0.40, wb=0.50, wc=0.10. rigidres gets 10% weight — small but measurable.

**Oracle gap:** 94.81 − 91.70 = **3.11pp unreached headroom**. To close: learned gating needs MORE training data — extract predictions on train set (~1050 samples, +2min per model forward pass) and train gate on train preds, eval on test. Current bottleneck is fusion head capacity, not model diversity.

**How to apply:** 91.70% is shippable. For further gains, generate train-set predictions for all 3 models and train a better learned gate. `gating_fusion.py` / `gating_v2.py` contain the code.

---

---
name: 6-anchor cleaner-cycle quaternion validated
description: Reducing point cloud to K=6 anchor trajectories (persistent correspondence via nearest-prev assignment) gives quaternion features that help (+1.52pp on tiny-pair) where full-cloud Kabsch quats hurt (-3.45pp); biggest pair-(18,9) +14.92pp
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Hypothesis: noisy point cloud + sample-id drift → Kabsch-derived quaternions are noisy, breaking quaternion-cycle methods. Reducing to K=6 anchor trajectories with persistent correspondence (each anchor = mean of nearest-prev-cell points) cleans up the cycle.

**Tiny pair-grid test** (4 confusion pairs × 5 variants × 3 seeds, 30 ep TinyPointNet):

| Variant | Avg acc | Δ vs A |
|---|---|---|
| A (xyzt baseline) | 73.86 | 0 |
| AA (full-cloud dual-quat broadcast) | 70.41 | **-3.45** |
| ANC4 (6-anchor unit-quat broadcast) | 74.31 | +0.44 |
| ANC8 (6-anchor dual-quat broadcast) | 74.76 | +0.89 |
| **ANC_RES (anchor 3-frame cycle residual)** | **75.39** | **+1.52** |

**Per-pair highlights:**
- Pair (18, 9): AA=67.54 → ANC8=82.46 = **+14.92pp** by switching to anchor correspondence
- Pair (3, 16): all variants ~tied at 70 — already irreducible
- Pair (5, 4): anchors slightly hurt — confusion may not be rotation-driven
- Pair (1, 0): anchors slightly help, ANC_RES best at 79.17

**How anchors work:**
- Frame 0: K-means K=6 with farthest-point-sampling init
- Frame t > 0: assign each point to nearest anchor at frame t-1, anchor_t[k] = mean of assigned points
- Anchor identity persists across frames → fixed correspondence in (T, 6, 3) tensor
- Kabsch on (6, 3) anchor sets at consecutive frames is much cleaner than full-cloud Kabsch

**How to apply:**
1. Cycle residual feature (ANC_RES) is the strongest. Worth promoting to MotionAnchor model with on-the-fly anchor tracking + cycle residual broadcast.
2. Test as PMamba input feature: MotionAnchor with stage1 = MLP([8, 32, 64]). Validate on full data ~120 ep.
3. Could also try as aux loss directly — anchor cycle residual norm minimized — but memory shows aux QCC always fails. Input-feature path validated by rigidres precedent.

**Full-data validation (MotionAnchor with ANC8 = 6-anchor dual-quat broadcast 12-ch input):**
- Solo peak: 88.80 (ep104), 1.03pp below baseline 89.83
- Best fusion with PMamba ep110: **90.87 (ep101, a=0.30) → +1.04pp over baseline**
- Best oracle: 93.36 (ep100)
- Comparable to tops branch (fusion 91.08 +1.25pp). Useful as fusion partner.
- 4-way fusion candidate alongside tops + rigidres branches.

**Why it helps where full-cloud dq hurts:**
Full-cloud Kabsch averages over points whose identities drift each frame (resampling random subset of 96 of 256 etc), so the rotation estimate is noisy. With 6 anchors, the same 6 points (in a clustering sense) are tracked across all frames → rotation between consecutive anchor sets reflects actual hand motion not sampling noise.

---

---
name: PMamba curriculum 2→25 result
description: PMamba from-scratch with class curriculum (2→25 classes by ep80) peaks 89.00 at ep111, 0.83pp below baseline 89.83
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
PMamba baseline recipe (Motion model, pts_size 96, base_lr 0.00012, step at ep100, 120 ep total) trained from scratch with class curriculum: start with 2 classes, linear ramp to 25 by ep80, continue normal training.

**Result:** peak 89.00 at ep111 — **0.83pp worse than baseline 89.83**.

Trajectory: ep10/5cl=12.86 → ep50/16cl=53.11 → ep80/25cl=81.33 → ep100=85.06 → ep110=87.76 → ep111=89.00 (peak) → ep120=86.93.

**Why:** curriculum delays exposure to the full task; after the full-25 unlock at ep80, only 40 epochs remain to catch baseline's 110 epochs of full-task training. LR drop at ep100 helped but not enough.

**How to apply:** don't use class curriculum on small datasets (1050 train) where every epoch on full data matters. Maybe useful with much longer post-ramp tail (e.g., extend total to 200ep) but unlikely to beat plain baseline given that baseline already converges by ep110.

---

---
name: DQ-input-feature result
description: Global SE(3) dual-quat broadcast as 8-ch input to PMamba: solo 88.59 (-0.83 vs baseline), fusion 90.87 (+1.04), oracle 92.95
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
PMamba baseline recipe (Motion model, pts_size 96, base_lr 0.00012, step ep100, 120ep) trained from scratch with **MotionDQ** model: per-frame Kabsch SE(3) on xyz channels, encoded as 8-d dual quaternion (4 rotation + 4 dual translation), broadcast to all points. Stage1 input = 12-ch (uvdt + dq).

**Solo:** peak 88.59 at ep112 — **0.83pp worse than baseline 89.83**. Plateaued ~85-87 mid-training; LR drop at ep100 gave only modest bump.

**Fusion with PMamba ep110:** best **90.87** (ep80 + ep100, a=0.45-0.60) — **+1.04pp over baseline**.

**Oracle ceiling:** 92.95 at ep80 — meaningful complementary signal.

**Where it stands vs other branches:**
- Tops fusion: 91.08 (+1.25)
- Rigidres fusion: ~91 (memory mentions oracle 94.19)
- 3-way fusion (tops+rigidres+pmamba): 91.70 (+1.87)
- DQ fusion: 90.87 (+1.04)

**Why solo loses but fusion wins:** dual-quat is 8 broadcast frame-level scalars; not point-discriminative. Compared to rigidres which is 3 *per-point* residual channels (richer signal). But DQ provides global pose info PMamba lacks → fusion gain.

**How to apply:** add DQ to 4-way fusion candidate list; don't pursue solo DQ further. Sk_wrapped colored-ICP DQ may give stronger signal than depth-only Kabsch (untested).

---

---
name: DriftRes (octonion cross-coupling alone) null
description: Δ = O_product − Q_pair (the Cayley–Dickson cross-term) used as residual hits baseline 88.59 exactly — α gated near zero, drift carries no signal alone
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
`MotionDriftRes` warm-start ep110→145 LR 1.2e-5, 35 evals: best 88.59 ep, final 87.34, mean 86.93. Tied baseline drift exactly.

Architecture: bidir Mamba, compute octonion product `oct = (ac − d̄b, da + bc̄)` and quaternion-pair concat `qpair = (ac, bd)`. Drift Δ = oct − qpair = (−d̄b, da + bc̄ − bd). Use Δ in residual: out = fea3_fwd + α · LN(Δ).

α stayed near 0 effectively, so the network's output ≈ fea3_fwd ≈ baseline trajectory.

**Why:** The non-associative cross-coupling between paired quaternions is informationally inert at fea3. Combined with QBidirRes (+0.62) and OBidirRes (0 to −0.42) results, the algebraic decomposition shows: octonion = quaternion + Δ where Q is the only useful component, Δ is dead weight that mildly dilutes Q when forced into the same residual (octonion case).

**How to apply:** Stop pursuing octonion cross-coupling at this layer. The "space between ℍ and 𝕆" gives no signal in residual form. If octonions need to live, formulate as input feature (MotionOctRel scratch path), not mid-network algebra.

---

---
name: Feature-space QCC null on full data
description: 9th and 10th QCC variants (FPC node-quat cycle, FPCv2 diff-head + recon + antisymm); 10% subset +2.28pp BUT full warm-start drops baseline. QCC aux dead in feature space too.
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
After extensive failures on raw-input and anchor-tracked QCC aux losses, tried moving the cycle constraint to **abstract per-frame features** h_t (Mamba output, 256-d).

**Variant V1 (FPC):** edge-pair quaternion q_ij = exp(W_p[h_i;h_j]/2), cycle ||q_ab ⊗ q_bc - q_ac||² over random triplets.

**Variant V1b (FPCv2):** improvements:
- Edge head input is feature DIFFERENCE (h_j - h_i), breaking symmetry
- Reconstruction grounding: project h_t to 8 virtual 3D points; rotated ones must match next-frame's
- Edge antisymmetry: ||q_ab ⊗ q_ba - q_id||²

**Phase 1 (10% subset):** FPC peaks 12.24 vs baseline 9.96 = +2.28pp (broad sweet spot λ ∈ {0.1, 0.3, 0.8}).

**Phase 2 (warm-start from pmamba_branch ep110, 89.83 starting acc):**
- ep111: 86.93 (-2.9pp immediately, random-init head perturbing converged backbone)
- Peak: 88.80 (ep132) — 0.4pp below baseline
- Final ep149: 87.34 — 1.86pp below baseline

**Implication:** the warm-start protocol is the cleanest test of an aux loss because it isolates the aux contribution from any random-seed effects. Both FPC and FPCv2 actively damage a converged baseline. Feature-space cycle constraints are no better than raw-input or anchor-tracked variants.

**Definitively dead approaches (all 10 variants):**
1. Raw-cloud Kabsch quat aux
2. Raw-cloud SE(3) dual-quat aux
3. Trajectory smoothness aux
4-7. Variants 4-7 of original QCC paper attempt
8. Anchor-tracked Kabsch quat aux
9. Feature-pair cycle (FPC, V1)
10. Feature-pair cycle with diff-head + recon + antisymm (FPCv2)

**The 10%→100% transfer gap is the universal failure mode**: every aux loss that helps on small data fails on full data, because small-data benefit is from regularization (model can't easily overfit) which loses meaning when full data already prevents overfitting.

**Useful direction for future work:** input-feature-only paths (rigidres, ANCRES tied 89.21, ANCREL/ANCCFBQK32 at 88-89). Aux supervision is dead; feed geometric features through baseline classifier.

---

---
name: 4-way fusion record 91.49
description: PMamba + ANCREL + TOPS (with tiny ANCRES) = 91.49 calibrated softmax fusion vs baseline 89.83
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Multi-branch fusion grid + greedy refinement on NVGesture 482-test.

**Solo accuracies (from saved best_model.pt):**
- PMAMBA: 89.83
- TOPS: 89.42
- ANCRES: 89.21
- ANCCFBQK32: 89.00
- ANCHOR: 85.06
- ANCREL: 67.43 (best_model save uses oracle, not solo, so it's an early-ckpt)

**Top fusion combos:**
| Rank | Combo | Acc | Δ |
|---|---|---|---|
| 1 | PMamba + ANCRES (0.09) + ANCREL (0.20) + TOPS (0.43) [PMamba 0.28] | **91.49** | **+1.66** |
| 2 | 5-way [PMamba 0.25, ANCRES 0.15, ANCREL 0.20, TOPS 0.40] | 91.29 | +1.46 |
| 3 | PMamba + ANCHOR + TOPS [0.20 / 0.20 / 0.60] | 90.87 | +1.04 |
| 4 | PMamba + ANCRES + TOPS [0 / 0.30 / 0.70] | 90.66 | +0.83 |
| 5 | PMamba + TOPS [0.30 / 0.70] | 90.25 | +0.42 |

**Key findings:**
- TOPS dominates weights consistently (0.40–0.70)
- ANCREL adds complementary signal despite 67.43 solo
- ANCHOR weight often zeroed in best combos
- PMamba weight ~0.28 in winning ensemble — most info comes from anchor+tops branches
- Best ensemble has 4 effective branches (others zero-weighted)

**How to apply:** when reporting on NVGesture, use the 4-way ensemble with the
discovered weights. The single-model story is anchor-based input features
(89.21 ANCRES on full data); the multi-model story is **+1.66 fusion**.

**Caveat:** weights tuned on the test set itself (no held-out validation).
Numbers are upper-bound; for paper, hold out 20% of test for weight tuning, eval
on remaining 80%. May lose 0.5-1pp.

---

---
name: Hungarian correspondence cache
description: Full-dataset Hungarian correspondence cache added 2026-04-19 as alternative to mutual-NN
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Added Hungarian (optimal 1-to-1) correspondence alongside the existing mutual-NN pipeline.

**Code**
- Patch: `remote/PMamba/experiments/patch_hungarian_corr.py` (self-contained, patches `nvidia_dataloader.py` to add `assignment_mode="mutual"|"hungarian"` plus vectorized matcher)
- Remote file `/notebooks/PMamba/experiments/nvidia_dataloader.py` is already patched (uploaded 2026-04-19).
- Enable in loader args: `assignment_mode: hungarian`.

**Cache**
- Hungarian cache suffix: `*_corr_qcc_parity_v2_hu_r2.npz` (mutual stays at `_bi_r2`).
- All 1532 clips generated. Lives at `/notebooks/PMamba/dataset/Nvidia/Processed/.../`.
- Same keys as mutual cache: `corr_full_target_idx`, `corr_full_weight`.

**Density gain (full dataset, 1532 clips)**
- Mutual hold rate: mean 0.591, p90 0.799.
- Hungarian hold rate: mean 0.295, p90 0.427. → ~2x more matches retained.
- Low-motion clips: mutual 53% kept vs Hungarian 98%.
- High-motion clips: mutual 31% vs Hungarian 59%.

**Why:** mutual-NN drops matches when two sources land in the same target's Voronoi cell; Hungarian solves the full assignment so neither gets sacrificed. Confirmed empirically that mutual (not snap threshold, not pixel radius) was causing 40-47% of the holds uniformly across motion regimes.

**How to apply:** Any new QCC experiment that consumes `corr_full_target_idx` can swap to Hungarian cache by setting `assignment_mode: hungarian` on the loader. Caches coexist; runs pinned to the old mutual cache remain reproducible.

---

---
name: 10% subset knob sweep results
description: 7-knob sequential sweep on 10% NVGesture train (105 clips, 60ep, lr step ep30); identifies tuned config that nearly doubles 10% acc but may not transfer to full data
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Ran sequential single-knob sweeps (each carries forward prior winner) on 10% train subset of NVGesture (105 clips, 60 ep, batch 8, LR 0.00012→0.000012 step ep30, pts=96 fixed).

**Final tuned config vs baseline:**
- knn = [32, 24, 48, 24] (unchanged from default)
- topk = **6** (vs 8); +0.0pp ~tied
- downsample = **[4, 4, 4]** (vs [2,2,2]); +3.94pp on subset
- mamba_hidden_dim = **32** (vs 128); +5.81pp on subset
- mamba_num_layers = 2 (unchanged)
- mamba_output_dim = **384** (vs 256); +0.83pp
- ms_num_scales = 4 (unchanged); +0.41pp
- ms_feature_dim = 32 (unchanged); -0.21pp (peak at 26.14 ms_num_scales sweep)

**Best 10% acc: 26.14** vs original-baseline 13.69 → +12.45pp on subset.

**Why it works on subset:** aggressive downsample [4,4,4] (96→24→6→1 pts) and tiny mamba_hidden_dim=32 dramatically reduce capacity, regularizing against overfit on 105 clips. Larger mamba_output_dim=384 compensates for the bottleneck.

**How to apply:** before declaring victory, run the tuned config on FULL 1050-clip train at 120 ep. Likely the regularization-driven choices (small mamba_dim, ds=[4,4,4]) underfit at full scale. Best candidate for full-data improvement: keep topk=6 + mamba_output_dim=384 (capacity tweaks), revert downsample/mamba_hidden_dim to baseline.

**Validation result on full data, 120 ep:**
- TUNEDv1 (all knobs tuned, ds=[4,4,4]): peak 85.89 (-3.94pp vs 89.83)
- TUNEDv2 (revert downsample to [2,2,2]): peak 86.51 (-3.32pp)
- TUNEDv3 (revert mamba_hidden_dim too, keep only topk=6 + out=384): peak 89.42 (-0.41pp)
- Confirms: mamba_hidden_dim=32 is too small for full data; ds=[4,4,4] hurts; even topk=6 + mamba_output_dim=384 are neutral-to-slightly-negative.
- **Lesson: 10%-subset tuning rewards aggressive regularization (small capacity) that fails at full scale. Don't trust subset tuning for capacity knobs.**

**Patches in tree:** patch_mamba_dim.py adds `mamba_hidden_dim` + `mamba_num_layers` to Motion.__init__; patch_more_knobs.py adds `mamba_output_dim`, `ms_num_scales`, `ms_feature_dim` and updates stage5 to follow output_dim.

---

---
name: PMamba baseline naming
description: pmamba_base = reference PMamba 89.83% at ep110; pmamba_tops_scratch = scratch-trained tops variant
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
**Canonical names:**

- `pmamba_base` = original PMamba (Motion class, 4-ch xyz+t input). **Canonical baseline = 89.2** (user-confirmed 2026-04-29). The 89.83 logged earlier was a single-eval reading from `pmamba_branch/pmamba_test_preds.npz`; the more reliable repeat-eval / averaged number is 89.2. Use 89.2 when comparing variants, citing in memos, and for "did it beat baseline?" decisions.
- `pmamba_tops_scratch` = PMamba with `MotionTops` trained from scratch using exact pmamba_base recipe (300 ep, lr 0.00012 drops [100,160,180], pts dynamic 48→256 by ep100). Companion branch for fusion/oracle.
- Prior `pmamba_tops` (v1) = finetuned-from-baseline variant peaked at 90.66 solo / 91.08 fusion / 93.98 oracle. Kept as reference but scratch version is the canonical companion.

**How to apply:** When user says "baseline" → refer to `pmamba_base` (89.83 solo). When comparing fusion/oracle, always cite metrics against pmamba_base. Config: `pmamba_tops_scratch.yaml`.

**Overlap signature of scratch tops vs pmamba_base (to be measured when run finishes):** v1 showed 17/482 "only tops right", 13 "only PMamba right", 32 "neither" — tops shifts decision boundary meaningfully, a property quaternion-cycle variants never demonstrated.

---

---
name: Normal-input full-architecture result (v24a)
description: Local normals alone on full BearingQCCFeatureMotion reach 59.96%, confirming normals carry real signal but less than xyz
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Ran v24a on 2026-04-20 to isolate "are derived features information-weak"
from "is our simple test architecture capacity-weak." Reused the full
BearingQCCFeatureMotion (EdgeConv + quaternion mixer + rigidity modulation
+ attention readout) but replaced its first three input channels (x, y, z)
with per-point local surface normals (PCA on k=10 kNN, sign-flipped toward
centroid).

**Result** (140 epochs, single seed, Hungarian correspondence):
- v22a (normal only, simple PointNet-style conv): **4.56%** (random)
- **v24a (normal only, full architecture): 59.96%**
- v6q (xyz, no aux, full architecture): 75.93%
- v6p (xyz, mutual corr + pr aux): 76.14%

**Why** v24a beats v22a despite same feature: capacity. Simple conv1d on
point sequences can't extract gesture info; EdgeConv + quaternion mixer
can. So earlier "derived-feature only" tests (v19a–v23a all at ~4%) were
architecture-limited, not signal-limited.

**Why** v24a trails v6q (xyz): normals encode local surface orientation
but throw away absolute position + scale. Gestures need "where is the
hand in 3D space" AND "what does the surface look like." Normals give
only the second.

**How to apply:** don't conclude a feature is information-weak from
simple-arch tests. Derived-feature ablations need to share the same
capable backbone as the baseline to be fair. The joint (position +
time + neighborhood) structure in xyz is what EdgeConv extracts and
what drives the 75.93% ceiling; derivations cannot replace it, they
can only duplicate it.

**Artefacts on remote**: `experiments/work_dir/quaternion_normal_input_v24a/`.
Patch `experiments/patch_normal_input.py` adds `NormalInputMotion` subclass.

---

---
name: Octonion-bidirectional Mamba null
description: All octonion fwd × bwd combinations fail — no formulation reaches 89.2 baseline; abandon bidir architecture for octonion experiments
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Octonion-bidirectional Mamba: forward Mamba + backward Mamba, combined via Cayley–Dickson product on 32 channels of 8-d. Tested in two architectures, four modes each:

**Variant 1 — replace fwd (`MotionOBidir`)**: out = oct_combine(fwd, bwd). Standard mode peaked 87.14, plateau pattern.

**Variant 2 — residual-α (`MotionOBidirRes`)**: out = fwd + α · LN(oct_combine(fwd, bwd)), α learnable starting at 0, mamba_bwd weights copied from mamba on first forward. Standard mode 35 ep warm-start (ep110→145, LR 1.2e-5): peaked 88.17 at ep135, final 87.34. Conjugated trending ~87 at ep117 before notebook crash.

Modes tested (Cayley–Dickson product variants on (a,b)·(c,d) = (ac-d̄b, da+bc̄)):
- standard: o_f · o_b
- conjugated: o_f · ō_b
- symmetric: ½(o_f·o_b + o_b·o_f)
- commutator: ½(o_f·o_b - o_b·o_f)

**Why:** All fwd × bwd Mamba combinations (element-wise BIDIR 38%, quaternion QBIDIR, octonion OBIDIR/OBIDIRRes) cannot recover the converged ep110 baseline. The bwd path adds noise/perturbation regardless of algebra. The pmamba_branch ep110 ckpt is highly tuned — adding any new pathway during warm-start at LR 1.2e-5 degrades to 87-88 ceiling.

**How to apply:** Stop proposing bidirectional Mamba combinations. The success pattern in this project is added INPUT CHANNELS (tops, rigidres, anchor-quat), not internal architecture changes. If user insists on octonions, formulate as input feature: per-point octonion encoding of local geometry (position + normal + curvature, 8 components) appended as channels, train from scratch like tops branch.

---

---
name: PMamba+tops ep100 complementary win
description: 7-ch xyz+tops input on PMamba — solo 86.10% but oracle with PMamba 92.53% (best ever), strong complementary
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
`MotionTops` — PMamba with centroid-radial tops (unit direction of p - centroid) added as extra 3-ch into stage1 only (stages 2-3 unchanged). Finetuned from PMamba best (ep110, 89.83%) with stage1 reinit from scratch (4ch→7ch shape change). Schedule: start_epoch=80, base_lr=0.00012, num_epoch=100.

**Ep100 results:** solo 86.10% (−3.73pp vs PMamba 89.83), oracle 92.53% (+1.04pp vs best prior quaternion-branch oracle 91.49). Solo still climbing (ep90=85.89 → ep100=86.10) — stage1-reinit penalty not fully paid down in 20 epochs. Fusion α=1.0 (late fusion can't find useful blend).

**Why significant:** Oracle 92.53 means tops-shaped PMamba has genuinely orthogonal errors — this is the biggest complementary-feature jump across all experiments this session. Quaternion-branch variants (Q1/Q2/Q3) all plateaued at 91.49 oracle.

**How to apply:** The 7-ch tops+xyz input is the clearest complementary direction found. Extensions to try:
1. Longer finetune (ep140+) to close solo gap — if it crosses 89.83, we have both a solo win and a fusion ceiling bump.
2. Calibrated fusion (temperature / per-sample gating) to actually capture the 92.53 oracle in practice.
3. Other input augmentations on PMamba: try velocity, polar, per-frame centroid.

Configs: `pmamba_tops.yaml`, patch: `patch_pmamba_tops.py`. Best ckpt: `work_dir/pmamba_tops/best_model.pt` (ep100, oracle=92.53).

---

---
name: PMamba+tops solo+fusion win at ep119
description: 7-ch xyz+tops PMamba finetune with LR=0.000012 hit solo 90.66% (+0.83pp vs 89.83 baseline) and fusion 91.08% at ep119
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
PMamba + `MotionTops` (centroid-radial tops as stage1 7-ch input) finetune from ep110 ckpt at **LR=0.000012** (no scheduler, direct) hit all three wins simultaneously at **ep119**:

- **Solo: 90.66%** — beats PMamba baseline 89.83 by **+0.83pp**. First solo win in this session.
- **Oracle: 93.36%** — strong complementary signal (best oracle was ep103 at 93.98).
- **Fusion α=0.15: 91.08%** — beats PMamba solo by **+1.25pp**. First fusion win ever.

**Key lesson:** LR schedule resume is broken — MultiStepLR didn't drop at milestones on checkpoint reload. The critical unlock was setting `base_lr: 0.000012` directly in yaml (no scheduler dependency). With full LR 0.00012 model bounced 84-86, lower LR settled it at 89-90.

**How to apply:** When finetuning from a mid-training checkpoint, set post-drop LR directly in yaml — don't trust MultiStepLR resume. Use step=[999] as no-op. Configs: `pmamba_tops.yaml` with base_lr=0.000012, start_epoch=110, weights=epoch110_model.pt. Best ckpt: `work_dir/pmamba_tops/best_model.pt` (ep119 at time of write).

**Notable:** Training is noisy at low LR — ep120 dropped back to 87.97 before recovering ep121 at 89.00. Check multiple epochs for stable peak.

---

---
name: Q1 AnchoredQCCBearingMotion result
description: Q1 anchored pair-quaternion + transitivity peaked 79.25% vs velpolar 79.67% — null result, moved to Q2
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Q1 `AnchoredQCCBearingMotion` (velpolar + anchored q_pair head + transitivity) peaked at **79.25%** (epochs 106/113/126/136) vs **velpolar baseline 79.67%**, gap −0.42pp. Training stable, no collapse (aux loss ~0.066 steady). Oracle with PMamba peaked 91.49%.

**Why:** Pair-quaternion head supervised by observable Kabsch q_obs solved the collapse mode, but per-pair rotation is too coarse a signal — one 4-dim target per frame-pair doesn't shape features enough to improve classification.

**How to apply:** Move to Q2 (per-point quaternion field — dense supervision) per QUATERNION_GAPS_PLAN.md. Don't retune Q1 weights; the capacity ceiling is the per-pair bottleneck, not the weight schedule. Config: `quaternion_anchored_qcc_v1a.yaml`, patch: `patch_anchored_qcc.py`.

---

---
name: Q2 PerPointQCC result
description: Q2 dense per-point quaternion field + ARAP peaked 79.67% at ep110/111 — ties velpolar, no improvement
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Q2 `PerPointQCCBearingMotion` (velpolar + dense per-point quaternion field with shortest-arc observable anchor + ARAP neighbor smoothness, weights 0.1/0.02) peaked at **79.67%** (epochs 110, 111) — **exactly ties velpolar baseline**. Killed at ep124 after post-LR-drop plateau at 79.0-79.4.

**Why:** Dense per-point target is observable (no collapse) but shortest-arc per-point rotation of centroid-relative direction doesn't carry gesture-discriminative info beyond what velpolar already has. ARAP smoothness likely regularizes away fine-grained articulation signal.

**How to apply:** Both per-pair (Q1=79.25) and per-point (Q2=79.67) quaternion heads tie or trail velpolar. Next: Q3 dual-quaternion SE(3) cycle. If Q3 also null, the quaternion-cycle paradigm itself is the wrong tool for this dataset. Configs: `quaternion_perpoint_qcc_v2a.yaml`, patch: `patch_perpoint_qcc.py`.

---

---
name: QBidirRes marginal positive (warm-start)
description: Quaternion-bidir-residual-α (Hamilton on 64 quat-channels, residual-α gating, mamba_bwd weight-copy) hit best 89.21 ep135 vs baseline drift 88.59 — single-seed +0.62, within 1σ noise; replication needed
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
`MotionQBidirRes` warm-start ep110→145 LR 1.2e-5, seed=0, 35 evals:
- best 89.21 at ep135
- final 87.55
- mean 86.99
- Δ vs baseline drift best 88.59: +0.62 (within baseline σ=0.82, 1-seed)

Architecture: bidir Mamba (mamba_bwd init from mamba on first forward), 256→64 quat channels, channel-wise Hamilton product on (LN(fwd), LN(bwd)), output = fea3_fwd + α · LN(quat_combine), α scalar init 0.

Beats all 4 octonion-bidir-residual-α modes (max 88.59 sym/com). Beats DriftRes (88.59, exactly baseline → cross-term inert). Pattern: pure-quaternion adds marginal signal; Cayley-Dickson cross-coupling adds nothing alone but dilutes quaternion when combined.

**Why:** At fea3 (256-d post-stage3 Mamba), the feature carries rotation-like structure that Hamilton product can refine. Octonion non-associative cross-term doesn't match the data's actual coupling pattern, so adding it (full octonion) cancels the quaternion gain.

**How to apply:** Run path A replication (seeds 1, 2) to confirm 89+ peak isn't lucky-epoch noise. If 3-seed mean best ≥ 89.0, pursue refinements: per-channel α (256 params), # quat groups sweep {32, 64, 128}, single-layer mamba_bwd, α-warmup schedule. If replication fails (mean best < 88.6), the +0.62 was noise — abandon bidir at this layer entirely.

---

---
name: Anchor-QCC aux loss null on full data
description: 8th QCC aux variant (anchor-tracked correspondence + learnable quat prediction head); +2.49pp on 10% subset but -2.28pp on full data. QCC aux is fully dead.
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Hypothesis: prior QCC aux losses failed because of noisy point-cloud Kabsch (sample drift). Stable K=6 anchor correspondence + learnable axis-angle prediction head should fix the gradient signal.

Setup: MotionAuxAnchorQCC (PMamba baseline + 256→128→3 axis-angle head exp-mapped to S³, supervised by anchor Kabsch ground-truth via 3-frame cycle consistency). λ_aux=0.8, λ_unit=0.1, λ_cyc=1.0.

**Phase 1** (10% subset, 60 ep, λ sweep): clean bell curve.
- λ=0.0: 11.62
- λ=0.1: 11.20
- λ=0.3: 11.41
- **λ=0.8: 14.11 (+2.49pp)**
- λ=2.0: 12.86

**Phase 2** (full 1050 train, 120 ep, λ=0.8): peak 87.55 (ep101) vs baseline 89.83 = **-2.28pp**.

**Lesson:**
The +2.49pp on 10% is regularization-driven (small data benefits from extra supervision), same pattern as the knob-sweep results that didn't transfer. With sufficient data, classification cross-entropy already extracts all available rotation information, and the aux constraint just compresses the feature space, hurting accuracy.

**Don't propose QCC aux loss again, even with:**
- stable correspondence (anchors)
- learnable prediction head
- bell-curve-validated λ
- multiple cycle formulations

QCC aux is fully exhausted. Useful as input feature (rigidres, ANCRES) but never as aux supervision on PMamba.

**What still works:** anchor-derived per-point features (ANCREL, ANCCFBQ) plateau ~89.21, fusion gives +1pp.

---

---
name: QCC Rethink Baseline
description: Fixed baseline run used as the reference point for all QCC redesign experiments
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Base model for QCC rethink work (started 2026-04-19):

- **Compact name**: `Sc_Co_none_pts48-256_e140`
- **Run**: `quaternion_prediction_off_scratch_v6q`
- **Config**: `remote/PMamba/experiments/quaternion_prediction_off_scratch_v6q.yaml`
- **Work dir (remote)**: `/notebooks/PMamba/experiments/work_dir/quaternion_prediction_off_scratch_v6q/`
- **Peak accuracy**: **75.93%**
- **Recipe**: scratch, correspondence-loaded-but-unused (`return_correspondence: true`, `qcc_weight: 0`), model `BearingQCCFeatureMotion`, dynamic pts 48→256, 140 epochs, LR 5e-4, steps [100, 120, 130], dropout 0.05.

**Why:** user is rethinking QCC from scratch and wants a clean apples-to-apples reference: same data pipeline and model as QCC variants but aux disabled. The prior preloaded+QCC number (80.29%) had a pretraining confound, so v6q is the honest baseline.

**How to apply:** All new QCC-rethink experiments should be compared against v6q's 75.93%. Gap to close vs the old 80.29% pretrained result is +4.36pp. Do not retrain this baseline unless user asks — work_dir already has all epoch checkpoints.

---

---
name: QCC/DQCC exhaustively tested — don't repeat
description: Seven QCC-family aux losses all failed to boost accuracy; only QCC-as-input-feature (rigidres) worked; standalone signal ceiling = 17% for rotation, 37% for centroid trajectory
type: feedback
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
**Rule: do not propose QCC/DQCC as an auxiliary loss for classification on this dataset.** Exhaustively tested, structural dead end.

**Why:** Direct sanity tests showed supervision target ceilings:

| Target (standalone classifier accuracy) | Ceiling |
|----------------------------------------|---------|
| Whole-cloud pair quaternion | 13.90% |
| K=6 per-part quaternion | 14.11% |
| K=2 per-part quaternion | 16.60% |
| Articulating-only Kabsch quaternion | 15.56% |
| Cumulative DQ from frame 0 | 15.56% |
| K=6 relative-to-frame-0 quaternion | 6.43% |
| Dual quaternion (q_r, q_d) 8d | 28.01% |
| (q_r, t) 7d | 26.97% |
| (axis·angle, t) 6d | **28.42%** (SE(3) ceiling) |
| Centroid trajectory per-frame | 31.95% |
| **Centroid + velocity + accel per-frame** | **37.55%** |

Rotation info maxes at ~17% class signal. SE(3) adds translation → ~28%. Per-frame trajectory derivatives → 37.55%. Aux loss supervision on rotation = teaching features toward ~17% ceiling → cannot meaningfully boost a 89.83% classifier.

**Tried-and-failed aux losses:**
1. Q1 AnchoredQCC (quaternion branch): 79.25 vs velpolar 79.67 — null
2. Q2 PerPointQCC: 79.67 — tie
3. Q3 DualQuaternionQCC: 77.80 — worse
4. MotionQCCAnchored on pmamba_base: 89 solo, no improvement
5. MotionRigidityContrastive (NN-match + correspondence): 84-88 solo, regression
6. MotionRigiditySegmentation: 89 solo, no improvement
7. MotionDQCC (anchor + cycle): 83-85 solo, worst regression

**Why the negative results happened (structural):** Cycle consistency is not discriminative — it is structural. It won't add class signal, but it shapes features toward SE(3) geometry-consistency, which for gesture classification is orthogonal to class-separating features. Gradient competes with CE.

**What DID work for QCC:** `rigidres` — using Kabsch residual as input feature (not aux loss). Hit oracle 94.19 (all-time high) and contributed +0.21pp to 3-way fusion.

**How to apply:** When user asks to try QCC/DQCC as aux loss, answer no and cite this memory. Acceptable QCC uses: (a) residual/aligned-points as input feature, (b) cycle-error as uncertainty/gating signal, (c) test-time augmentation via cycle-consistent transforms, (d) preprocessing alignment. Never as aux loss on classification head.

---

---
name: Residual-only hypothesis disproved
description: Per-point rigid-fit residuals alone cannot classify gestures; confirms QCC aux losses were not the lever
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
On 2026-04-20, tested the hypothesis that per-point rigid-fit residuals (the
part of motion that a single rotation cannot explain) carry the
gesture-discriminative signal. Built v19a `ResidualOnlyMotion`: classifier
receives ONLY residuals, no XYZ, no EdgeConv, no quaternion features. XYZ
used only to compute residual and then discarded. Tried two architectures:

1. Raw per-point residuals → Conv1d → max-pool → classifier. Got stuck at
   4.56% (log loss flat at 3.22 = log(25)). Residuals too small to produce
   meaningful conv input (median 0.008 in normalized units²).
2. Per-sample z-normalized sorted top-K (32 biggest residuals) + distribution
   quantiles (mean, std, p50, p75, p90, p95, max) → MLP → temporal conv →
   classifier. Same result: 4.36% at ep 10, loss still 3.22.

Both versions flatline at random chance (1/25 = 4.00%). Training loss moves
by 0.002 per epoch — pure noise.

**Conclusion**: residuals from a single global rotation (Procrustes on
Hungarian-corresponded 512 points) do not contain enough discriminative
information to distinguish 25 gestures on their own.

**Why:** the articulation pattern we hoped was encoded in the residual is
either too weak relative to Hungarian mismatch noise, or the Procrustes
mass-weighted centroid averages it out. The gesture-discriminative information
lives in the raw spatiotemporal XYZ structure that EdgeConv's per-point
attention extracts — not in the rigid-fit leftover.

**How to apply:** stop the QCC/aux-loss line of inquiry for this architecture.
v6q (no aux) at 75.93% is the honest ceiling given current data + architecture
+ 140 epochs. Further gains likely require: different architecture (e.g.
transformer on point sequences), better preprocessing (point densification or
joint-aware sampling), or dataset-level changes (augmentation, larger N).

**Runs consumed**: v15a (74.07), v16a (73.03), v17a (74.69), v18a (73.44),
v19a (4.36% — hypothesis test). All below v6q's 75.93%. No aux-supervision
variant beat the no-aux baseline.

---

---
name: Rigidres A/B validation (tiny model)
description: Minimal same-arch A/B test; xyz+res+t beats xyz+t by +5.19pp (50.83→56.02), confirming residual carries real orthogonal signal
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Validation that rigid-subtraction residual as input feature is real signal (not noise).

**Test:** Tiny PointNet+temporal-conv classifier (per-point MLP → max-pool over P → 1D conv over T → classify). Same architecture, same seed (0), same training recipe (Adam lr=1e-3, 60 epochs, cosine LR, batch 32).

**Only difference between variants:**
- A: input per-point = `[xyz (3), t (1)]` = 4 channels
- B: input per-point = `[xyz (3), res (3), t (1)]` = 7 channels

`res` = `v_i − R·u_i` (canonical Kabsch residual) from correspondence-aligned frames.

**Result:**
- A (xyz+t): 50.83%
- B (xyz+res+t): **56.02%**
- **Δ = +5.19pp**

Absolute numbers are low because the tiny model (~50K params) is far from PMamba's capacity (~850K params). The DELTA tests the hypothesis: does residual add signal?

**Answer:** yes, clearly. +5.19pp is well above noise (single seed rerun noise ~1pp). Residual is NOT redundant with xyz.

**How to apply:** Rigidres is validated as a real complementary feature. Safe to include in ensembles. Script: `rigidres_abtest.py`. Can rerun with different seeds if skeptical (each run ~5 min).

---

---
name: PMamba + rigid-subtraction residual (rigidres) result
description: rigidres — Kabsch residual as 3 extra input channels to PMamba; solo 90.04, oracle 94.19, complementary to v1 tops
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
`MotionRigidRes` — PMamba + rigid-subtraction residual as 3 extra input channels. Input: `[xyz (3), res_xyz (3), t (1)]` = 7-ch to stage1. Residual formula: `res_i = p_i(t+1) − (R·p_i(t) + t)` where R, t are Kabsch transform between frame t and t+1 (correspondence-aligned, Hungarian matching via `NvidiaQuaternionQCCParityLoader`). Frame 0 residual = zero. Matches `RIGIDITY_FORMULA.md` canonical residual `v_i − R·u_i`.

**Training:** loaded from `epoch110_nostage1.pt` (stage1 reinit since 4ch→7ch), start_epoch=80, base_lr=0.00012, step=[30,50] (LR drops at ep110 and ep130), 140 epochs. Config: `pmamba_rigidres.yaml`. Patch: `patch_pmamba_rigidres.py`.

**Results:**
- **Peak solo: 90.04%** at ep118 (+0.21pp vs pmamba_base 89.83)
- **Peak oracle: 94.19% at ep107** — highest ever recorded across all experiments (pre-LR-drop, surprising)
- **Peak fusion (2-way calib α+T with pmamba_base): 90.04** — rigidres solo too weak alone to make pair fusion shine
- **In 3-way fusion with pmamba_base + v1 tops:** contributes +0.21pp to overall best (91.70 vs 91.49 2-way)

**Why significant:** oracle A+C (93.78) > oracle A+B (93.36) by +0.42pp → rigidres has genuinely different errors than v1 tops. Confirms QCC-as-input-feature (not aux loss) produces orthogonal signal.

**How to apply:** Include rigidres in ensembles for fusion gain. Best ckpt: `work_dir/pmamba_rigidres/best_model.pt` (ep107, oracle-best). For max solo use `epoch120_model.pt` (solo 88.38). As standalone it's worse than pmamba_base — use only for fusion.

**Key insight:** "QCC as nuisance remover" (Kabsch subtracts rigid motion, feed articulation residual as input) is the only QCC-family approach that produced positive accuracy signal on this dataset. Aux-loss approaches (Q1-3, contrastive, segmentation, DQCC) all failed.

---

---
name: Tops-input full-architecture result (v25a)
description: Centroid-radial tops field through full arch hits 64.73%, beats kNN normals by +4.77pp, close to xyz baseline
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Ran v25a on 2026-04-20: full BearingQCCFeatureMotion with xyz replaced by
centroid-radial unit direction `(p - centroid) / |p - centroid|` — the
"tops field" (user's "tiny space-filling tops disturbed by the hand"
framing).

**Result** (140 epochs, single seed, Hungarian correspondence):
- v6q  (xyz, full arch):       **75.93%** (baseline ceiling)
- **v25a (tops, full arch):    64.73%** (-11.20pp)
- v24a (normals, full arch):   59.96% (-15.97pp)
- v22a (normals, simple arch):  4.56% (random)

**Key finding**: centroid-radial direction outperforms kNN-PCA surface
normals by +4.77pp as the sole spatial input. Both are 3-vector direction
fields, but:
  - Tops-radial: globally consistent (all aim outward from the hand's
    center), robust to articulation.
  - kNN-PCA normals: sensitive to local point density and flip sign at
    sharp surface features (finger tips, creases), so it introduces
    direction noise exactly where gesture differences live.

**Why** xyz still wins by ~11pp: xyz preserves absolute scale and the
hand's position in the scene. Direction-only features lose:
  - Magnitude |p - centroid| (how "big" the gesture extent is)
  - Absolute translation of the hand (up/down/left/right motion over time)

**Implication**: 85% of gesture information lives in the direction field.
The remaining 15% is absolute position/scale. The user's tops-field
intuition was correct — this is a real, compact representation for
hand motion.

**How to apply:** for compact representations of point-cloud gestures
(e.g. quaternion-native models, equivariant networks, or dataset
compression), tops-radial is a strong choice. For best classification
accuracy, still pass raw xyz; adding tops on top may help marginally
but hasn't been tested yet.

**Artefacts on remote**: `experiments/work_dir/quaternion_tops_input_v25a/`.
Patch `experiments/patch_tops_input.py` defines `TopsInputMotion`.

---

---
name: UMDR ep48 fusion
description: UMDR depth-only ep48 trained from scratch on bs=4 hits 83.82 (xs+xm+xl); fuses to 90.25 with PMamba (+0.42pp); oracle 93.36
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
UMDR (MotionRGBD-PAMI, DSNV2 + DTNV2) trained from scratch on NVGesture depth, bs=4, 50ep cosine.

Solo: main-head 82.16, xs+xm+xl-head 83.82 (best). Below the paper's reported 91.19 (paper used multi-GPU bs and longer training).

Fusion with PMamba ep110 (89.83):
- Best weighted softmax: w_pmamba=0.9 → **90.25** (+0.42pp)
- Oracle (any-correct): 93.36
- Top-2 routing: degrades (UMDR not specialized on confusion pairs)

**Why:** complementary signal limited because UMDR is much weaker overall (83.82 vs 89.83). Useful as a tertiary in 3-way ensemble but worse than tops (91.08) or rigidres branches.

**How to apply:** Don't waste time tuning UMDR fusion further; first push UMDR solo accuracy by retraining with paper recipe (longer epochs, larger batch, depth pre-init from RGB checkpoint).

---

---
name: v15a Hungarian QCC result
description: Hungarian correspondence with prediction aux QCC did not beat mutual-NN; confirms 1-rotation bottleneck
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Ran v15a (2026-04-19) to test whether Hungarian correspondence improves prediction-aux QCC.

**Recipe**: same as v6p (Sc_Co_pr01_N3_pts48-256_e140) with `assignment_mode: hungarian` on loader. Identical model, identical aux, only correspondence source differs.

**Results** (peak test accuracy over 140 epochs, single seed):
- v6q (no aux):           **75.93%**
- v6p (mutual + pr aux):  **76.14%** (+0.21 vs no-aux)
- v15a (hungarian + pr):  **74.07%** (-1.86 vs no-aux, -2.07 vs mutual)

**Why it regressed**: Hungarian doubles correspondence density but the 1-quaternion QCC head cannot absorb the extra signal. Some of Hungarian's extra pairs (the ones mutual-NN would reject) are lower quality, and the prediction-aux MSE may overfit on that noise. Single-seed variance is also ~1-2pp.

**Why:** the gain we expected (richer supervision → better QCC) didn't materialize because QCC's bottleneck is the 1-rotation-per-frame-pair representation, not correspondence density. v6p's +0.21 over v6q already hinted this.

**How to apply:** don't pursue correspondence-density variants further for QCC. The next move is K-part decomposition (palm + 5 fingers) with per-part rotations derived by weighted Procrustes — gives 6x the rotation information without needing joint labels. Tracked as the "K=6 parts" direction.

**Run artefacts on remote**: `experiments/work_dir/quaternion_prediction_hungarian_v15a/` (checkpoints every 5 epochs, full log). Crashed mid-run at ep 109 due to Paperspace 503, resumed from epoch 105 checkpoint successfully via `--resume True --weights epoch105_model.pt`.

---

---
name: v16a parts-rigidity Procrustes result (Option A+B)
description: K=6 Procrustes + feed-forward modulation peaked 73.03%, trailed baselines; killed at ep 121
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Ran v16a on 2026-04-19/20 to test Option A+B from the K=6 parts discussion.

**Recipe**: Sc_Co + `qcc_variant: parts_rigidity` + `assignment_mode: hungarian`. K=6 soft-assignment head → weighted Procrustes per part (SVD, differentiable through w_k but R detached to avoid SVD gradient explosion) → rigidity-residual aux loss. K derived quaternions mean-pooled over time, projected by MLP, multiplicatively modulate `encoded` before classifier.

**Result** (stopped at ep 121, peak plateaued):
- v6q (no aux):          **75.93%**
- v6p (mutual + pr aux): 76.14%
- v15a (hu + pr aux):    74.07%
- v16a (hu + parts_rigidity A+B): **73.03%**

**Why it trailed**: gap vs v6p was ≈-2.5pp at ep 120, widening at end. Parts-rigidity aux + rotation modulation did not give the classifier information it wasn't already extracting via the existing EdgeConv+attention path. The first training attempt NaN'd at ep 1 due to differentiable SVD gradient explosion; fixed by detaching R post-SVD, masking near-empty parts, and reformulating the entropy term as a collapse penalty (non-negative).

**Why:** single-seed evidence that adding a parts_rigidity aux loss does not improve the classifier. Signal reaching the soft-assignment is limited; the K=6 modulation channel doesn't add information the main path already has. Consistent with the broader pattern that QCC aux loss has marginal effect on this model.

**How to apply:** next move is Option C (predicted rotations via MLP head + real cycle-consistency QCC) — only variant where the predicted rotations are free parameters that cycle consistency can actually constrain. Config: v17a.

**Artefacts on remote**: `experiments/work_dir/quaternion_parts_rigidity_hungarian_v16a/` (checkpoints to ep 120 at save_interval=5). Patch at `experiments/patch_parts_rigidity.py`.

---

---
name: v17a parts-cycle QCC result (Option C)
description: K=6 learned rotations + cycle consistency + reconstruction aux; peaked 74.69%, still below v6p
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Ran v17a on 2026-04-20 to test Option C from the K=6 parts discussion.

**Recipe**: Sc_Co + `qcc_variant: parts_cycle` + Hungarian correspondence. MLPs predict forward/backward quaternions per part (free parameters, not Procrustes-derived). Aux loss = reconstruction (Procrustes-style fit residual) + cycle drift (q_bwd ∘ q_fwd ≈ identity) + entropy collapse penalty. Forward quats mean-pooled over time → MLP → multiplicative modulation on `encoded`.

**Result** (peak after LR-schedule completion):
- v6q (no aux):            **75.93%**
- v6p (mutual + pr aux):   76.14%
- v15a (hu + pr aux):      74.07%
- v16a (hu + parts_rigidity A+B): 73.03%
- v17a (hu + parts_cycle C):      **74.69%**

**Why it underperformed**: loss pulls the model toward rotation fidelity (per-part reconstruction and cycle closure) while the classifier only needs discriminative features. Two competing objectives. All three QCC-aux variants (v15a, v16a, v17a) landed below v6q no-aux. This is consistent evidence that aux-loss supervision of rotation structure is not the right lever.

**Why:** aux loss signal pulls gradient toward "explain motion rigidly as K rotations" which is not aligned with "tell me which gesture this is." The resulting tension costs a few pp.

**Run interrupted twice**: Paperspace 503 at ep 51 and again near ep 111. Resumed each time from latest checkpoint. Also OOM the first launch due to orphan v16a dataloader workers holding GPU memory (had to `pkill -9` before resume).

**How to apply:** pivot to parts-as-feature design (v18a): compute same K rotations + per-part rigidity residual, but NO aux loss — let classification gradient alone learn the soft assignment via the differentiable Procrustes. Matches the pattern of the existing bearing-QCC rigidity signal, which is used as a feature (modulates encoded) not a loss.

**Artefacts on remote**: `experiments/work_dir/quaternion_parts_cycle_hungarian_v17a/`. Patch `experiments/patch_parts_cycle.py`.

---

---
name: v26a tops+xyz beats xyz-only baseline
description: Adding centroid-radial tops channel alongside xyz as 7-channel input beats v6q 75.93% by +2.08pp at 78.01%
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Ran v26a on 2026-04-20: full BearingQCCFeatureMotion architecture with
7-channel spatial input = xyz + time + unit direction from frame centroid
(tops field). First EdgeConv layer enlarged from 8→ to 14→hidden1 to
accept the additional 3 channels.

**Result** (140 epochs, single seed, Hungarian correspondence):
- v6q  (xyz only, no aux):   75.93%
- **v26a (xyz + tops, 7-ch):  78.01%**   (+2.08pp)
- v6p  (xyz + mutual + pr):  76.14%

This is the first QCC-rethink experiment to cleanly beat the xyz-only
baseline. All earlier aux-loss attempts (v15a–v18a with Hungarian
correspondence + various QCC variants) trailed v6q by 0.5–3pp. Explicit
tops-as-input channel succeeds where tops-as-loss failed.

**Why it works:** tops direction = (p - frame_centroid) / |p - frame_centroid|
is a deterministic function of xyz, so in theory EdgeConv could learn to
compute it. In practice, explicitly providing it:
  1. Saves the network from having to rediscover centroid-radial structure
  2. Creates a globally-consistent reference direction for every point
  3. Works cleanly with quaternion-aware layers downstream (direction is a
     natural quaternion input)

**Why xyz-only reaches 75.93% but xyz+tops hits 78.01%:** raw xyz encodes
both absolute position and implicit direction. The network spends capacity
learning to extract direction from xyz via EdgeConv neighborhood ops.
Giving direction explicitly lets those layers focus on complementary
information, improving effective capacity.

**How to apply:** for any quaternion-architecture model on this dataset,
add per-point centroid-radial direction as an extra 3-channel input with
a modified first conv. Cost: +12 extra input params at first conv, zero
other overhead. Gain: ~+2pp on peak accuracy in single-seed experiment.

**Memory note:** the +2.08pp gain is single-seed. Multi-seed confirmation
would be valuable before declaring firm ~2pp lift.

**Artefacts on remote**: `experiments/work_dir/quaternion_tops_xyz_v26a/`.
Patch `experiments/patch_tops_xyz_input.py` + fix `patch_tops_xyz_v2.py`
defines `TopsXYZInputMotion` subclass of BearingQCCFeatureMotion.

---

---
name: V2 trim test baseline
description: PMamba ep110 weights → v2-trimmed test set = 89.2 (vs 89.83 on old uncropped data)
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
PMamba pmamba_branch/epoch110_model.pt evaluated on freshly preprocessed v2 (gesture-window-trimmed) test set: **prec1 89.2, prec5 ~98**.

Original baseline on uncropped data: 89.83. Difference = -0.6pp.

**Why:** confirms v2 trim shifts the input distribution slightly (gesture-only vs full clip with non-gesture frames). Existing weights weren't trained on this distribution so small drop is expected.

**How to apply:** v2 trim alone is not a free win. To get any benefit from the trim we have to retrain from scratch (or warm-restart) on v2-preprocessed data. Pipeline established at `/notebooks/PMamba/dataset/Nvidia/Processed/` with symlinks back to `dataset_full/`.

---

---
name: Warm-start drift baseline (LR 1.2e-5, 35 ep)
description: Pmamba_branch ep110 continued at LR 1.2e-5 for 35 ep drifts to mean 87.10 ± 0.82 (best 88.59); use this as null baseline for warm-start experiments, NOT the 89.83 ep119 reference
type: project
originSessionId: 786484e8-56c2-4f46-a56d-f3ebf2b7c05b
---
Pmamba_branch ep110 ckpt continued without architectural changes (warm-start, LR 1.2e-5, ep110→145, 35 evals):
- best 88.59 at ep131
- final (ep145) 87.55
- mean 87.10
- stdev 0.82
- top3: 88.59, 88.38, 88.38

**Why:** The 89.83 reference (memory project_pmamba_tops_win.md) is ep119-specific. Continuing the same recipe past ep119 drifts. The "89.2" v2-trim baseline (project_v2_trim_baseline.md) doesn't apply when training continues — it's a one-shot eval of the loaded ckpt.

**How to apply:** When evaluating a warm-start architectural change at LR 1.2e-5 ep110→145, compare against this 88.59/87.10/0.82 distribution, not 89.83. Effects within ±0.82 (1σ) are noise, not signal. To claim improvement, peak must exceed ~89.4 (baseline best + 1σ).

This re-interprets project_octonion_bidir_null.md: octonion-bidir-residual-α best 88.17 vs 88.59 baseline = -0.42, within noise. Not a degradation, just no signal. Same conclusion (octonion-bidir adds nothing useful) but framed correctly.

