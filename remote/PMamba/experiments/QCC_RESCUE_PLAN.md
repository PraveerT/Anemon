# QCC rescue plan — 5 roles where DQCC can boost accuracy

Baseline: pmamba_base 89.83% solo. Best so far: v1 tops 90.66 / 91.49 calib fusion.
ChatGPT reality check: as aux loss, QCC is dead (tested ×5). These 5 roles are
alternative uses that could still boost accuracy.

## Approach 1 — Rigid-subtraction preprocessing ⭐ most promising

**Idea:** factor out rigid hand motion via Kabsch, feed classifier the
articulation residual. Gesture class is about articulation, not global rotation.

**Implementation:**
- For each correspondence-aligned pair (t, t+1), compute Kabsch (R, t)
- Rigid-predicted position at t+1: `rigid_pred_i = R · p_i(t) + t`
- Residual: `res_i = p_i(t+1) - rigid_pred_i` (3D per point per frame)
- Input to PMamba: `[x, y, z, res_x, res_y, res_z, t]` = 7-ch (analog of v1 tops structure)
- Residual for frame 0 set to zero

**Hypothesis:** classifier sees "articulation motion minus rigid motion" —
orthogonal to what xyz alone provides. Expected: +0.5-1pp solo; strong
fusion complement because residual errors differ from xyz errors.

**Config:** pmamba_rigidres.yaml; patch_pmamba_rigidres.py. Same LR recipe as v1 tops.

## Approach 2 — Canonical alignment preprocessing

**Idea:** all frames aligned to frame-0's coordinate system via inverse
cumulative Kabsch. Classifier never sees global rotation/translation.

**Implementation:**
- Cumulative Kabsch: R_t, t_t = chain of Kabsch(t-1, t) compositions from frame 0
- Aligned position: `p_aligned_i(t) = R_t⁻¹ · (p_i(t) - t_t)`
- Input to PMamba: aligned_xyz + t = 4-ch, standard PMamba
- Need to handle cumulative quaternion composition (QCC-style cycle)

**Hypothesis:** removes orientation-variance nuisance. Expected: +0.3-0.5pp.

## Approach 3 — Cycle-error confidence weighting

**Idea:** cycle inconsistency = noisy/ambiguous sample. Down-weight such samples
in fusion.

**Implementation (post-hoc):**
- Compute per-sample cycle error: `||DQ(0,1) ⊙ ... ⊙ DQ(30,31) - DQ_obs(0,31)||`
- Weight fusion prediction by `1 / (1 + cycle_err)`
- No retraining — apply to existing v1 tops + pmamba_base logits

**Expected: +0.1-0.5pp fusion.**

## Approach 4 — Test-time augmentation via cycle-consistent transforms

**Idea:** generate N small transforms that satisfy cycle consistency, apply
each to test sample, average predictions.

**Implementation:**
- For each test sample, generate N random DQ perturbations (small magnitude)
- Apply each to xyz, run through trained v1 tops model, get N logits
- Average, argmax

**Expected: +0.2-0.5pp solo. Works for any model.**

## Approach 5 — DQ-cycle-consistent data augmentation

**Idea:** training-time augmentation using cycle-consistent synthetic
transforms (not random rotations).

**Implementation:**
- At training, sample from observed DQ distribution per class
- Apply to input clip, keep label same
- Train pmamba_base + DQ aug for 40 epochs

**Expected: +0.3-1pp solo.**

---

## Execution order (self-monitored)

1. Approach 1 first (novel + biggest expected gain)
2. Approach 3 in parallel (post-hoc, no GPU conflict)
3. Approach 2 after 1
4. Approach 4 after 2 (needs trained model)
5. Approach 5 last (longest runtime)

Each experiment kills the previous before launching. Self-scheduled wake-ups
for checking results.
