# ChatGPT v2 review of QCC rescue plan

## Key corrections to original plan

### 1) Rigid-motion subtraction — critical nuance
Naive `X_res(t) = R_t^T (X_t - t_t)` risks removing translation, which IS
class-discriminative (swipe left vs right).

**Correct version (Option B):**
- `c_t = centroid(X_t)`
- `X_centered = X_t - c_t`
- `X_aligned = R_t^T · X_centered`
- Feed `[X_aligned, c_t]` → articulation (clean) + trajectory (explicit)

Aligns with C_frame result (trajectory is strongest signal).

**QCC's real role here:** temporal smoother for `R_t, t_t` to avoid
frame-to-frame jitter in the canonicalization transform.

**Expected:** solo +0.3-0.8, fusion possibly more.

### My current run's status on this concern
My implementation uses **7-ch [xyz, res_xyz, t]** where:
- `res = p(t+1) - (R·p(t) + t)` — per-pair rigid residual
- **xyz preserved as first 3 channels** → trajectory info NOT lost

Equivalent to ChatGPT's "dual input" [X_raw, X_aligned] idea, just with pair-
residual instead of frame-aligned points. Both preserve trajectory via explicit
channel.

### Possible upgrade for v2 of approach 1 if current fails
Frame-level Option B: input `[xyz_aligned, centroid_xyz, t]` = 7-ch:
- xyz_aligned = R_from_frame_0^T · (xyz - centroid)
- centroid = per-frame centroid, repeated per point
- More explicit separation of articulation vs trajectory

## Reprioritization

**Tier 1 (do first):**
1. Velocity-input PMamba (not yet run)
2. Rigid-aligned input (current run; Option B variant if needed)
3. Learned gating fusion (not yet run)

**Tier 2 (fast gains):**
4. Cycle-error confidence weighting
5. Test-time augmentation

**Tier 3 (research framing):**
6. QCC branch (for paper)
7. DQ-based augmentation

## Key reframing
> QCC is not dead — it's now: **a tool for removing nuisance structure,
> not predicting labels**

Rigid-subtraction = QCC as "isolator of articulation" — cleanest, most novel
use of QCC in our pipeline.

## Dual-input suggestion (important)
Don't replace raw with aligned. Train on both:
- `[X_raw, X_aligned]` or `[X_tops, X_aligned]`
- Raw keeps global motion cues
- Aligned isolates articulation

My current 7-ch [xyz, res, t] is a form of this, but could add aligned-frame
version as a v2 if residual formulation isn't enough.
