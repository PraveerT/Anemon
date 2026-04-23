# Quaternion-cycle gap exploration plan

Goal: find a QCC variant that **improves solo accuracy** over the best
non-cycle baseline on the quaternion branch.

Baselines to beat:
- quaternion_branch_v2_velpolar: **79.67%** solo (last verified best-in-family)
- BearingQCCFeatureMotion w/ xyz+time: around 75–76% (v6p/v6q)

Every previous cycle attempt collapsed (identity attractor). Mittal 2020 names
this exactly: "the cycle consistency loss has a degenerate solution: the zero
flow / identity, will produce 0 loss." Their fix is **anchoring** — observable
targets or anchored reverse flow. We apply that idea to quaternion cycles.

## Candidate experiments (ranked, run sequentially)

### Q1. Anchored quaternion-pair supervision + transitivity ⭐ start here
- Predict `q_pair(t, t+k)` from feature pair via small head.
- Anchor: observed Kabsch quaternion from correspondence-aligned sampled pts.
- Losses:
  - `L_anchor = 1 − |q_pred · q_obs|²`    (sign-ambiguous cos² direct supervision)
  - `L_trans = 1 − |q_pred(t, t+2) · (q_pred(t+1, t+2) · q_pred(t, t+1))*|²`
- No collapse because `L_anchor` target is observable and non-trivial.
- `L = CE + λ_a · L_anchor + λ_t · L_trans`
- Implementation: new class `AnchoredQCCBearingMotion` extending `BearingQCCFeatureMotion`; internal qcc_weight=0.
- Why start here: simplest, directly tests whether observable-target cycle actually teaches anything useful.

### Q2. Per-point quaternion field with anchored cycle
- Predict a unit quaternion per point per frame-pair (dense rotation field).
- Anchor per-point: shortest-arc from `(p_i − c_t)` to `(p_i' − c_{t+1})`.
- Cycle loss: per-point forward · backward = identity; supervised via anchor.
- Add rigidity regularizer: neighbouring points' quaternions should agree (ARAP-like).
- This is the strongest real gap from the literature map.

### Q3. Dual-quaternion SE(3) cycle
- DQ encodes rotation + translation.
- Predict DQ per pair; cycle on DQ space.
- Watch numerical stability; add unit-norm + orthogonality projection.

### Q4. Lie-algebra tangent-space cycle loss (lowest priority)
- Express cycle error in so(3) tangent via `log(q_cycle)` (vector in ℝ³).
- Backprop through `log` more stable than raw quaternion distance near identity.
- Likely incremental over Q1 at best; de-prioritised.

## Success criterion

Any variant whose solo test accuracy beats 79.67 on the Nvidia test set, while
using correspondence data from `NvidiaQuaternionQCCParityLoader`. If Q1 fails,
try Q2. If Q2 fails, re-examine assumptions (maybe we need a different
backbone, not a different loss).

## Negative results we already have (won't repeat)

- Cycle-to-identity on any head (v3 global, v4 part, v5 tops) → collapse to q=1.
- Global rigidity residual as input channel (option 1) → hurt solo.
- Rigidity as sigmoid-gate on readout (option 2) → gate learned to be fully open.
- Tops-anchored q target (v5) → q_obs ≈ identity (frame-to-frame motion too
  small), same trivial attractor.

## Key technical ingredient for all experiments

**Observable anchor.** For every quaternion the network predicts, produce an
observable counterpart from the actual point data via Kabsch (or dual-Kabsch
for SE(3)). Direct supervision against that anchor is what removes the
collapse mode. Cycle constraints then ride on top as regularisation.
