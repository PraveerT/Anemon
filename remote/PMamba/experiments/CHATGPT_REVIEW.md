# ChatGPT Second Opinion — Session Review

## Headline verdict
You're not stuck because of architecture — you're trying to squeeze signal out of
the wrong invariances.

## What the results say

Cleanest hierarchy in the data:

- Rotation-only → ~14–17%
- SE(3) (q + t) → ~27–28%
- Centroid trajectory → ~32%
- Centroid + velocity + accel → ~37.5%

→ `classification signal ∝ translational temporal structure`. Not geometry, not
rigidity, not consistency.

So:
- QCC/DQCC → shaping toward rotation consistency ❌
- Rigidity aux → shaping toward motion smoothness/local agreement ❌
- Contrastive → shaping toward similarity structure already learned ❌

All low mutual information with labels → must fail or plateau.

## Why QCC will never give you gains here

Even if perfectly optimized, it pushes features toward SE(3) consistency while
your dataset's discriminative axis is (1) direction of motion, (2) temporal
evolution, (3) which parts move when — not constrained by cycle consistency.

Best-case: QCC becomes a regularizer → +0.1 at best.
Worst-case (observed): it fights CE, pulls features away from class-separable
directions.

## Why "tops" worked

v1 tops success is not random. It injects local radial structure relative to
centroid, implicitly giving:
- motion direction relative to body
- articulation hints (fingers vs palm)
- weak normalization of global translation

i.e. it injects a coordinate system aligned with motion. That's why it gave
solo gain ✅ orthogonal errors ✅ fusion boost ✅.

## High-probability next wins

### 1. Add velocity as an INPUT channel (not aux)
You already proved velocity is strong (via centroid derivatives).
Input: `[x, y, z, vx, vy, vz, t]` or even `[x, y, z, tops_xyz, vx, vy, vz, t]`.
Converts temporal signal → instant spatial feature. Mamba doesn't need to infer
derivatives anymore. This is exactly why tops worked.
→ likely +0.5–1.0% solo.

### 2. Build a separate centroid-trajectory model (for fusion)
Don't use it as aux. Use it as a branch.
Input: `32 × [cx, cy, cz, vx, vy, vz, ax, ay, az]`. Tiny 1D-conv or small
transformer. Low solo, high complementarity → ideal for fusion.

### 3. Train multiple input-perturbed PMamba branches
Orthogonal views, not better losses:
- A. tops (best)
- B. velocity-augmented
- C. tops + velocity
- D. normals (ONLY as input to PMamba, not quaternion branch)

Fuse all 4. Ceiling: single branch fusion ~91–92, multi-branch 93%+ realistic.

### 4. Fix fusion properly — learned per-sample gating
Current α-blend + temperature is primitive. Train a gating model:
- Input: `[logits_pmamba, logits_tops, logits_velocity, entropy, max_prob]`
- Output: softmax weights per model
- Learns "which model to trust per sample"
- Can close 91.49 → ~92.5+

### 5. Stop investing in aux losses (seriously)
If standalone target < ~30% → aux useless.
If aux gradient flat → harmful.
If aux fights invariance → harmful.
No hidden win left in QCC / DQCC / rigidity / contrastive.

## Deeper insight
For dynamic gestures, "what moves where over time" dominates "how it rotates".
This is why SE(3)-heavy methods fail, temporal derivatives win, input
augmentation beats equivariance.

---

# On: "but I need QCC for my research, is it dead dead?"

Short answer: no, it's not "dead dead" — but it's dead in its current role (as
generic aux loss to boost classification).

## Reframing
Your experiments proved something deep:
**Cycle consistency is not discriminative — it is structural.**

It won't add class signal, but it can shape representations in useful ways.
You were asking it to do the wrong job.

## Why QCC failed here
1. Optimizes wrong invariance — SE(3) consistency, not direction-of-motion
   signal.
2. Competes with CE — CE wants class-separating features, QCC wants
   geometry-consistent. If geometry ≠ class signal → gradients conflict. You
   saw: aux ↓ nicely, accuracy ↔ or ↓.
3. Your signal-strength test predicted this: QCC target ≈ 28% max, centroid
   dynamics ≈ 37.5%. QCC is literally a weaker supervision target.

## How to make QCC actually work

### Option 1 — QCC as a separate branch (cleanest)
Instead of "QCC helps PMamba classify", do "QCC produces a different view of
motion":
- Branch A: PMamba (xyz + tops) → strong classifier
- Branch B: QCC/DQCC encoder → predicts motion-consistent features, classify
  from it (~80–85%)
- Fuse with PMamba

Even weak classifiers can be highly complementary. Overlap numbers already
prove this principle.

### Option 2 — QCC as pretraining objective
Flip the order:
- Stage 1: train encoder with QCC/DQCC only
- Stage 2: fine-tune with CE

QCC no longer competes with CE — it just initializes a geometry-aware latent
space. This is where QCC usually shines.

### Option 3 — Apply QCC locally, not globally
Rotation signal is weak globally, but not necessarily locally. Try:
- High-rigidity regions only
- Short temporal windows (t → t+1 → t+2)

Turns QCC into local motion regularization, not global constraint.

### Option 4 — Predict QCC features, don't enforce them
Instead of loss `|| q_pred - q_obs || + cycle_loss`, have the model output DQ
features and feed them into classifier. Treat QCC quantities as features, not
constraints. Aligns with your biggest success: input augmentation works, loss
augmentation doesn't.

### Option 5 — QCC for uncertainty / confidence
Cycle inconsistency is useful for detecting noisy/ambiguous frames. Use
`cycle_error → confidence score` to weight logits in fusion. Subtle but
powerful.

## Reality check

If goal is "maximize accuracy" → QCC is not the best lever on this dataset.
If goal is "meaningful research with QCC" → you're sitting on something strong:

You have demonstrated:
- QCC learns well (loss drops)
- but doesn't improve classification
- because geometric consistency ≠ discriminative signal

**That's a publishable negative result** if framed correctly.

## Honest conclusion
QCC is not dead — but it's not a classification signal, it's a structural
prior. Your experiments beautifully demonstrate that.

## What I'd do in your position
Keep QCC central, but pivot:
- Build QCC-based branch
- Fuse with tops model
- Report solo (even if lower), oracle, fusion gain

If fusion improves, you've proven QCC captures complementary structure — strong
result.
