# Rigidity signal — future directions

Per-point position residual `r_i = p_i' − (q_best·(p_i − c) + c')` is an
observable, real-valued, non-learned signal of per-point non-rigidity. Computed
from correspondence-aligned point clouds using Kabsch.

Option 1 (extra input channel) is implemented in `RigidityInputBearingQCCFeatureMotion`.
The following are reserved for later iterations.

---

## Option 2 — Attention weighting on readout

Weight per-point features by a function of rigidity before pooling.

Implementation sketch
```python
# encoded: (B, C, N)  where N = F * P
# rigidity: (B, F, P)  -> flatten to (B, N)
rigidity_flat = rigidity.reshape(B, -1)                        # (B, N)
w = torch.sigmoid((rigidity_flat - tau) * alpha)               # soft gate
# Use for pooling:
pooled_attn = (encoded * w.unsqueeze(1)).sum(-1) / (w.sum(-1, keepdim=True) + eps)
```

Choices
- `tau` = soft threshold (mean of rigidity over batch, or a learned constant)
- `alpha` = sharpness (hyperparam)
- Direction: `sigmoid((r - tau) * alpha)` weights articulated points up.
  Inverse `sigmoid((tau - r) * alpha)` weights rigid-body points up.
- Add as *extra* pooled feature alongside current max + attention readouts so
  the model can choose.

Why it should help
- Explicit inductive bias: articulated points (fingers) carry gesture info.
- No new loss term, no collapse risk.

Risks
- Hyperparameter-sensitive; may need tau to adapt during training.

---

## Option 3 — Rigidity-disentanglement loss

Force latent representations of rigid points to be rotation-invariant (they
move with the global q_best), and let non-rigid points encode the deviation.

Loss sketch
```python
# Given per-point feature f_i and rigidity scalar r_i:
# For low-r_i points, feature of p_i and feature of q_best·p_i (augmented
# rigid-rotated copy) should match.
# For high-r_i points, skip that constraint.

augmented_points = rotate_all_with_q_best_random_rotation(sampled)
f_orig     = backbone(sampled)                                 # (B, C, N)
f_aug      = backbone(augmented_points)
rigid_mask = (rigidity_flat < threshold).float().unsqueeze(1)  # (B, 1, N)
loss_diseng = ((f_orig - f_aug) ** 2 * rigid_mask).sum() / (rigid_mask.sum() + eps)
```

Why it should help
- Target (rigidity-masked feature similarity) is observable, not identity-
  collapsible like prior QCC attempts.
- Forces the backbone to separate rigid-body motion from articulation.

Risks
- Heavier: needs a second forward pass on augmented input.
- Threshold sensitivity.

---

## Option 4 — Correspondence quality filter

High residual at a point often indicates a *bad* Hungarian correspondence
(matched to the wrong physical point) rather than real articulation. Weight the
correspondence-based QCC losses inversely by rigidity.

Implementation sketch
```python
# bearing_qcc uses correspondence-matched pairs. Downweight bad ones.
corr_quality = 1.0 - torch.tanh(rigidity_flat / sigma)         # in (0, 1]
# In loss:
loss_qcc = (pair_loss * corr_quality).sum() / corr_quality.sum()
```

Why it should help
- Cleaner QCC gradient signal: errors from real articulation still contribute,
  but correspondence noise gets muted.
- Could rescue the past QCC experiments that saw identity collapse under noisy
  correspondence.

Risks
- If `sigma` is too small, every non-zero residual gets downweighted, defeating
  the point. If too large, effectively disabled.
- Training stability: correspondence quality depends on the same data that the
  model is being trained to classify — entangles loss scales.

---

## Composition notes

- Options 1 + 2 are orthogonal (input feature + pooling weight). Natural first
  pair to stack after confirming Option 1 alone helps.
- Option 3 requires care with augmentation pipelines; avoid double-rotating.
- Option 4 combines with Option 1 (channel) cleanly — different usage of the
  same rigidity tensor.
