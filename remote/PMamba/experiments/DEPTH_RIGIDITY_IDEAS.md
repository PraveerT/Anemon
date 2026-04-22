# Loss-function ideas using rigidity residuals for depth_v2

Baseline = **depth_v2** (depth+tops CNN-LSTM, no rigidity).
Target = **beat 93.36% oracle / +0.83pp fusion vs PMamba @ e110**.

Rigidity signal = per-frame summary of per-point Kabsch residuals, as defined
in `RIGIDITY_FORMULA.md`. Currently K=6 scalars per frame
(mean, std, max, p75, p90, p95). Precomputed offline in
`preprocess_rigidity.py`, stored as `{stem}_rigidity.npy`.

What already failed:
- v6: concat rigidity stats directly into LSTM input → disrupted temporal
  learning. Solo 58.92% (−7.5pp vs v2), fusion gain 0.00 (−0.83pp).

What we try next — all keep v2's inputs clean, use rigidity only as a
*supervisory* or *weighting* signal.

---

## v7 — Auxiliary prediction head (highest priority)

Add a small MLP on per-frame CNN features to PREDICT the rigidity stats from
the features. Backbone and classifier head unchanged.

```
cnn_feat_t ∈ ℝ^D    (D=256 per frame)
aux_head   : ℝ^D → ℝ^K
pred_rig_t = aux_head(cnn_feat_t)

L = CE(logits, label) + λ · MSE(pred_rig_t, rigidity_t)
```

- Target is observable → no collapse risk.
- Forces features to encode articulation structure.
- Primary pipeline undisturbed; gradient only shapes CNN features.
- Start λ = 0.1, head = 2-layer MLP (D → 64 → K).

## v8 — Binary articulation classifier

Same architecture as v7 but target is binarised: is this frame above median
rigidity within the clip? Softer supervision, more robust to noise in stats.

```
target_t = rigidity_mean_t > median_over_clip(rigidity_mean)      # ∈ {0, 1}
pred_t   = sigmoid(aux_head(cnn_feat_t))
L = CE + λ · BCE(pred_t, target_t)
```

## v9 — Per-clip loss reweighting (zero-arch-change)

No model change. Weight cross-entropy by a scalar derived from the clip's
rigidity dynamics:

```
w_i = 1 + β · std_over_frames(rigidity_mean_in_clip_i)

L = Σᵢ w_i · CE(logits_i, label_i) / Σᵢ w_i
```

Clips with more articulation variability (interesting gestures) get more
gradient weight.

## v10 — Temporal contrastive with rigidity-weighted agreement

Two augmentations of the same clip should produce matching per-frame features,
weighted by rigidity (articulated frames matter more):

```
feat_t^(1), feat_t^(2) = cnn(aug1(clip_t)), cnn(aug2(clip_t))
w_t = rigidity_mean_t (normalised)

L_contrast = Σ_t w_t · ‖feat_t^(1) − feat_t^(2)‖²
L = CE(aug1) + CE(aug2) + λ · L_contrast
```

Doubles forward cost. Potentially strong but more implementation.

## v11 — Rigidity-gated temporal pooling (architectural, not loss)

Replace or supplement the LSTM's temporal mean/max pool with attention
weighted by rigidity. No new loss term.

```
α_t = softmax(β · rigidity_mean_t)
pooled_rig_attn = Σ_t α_t · lstm_feat_t
feat = concat(pooled_mean, pooled_max, pooled_rig_attn)
```

Risk: discards info from rigid frames that also carry gesture cues.

---

## Priorities

1. **v7 first** — cleanest theory, low implementation cost.
2. **v9 second** — trivially cheap sanity check.
3. **v8** — try if v7 is mixed and the noise hypothesis is right.
4. **v10** if v7/v8 show weak signal and we need a stronger inductive bias.
5. **v11** last — architectural risk.

---

## Notes

- Every variant trains from scratch, 140 epochs, same hparams as v2.
- Oracle vs PMamba@e110 on best_model. Table in `DEPTH_RIGIDITY_RESULTS.md`.
- If none beats v2 (+0.83pp fusion), the rigidity signal computed this way
  carries no class-discriminative information beyond what the backbone already
  extracts from the image. Would then move to other axes (e.g., cleaner
  rigidity via weighted Kabsch on correspondence-confidence).
