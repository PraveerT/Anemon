# Research Brief: Skew-Symmetric Temporal Cross-Covariance Pooling (Skew-TCC)
# One publishable non-quaternion layer for fine-grained gesture recognition
# (synthesized 2026-05-29 from a 16-agent literature workflow: 81 candidates → 6 verified → 1 recommended)

## 0. Problem recap (design driver)
CN-XXL on NVGesture (25-class, 1050/482 subject-disjoint, point clouds (T=32,N~172,4ch)) = 91.29% test (440/482, 42 errors).
Errors = (i) systematic fine-grained confusions (4→17 x7, 19→10 x6, 15→14 x3; two pairs = 13/42), (ii) DIRECTIONAL: rotation/scale/jitter TTA fixes ~0, but MIRROR collapses 440→121 and TIME-REVERSAL collapses 440→363. The first-order max-pool head discards relational + directional co-occurrence. Symmetric covariance/bilinear pooling recovers relational structure but is ANTISYMMETRIC-BLIND — discards exactly the chirality/order signal that IS the problem.

## 1. Guiding theorems
- **P1 — The dominant error axis is antisymmetric, not invariant.** Mirror (440→121) and time-reversal (440→363) destroy accuracy while rotation/scale/jitter don't → the missing signal flips sign under reflection/time-reversal = a SKEW object.
- **P2 — Symmetric second-order pooling is provably insufficient here.** Gram/covariance C=Cᵀ is invariant to u↔v swap and feature-axis reflection → cannot separate push/pull or left/right. The whole MPN-COV/iSQRT-COV/TCP/bilinear family is the WRONG symmetry → they are CONTROLS, not the contribution.
- **P3 — Direction must be a learned sign-bearing scalar read-off, not a group-equivariant feature.** Vector-Neuron/SO(3) heads act on spatial coords, discard direction after invariant readout, and have zero temporal-order mechanism. The descriptor must be antisymmetric AND reduce to plain reals a vanilla linear classifier reads.
- **P4 — On 1050 samples the addition must be low-rank, low-parameter.** Second-order needs ≥10-50k samples for stable gains; the skew sub-part is higher-variance. Use rank-r (r≈8-16) → descriptor O(r²), new params ≤1-2% of model.
- **P5 — Any geometric-sounding addition must beat a param-matched real control or it is decorative.** (The 91.08 quat-head: ~71% lift from structural params, output scale→0 at inference.) Quaternions/PHM excluded.
- **P6 — Prefer one solo, end-to-end addition over fusion.** No clean in-distribution validation set → any fusion tuning is dishonest. A single end-to-end layer never touches the calibration wall.
- **P7 — Temporal direction and spatial chirality are orthogonal failure modes; one descriptor should address both.** mirror (spatial) and time-reversal (temporal) collapse independently; the object must flip sign under BOTH a feature-channel swap AND a temporal-lag swap.

## 2. THE recommended addition — Skew-TCC
Plug at the pooling head, augmenting amax→mean/max/std on stage5 features X ∈ R^{B×D×T×N} (D=1024,T=32,N~172):
1. Spatial reduce (keep): z_t = max_n X[:,:,t,:] → Z ∈ R^{B×T×D}.
2. Two low-rank projectors (only new weights): U=ZW_u, V=ZW_v, W_u,W_v ∈ R^{D×r}, r≈8-12 → U,V ∈ R^{B×T×r}.
3. Lagged cross-Gram, lags δ∈{1,2}, boundary-masked: C_δ = Σ_{t=1}^{T-δ} u_t v_{t+δ}ᵀ ∈ R^{r×r}.
4. **Antisymmetric part (the contribution): A_δ = (C_δ − C_δᵀ)/2.**
5. Read strictly-lower triangle: a_δ = tril(A_δ,−1).flatten() ∈ R^{r(r-1)/2}.
6. Concat [a_1,a_2] onto existing first-order stats → existing compress Linear → global_bn → classifier. No downstream change.
Params: 2·D·r ≈ 25k + ~132 compress inputs (~0.x% of model).

### Why it captures DIRECTION (the gap symmetric covariance misses)
- Time-reversal t→T−t: C_δ → C_{−δ} ≈ −C_δᵀ, so skew A_δ FLIPS SIGN → separates push/pull (the 440→363 axis). Symmetric Gram is invariant → cannot.
- Mirror/chirality: reflection sign-flips u components → u'⊗v' carries opposite skew → A_δ flips (the 440→121 axis). Same object handles both (P7).
- Lower-triangle = plain reals → vanilla linear classifier reads them (P3).

### Novelty (honest): novel APPLICATION, not novel theory.
Sym/skew split is trivial math. But: symmetric covariance pooling (MPN-COV ICCV2017, iSQRT-COV CVPR2018, TCP NeurIPS2021 arXiv:2110.14381, compact/low-rank bilinear CVPR2016/2017) is antisymmetric-blind BY CONSTRUCTION; isolating the SKEW LAGGED CROSS-COVARIANCE to recover gesture chirality/order has NO precedent in action/gesture DL. Order-aware methods that exist are temporal-only (Rank Pooling arXiv:1512.01848, TDN arXiv:2012.10071) or directed-attention backbones (DirecFormer CVPR2022 arXiv:2203.10233) — none a single skew-bilinear pooling layer, none addresses spatial chirality. Contribution: "the antisymmetric component of a lagged feature cross-covariance is the minimal pooling object sign-sensitive in both time and feature axes."

### Mandatory controls (P5)
1. Param-matched SYMMETRIC control: read sym(C_δ)=(C_δ+C_δᵀ)/2, same r/params/length. If Skew≈Sym → decorative. [NOTE: the bilinear run already training IS essentially this control.]
2. Random/frozen-projector control: freeze W_u,W_v at init → isolates skew content vs structural presence (the quat-head exposer).
3. Zero-out at inference: a_δ→0 at test; if acc holds → decorative.
4. Symmetric-bilinear (TCP/MPN-COV) baseline at matched params = wrong-symmetry upper bound.

### Validation (honest, no test-tuning)
Train CN-XXL+Skew-TCC end-to-end, same recipe; report train-best AND test-best vs the 91.08 quat-head ceiling. Run ≥3 seeds (mean±std); claim valid only if Skew beats Sym AND random-projector by > seed noise. Error-pair recovery table (4→17/19→10/15→14 pre/post). Invariance re-test: success = added head INCREASES forward-vs-reversed accuracy gap. No fusion, no temperature tuning. Optional cross-dataset (SHREC'17/DHG) as-is to argue not-1050-overfit.

### Target venue + claim
Pattern Recognition (primary; TPAMI if cross-dataset+full ablations). Claim: "Symmetric second-order pooling is provably blind to gesture chirality and temporal order; the antisymmetric component of a low-rank lagged feature cross-covariance is the minimal pooling object that recovers both, resolving systematic fine-grained confusions first-order and symmetric-covariance heads cannot."

## 3. Runner-up
Directed Attention Temporal Pooling (DirecFormer-style): scores marginally higher on rubric but loses on P7 (blind to spatial chirality — the larger mirror collapse) and is more incremental (DirecFormer CVPR2022, PointLSTM CVPR2020 bidirectional baseline). Clifford bivector / Vector-Neuron lost on quaternion-precedent burden (P5) + spatial-only/equivariant-readout-lossy (P3). Spectral-sign pooling lost (eigenvector sign arbitrary).

## 4. Honest risk
P(survives controls) ≈ 35-45%. Primary failure (most likely): Skew≈Sym≈random-projector within seed noise → lift (if any) from the ~25k projector params + gradient flow, NOT skew content (identical signature to the quat-head). If controls fail → DO NOT publish; report negative ("antisymmetric pooling doesn't help fine-grained gesture on tiny datasets"). Secondary: the bidirectional temporal encoder may already partly capture order (redundancy); boundary masking artifact at small r. Mitigate via δ ablation + forward-vs-reversed gap test.
