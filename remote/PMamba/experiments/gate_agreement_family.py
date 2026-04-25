"""Test agreement-family gating signatures (12-18) on TinyKNN + pmamba fusion.
Reads /tmp/tinyknn_fuse.npz.

Signatures:
  12. top1_agree        : 1 if pm.argmax == tn.argmax, else 0
  13. top3_jaccard      : |top3_pm intersect top3_tn| / |union|
  14. top5_jaccard      : same at k=5
  15. KL(pm || tn)      : divergence (pm from tn)
  16. JS(pm, tn)        : symmetric Jensen-Shannon
  17. pm_on_tn_top1     : pm's assigned prob for tn's predicted class
  18. tn_on_pm_top1     : tn's assigned prob for pm's predicted class

For each signature: sweep threshold and (alpha_lo, alpha_hi) rules.
"""
import torch, torch.nn.functional as F, numpy as np

d = np.load('/tmp/tinyknn_fuse.npz')
pm = torch.from_numpy(d['pm_logits']); tk = torch.from_numpy(d['tk_logits'])
y  = torch.from_numpy(d['labels'])
N  = len(y)
pm_p = F.softmax(pm, -1); tk_p = F.softmax(tk, -1)

pm_argmax = pm_p.argmax(-1)
tk_argmax = tk_p.argmax(-1)

# 12. top1 agreement
top1_agree = (pm_argmax == tk_argmax).float()

# 13, 14. top-k jaccard
def topk_jaccard(k):
    topk_pm = pm_p.topk(k, -1).indices
    topk_tn = tk_p.topk(k, -1).indices
    j = torch.zeros(N)
    for i in range(N):
        s_pm = set(topk_pm[i].tolist()); s_tn = set(topk_tn[i].tolist())
        inter = len(s_pm & s_tn); union = len(s_pm | s_tn)
        j[i] = inter / union if union > 0 else 0.0
    return j
top3_jac = topk_jaccard(3)
top5_jac = topk_jaccard(5)

# 15. KL(pm || tn)
kl_pm_tn = (pm_p * (pm_p.clamp(min=1e-12).log() - tk_p.clamp(min=1e-12).log())).sum(-1)

# 16. JS(pm, tn)
m_p = 0.5 * (pm_p + tk_p)
kl1 = (pm_p * (pm_p.clamp(min=1e-12).log() - m_p.clamp(min=1e-12).log())).sum(-1)
kl2 = (tk_p * (tk_p.clamp(min=1e-12).log() - m_p.clamp(min=1e-12).log())).sum(-1)
js = 0.5 * (kl1 + kl2)

# 17. pm's prob on tn's top1 class
pm_on_tn_top1 = pm_p.gather(1, tk_argmax.unsqueeze(-1)).squeeze(-1)

# 18. tn's prob on pm's top1 class
tn_on_pm_top1 = tk_p.gather(1, pm_argmax.unsqueeze(-1)).squeeze(-1)

sigs = {
    '12.top1_agree':    top1_agree,
    '13.top3_jaccard':  top3_jac,
    '14.top5_jaccard':  top5_jac,
    '15.KL(pm||tn)':    kl_pm_tn,
    '16.JS(pm,tn)':     js,
    '17.pm_on_tn_top1': pm_on_tn_top1,
    '18.tn_on_pm_top1': tn_on_pm_top1,
}

# Per-group statistics (tn_only is the key group)
pm_right = pm_argmax == y
tk_right = tk_argmax == y
groups = {
    'both_right': pm_right & tk_right,
    'pm_only':    pm_right & ~tk_right,
    'tn_only':    ~pm_right & tk_right,
    'both_wrong': ~pm_right & ~tk_right,
}

print("=== Signature distribution per fusion-group (mean +/- std) ===")
print(f"{'sig':<20s}", end='')
for g in groups: print(f" {g:>13s}", end='')
print()
for name, s in sigs.items():
    print(f"{name:<20s}", end='')
    for gname, m in groups.items():
        if m.sum() == 0:
            print(f" {'na':>13s}", end='')
        else:
            mu = s[m].mean().item()
            sd = s[m].std().item() if m.sum() > 1 else 0.0
            print(f" {mu:>6.3f}+-{sd:<4.2f}", end='')
    print()


# tn_only z-score for each signature
print(f"\n=== tn_only z-score (signature uniquely describes tn_only samples?) ===")
print(f"{'sig':<20s} {'z-tn':>8s}  interpretation")
tn_m = groups['tn_only']
for name, s in sigs.items():
    overall_mean = s.mean().item(); overall_std = s.std().item() + 1e-9
    z = (s[tn_m].mean().item() - overall_mean) / overall_std
    interp = ""
    if abs(z) >= 1.0: interp = "STRONG signal"
    elif abs(z) >= 0.5: interp = "moderate"
    elif abs(z) >= 0.25: interp = "weak"
    else: interp = "null"
    print(f"{name:<20s} {z:>+8.3f}  {interp}")


# Threshold gate sweep for each
print(f"\n=== Best threshold-gate per signature ===")
print(f"baseline alpha-only (a=0.76): 90.66%\n")
print(f"{'sig':<20s} {'acc':>7s} {'+vs base':>9s}  rule")
best_global = 0; best_rule = None
for name, s in sigs.items():
    best_s = 0; best_s_rule = None
    qs = np.linspace(0.05, 0.95, 19)
    for q in qs:
        thr_t = torch.quantile(s.float(), torch.tensor([q], dtype=torch.float))
        thr = thr_t.item()
        for a_lo in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.76]:
            for a_hi in [0.76, 0.8, 0.85, 0.9, 0.95, 1.0]:
                # When signature is ABOVE threshold, use a_hi (for confidence: more trust pm)
                # When BELOW, use a_lo (more trust tiny)
                alpha = torch.where(s > thr,
                                     torch.full_like(s, a_hi),
                                     torch.full_like(s, a_lo)).float()
                f = alpha.unsqueeze(-1) * pm_p + (1 - alpha.unsqueeze(-1)) * tk_p
                acc = (f.argmax(-1) == y).float().mean().item()
                if acc > best_s:
                    best_s = acc
                    best_s_rule = (q, thr, a_lo, a_hi)
    gain = (best_s - 0.9066) * 100
    print(f"{name:<20s} {best_s*100:>6.2f}% {gain:>+8.2f}  q={best_s_rule[0]:.2f}, thr={best_s_rule[1]:.3f}, alo={best_s_rule[2]:.2f}, ahi={best_s_rule[3]:.2f}")
    if best_s > best_global:
        best_global = best_s; best_rule = (name, best_s_rule)


# Combined: try best signature combined with pm_max_prob (orthogonal info)
print(f"\n=== 2-signature combined gate ===")
pm_maxprob = pm_p.max(-1).values
# For each candidate signature, combine with pm_maxprob via mean-based threshold
best_2 = 0; best_2_rule = None
for name, s in sigs.items():
    # Standardize each
    s_z = (s - s.mean()) / (s.std() + 1e-9)
    pc_z = (pm_maxprob - pm_maxprob.mean()) / (pm_maxprob.std() + 1e-9)
    for w1 in np.arange(-2, 2.1, 0.5):
        for w2 in np.arange(-2, 2.1, 0.5):
            for b in np.arange(-2, 2.1, 0.5):
                score = w1 * s_z + w2 * pc_z + b
                alpha = torch.sigmoid(score).float()
                # rescale alpha to [0.5, 1.0] range (pm-biased)
                alpha_s = 0.5 + 0.5 * alpha
                f = alpha_s.unsqueeze(-1) * pm_p + (1 - alpha_s.unsqueeze(-1)) * tk_p
                acc = (f.argmax(-1) == y).float().mean().item()
                if acc > best_2:
                    best_2 = acc; best_2_rule = (name, w1, w2, b)
print(f"Best 2-sig: {best_2_rule[0]} + pm_maxprob")
print(f"  acc: {best_2*100:.2f}%  (+{(best_2-0.9066)*100:+.2f}pp vs alpha)")
print(f"  coeffs: w_sig={best_2_rule[1]}, w_pmc={best_2_rule[2]}, b={best_2_rule[3]}")


# Special: look at top1_agree specifically
print(f"\n=== top1 agreement analysis (the 48 disagreement cases) ===")
disagree = top1_agree == 0
print(f"top1_agree  count: {int(top1_agree.sum().item())}/{N} ({top1_agree.mean().item()*100:.1f}%)")
print(f"disagreement:     {int(disagree.sum().item())}/{N}")
# Within disagreement: who's right?
d_pm_right = (disagree & pm_right).sum().item()
d_tn_right = (disagree & tk_right).sum().item()
d_both_wrong = (disagree & ~pm_right & ~tk_right).sum().item()
print(f"Within disagreement:")
print(f"  pm right:    {d_pm_right}/{int(disagree.sum().item())}")
print(f"  tn right:    {d_tn_right}/{int(disagree.sum().item())}")
print(f"  both wrong:  {d_both_wrong}/{int(disagree.sum().item())}")

# Best alpha in disagreement region only
best_d = 0; best_d_a = 0
for a in np.arange(0.0, 1.001, 0.02):
    # If disagree: use a. If agree: alpha doesn't matter (same vote) → use neutral
    alpha = torch.where(disagree, torch.tensor(a), torch.tensor(1.0)).float()
    f = alpha.unsqueeze(-1) * pm_p + (1 - alpha.unsqueeze(-1)) * tk_p
    acc = (f.argmax(-1) == y).float().mean().item()
    if acc > best_d: best_d = acc; best_d_a = a
print(f"\nSplit-alpha (agree: ignore tn; disagree: alpha_d):")
print(f"  best acc: {best_d*100:.2f}% at alpha_d={best_d_a:.2f}")
print(f"  gain: {(best_d-0.9066)*100:+.2f}pp")


print(f"\n=== BEST OVERALL ===")
print(f"{best_rule[0]} threshold-gate: {best_global*100:.2f}%  {(best_global-0.9066)*100:+.2f}pp")
print(f"2-sig combined:              {best_2*100:.2f}%  {(best_2-0.9066)*100:+.2f}pp")
print(f"Split-alpha on disagree:     {best_d*100:.2f}%  {(best_d-0.9066)*100:+.2f}pp")
