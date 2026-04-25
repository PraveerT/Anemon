"""Fusion analysis: TinyKNN (81.54) vs prior tiny (77.34) vs pmamba_base (89.83).

Uses saved logits (no retraining):
  /tmp/tiny_knn_logits.npz  - TinyKNN test logits
  /tmp/fuse_final.npz       - pmamba_base test logits + labels

Report:
  - solo: pmamba, tiny(knn), tiny(noknn)
  - oracle for each tiny vs pmamba
  - best alpha-blend for each tiny vs pmamba
  - error correlation r (pmamba errors vs tiny errors)
  - unique-recovery samples (where tiny right, pmamba wrong)
Verdict: did adding kNN improve FUSION (not just solo)?
"""
import torch, torch.nn.functional as F, numpy as np

d_tk = np.load('/tmp/tiny_knn_logits.npz')
d_prev = np.load('/tmp/fuse_final.npz')

tk_logits = torch.from_numpy(d_tk['logits'])
tk_labels = torch.from_numpy(d_tk['labels'])

pm_logits = torch.from_numpy(d_prev['pm_te_logits'])
tn_logits = torch.from_numpy(d_prev['tiny_te_logits'])  # prior no-knn tiny
pm_labels = torch.from_numpy(d_prev['labels'])

assert torch.equal(tk_labels, pm_labels), "label mismatch"
y = pm_labels
N = len(y)

def metrics(pm, tn, name):
    pm_p = F.softmax(pm, -1)
    tn_p = F.softmax(tn, -1)
    pm_solo = (pm_p.argmax(-1) == y).float().mean().item()
    tn_solo = (tn_p.argmax(-1) == y).float().mean().item()
    pm_right = pm_p.argmax(-1) == y
    tn_right = tn_p.argmax(-1) == y
    oracle = (pm_right | tn_right).float().mean().item()
    # alpha sweep
    best_a = 0; best_a_acc = 0
    for a in np.arange(0.0, 1.01, 0.02):
        f = a * pm_p + (1 - a) * tn_p
        acc = (f.argmax(-1) == y).float().mean().item()
        if acc > best_a_acc: best_a_acc = acc; best_a = a
    # error correlation (Pearson phi)
    ep = (~pm_right).float(); et = (~tn_right).float()
    cov = ((ep - ep.mean()) * (et - et.mean())).mean().item()
    sp = ep.std().item(); st = et.std().item()
    r = cov / (sp * st + 1e-9)
    # confusion table
    both_right = (pm_right & tn_right).sum().item()
    pm_only = (pm_right & ~tn_right).sum().item()
    tn_only = (~pm_right & tn_right).sum().item()
    both_wrong = (~pm_right & ~tn_right).sum().item()
    print(f"\n=== {name} ===")
    print(f"  pmamba solo: {pm_solo*100:.2f}%   tiny solo: {tn_solo*100:.2f}%")
    print(f"  oracle: {oracle*100:.2f}%   gap vs pm: +{(oracle-pm_solo)*100:.2f}pp")
    print(f"  best alpha-blend: {best_a_acc*100:.2f}% at a={best_a:.2f}   gain vs pm: +{(best_a_acc-pm_solo)*100:.2f}pp")
    print(f"  error corr r = {r:.3f}")
    print(f"  both_right={both_right}  pm_only={pm_only}  tn_only={tn_only}  both_wrong={both_wrong}")
    print(f"  unique recovery potential (tn_only): {tn_only} samples = {tn_only/N*100:.2f}pp")
    return dict(pm=pm_solo, tn=tn_solo, oracle=oracle, fuse=best_a_acc, fuse_a=best_a,
                r=r, tn_only=tn_only, pm_only=pm_only, both_wrong=both_wrong)


prev = metrics(pm_logits, tn_logits, "PRIOR: pmamba + tiny-noknn (77.34)")
new  = metrics(pm_logits, tk_logits, "NEW:   pmamba + TinyKNN  (81.54)")

print("\n" + "="*60)
print("COMPARISON")
print(f"{'metric':<25s} {'prior (tiny)':>14s} {'new (TinyKNN)':>14s} {'delta':>10s}")
print(f"{'-'*25:<25s} {'-'*14:>14s} {'-'*14:>14s} {'-'*10:>10s}")
for k in ['tn', 'oracle', 'fuse', 'r', 'tn_only']:
    dv = new[k] - prev[k]
    fmt = ".3f" if k == 'r' else ".4f" if k in ('tn', 'oracle', 'fuse') else ".0f"
    print(f"{k:<25s} {prev[k]:>14{fmt}} {new[k]:>14{fmt}} {dv:>+10{fmt}}")

print("\nVerdict:")
if new['fuse'] > prev['fuse']:
    print(f"  Fusion IMPROVED: +{(new['fuse']-prev['fuse'])*100:.2f}pp. Architectural gain carries into fusion.")
elif new['fuse'] == prev['fuse']:
    print("  Fusion UNCHANGED. Solo gain but no fusion benefit.")
else:
    print(f"  Fusion WORSE: {(new['fuse']-prev['fuse'])*100:.2f}pp. Errors now more correlated with pmamba.")

if new['oracle'] > prev['oracle']:
    print(f"  Oracle UP: +{(new['oracle']-prev['oracle'])*100:.2f}pp -> TinyKNN recovers samples noKnn missed.")
else:
    print(f"  Oracle DOWN: {(new['oracle']-prev['oracle'])*100:.2f}pp -> TinyKNN becoming more correlated with pmamba.")
