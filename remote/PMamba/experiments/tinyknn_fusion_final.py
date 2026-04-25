"""Full fusion analysis: TinyKNN (82.57) + pmamba_base (89.83).
Reads /tmp/tinyknn_fuse.npz. Tests multiple fusion strategies.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

d = np.load('/tmp/tinyknn_fuse.npz')
pm = torch.from_numpy(d['pm_logits'])
tk = torch.from_numpy(d['tk_logits'])
y  = torch.from_numpy(d['labels'])
N  = len(y)

pm_p = F.softmax(pm, -1)
tk_p = F.softmax(tk, -1)
pm_right = pm_p.argmax(-1) == y
tk_right = tk_p.argmax(-1) == y

pm_solo = pm_right.float().mean().item()
tk_solo = tk_right.float().mean().item()
oracle = (pm_right | tk_right).float().mean().item()

print(f"=== Base numbers ===")
print(f"pmamba solo:  {pm_solo*100:.2f}%  ({pm_right.sum()}/{N})")
print(f"tiny solo:    {tk_solo*100:.2f}%  ({tk_right.sum()}/{N})")
print(f"oracle:       {oracle*100:.2f}%   headroom +{(oracle-pm_solo)*100:.2f}pp")

# Agreement matrix
both_r = (pm_right & tk_right).sum().item()
pm_only = (pm_right & ~tk_right).sum().item()
tk_only = (~pm_right & tk_right).sum().item()
both_w  = (~pm_right & ~tk_right).sum().item()
print(f"\nboth_right={both_r}  pm_only={pm_only}  tn_only={tk_only}  both_wrong={both_w}")


def sweep_alpha(pa, pb, y, step=0.01):
    best_a = 0; best_acc = 0
    for a in np.arange(0.0, 1.001, step):
        f = a * pa + (1 - a) * pb
        acc = (f.argmax(-1) == y).float().mean().item()
        if acc > best_acc: best_acc = acc; best_a = a
    return best_acc, best_a


# Strategy 1: raw alpha sweep
fuse1, a1 = sweep_alpha(pm_p, tk_p, y)

# Strategy 2: temp-calibrated alpha sweep
def temp_cal(logits, labels):
    bt = 1.0; bn = 1e9
    for T in np.arange(0.3, 3.1, 0.05):
        n = F.cross_entropy(logits / T, labels).item()
        if n < bn: bn = n; bt = T
    return bt
T_pm = temp_cal(pm, y); T_tk = temp_cal(tk, y)
pm_pT = F.softmax(pm / T_pm, -1); tk_pT = F.softmax(tk / T_tk, -1)
fuse2, a2 = sweep_alpha(pm_pT, tk_pT, y)

# Strategy 3: logit-level alpha (pre-softmax)
best_a3 = 0; best_acc3 = 0
for a in np.arange(0.0, 1.001, 0.02):
    f = a * pm + (1 - a) * tk
    acc = (f.argmax(-1) == y).float().mean().item()
    if acc > best_acc3: best_acc3 = acc; best_a3 = a
fuse3, a3 = best_acc3, best_a3

# Strategy 4: confidence-gated per-sample
# When pm is highly confident, trust pm; else shift toward tk
best_gfuse = 0; best_thr = 0
for thr in np.arange(0.2, 0.9, 0.05):
    pm_conf = pm_p.max(-1).values
    trust_pm = pm_conf > thr
    # fused probs: 1.0 weight to pm where confident, else 0.5
    alpha = torch.where(trust_pm, torch.ones_like(pm_conf), torch.full_like(pm_conf, 0.5))
    f = alpha.unsqueeze(-1) * pm_p + (1 - alpha.unsqueeze(-1)) * tk_p
    acc = (f.argmax(-1) == y).float().mean().item()
    if acc > best_gfuse: best_gfuse = acc; best_thr = thr

# Strategy 5: smooth confidence gate: alpha(x) = sigmoid(k * (pm_conf - c))
best_smooth = 0; best_params = (0, 0)
for c in np.arange(0.2, 0.9, 0.05):
    for kk in [2.0, 5.0, 10.0, 20.0]:
        pm_conf = pm_p.max(-1).values
        alpha = torch.sigmoid(kk * (pm_conf - c))
        f = alpha.unsqueeze(-1) * pm_p + (1 - alpha.unsqueeze(-1)) * tk_p
        acc = (f.argmax(-1) == y).float().mean().item()
        if acc > best_smooth: best_smooth = acc; best_params = (c, kk)

# Strategy 6: geometric mean (log-prob average)
def geom(pa, pb, a):
    return (pa.clamp(min=1e-12).log() * a + pb.clamp(min=1e-12).log() * (1-a))
best_g = 0; best_ag = 0
for a in np.arange(0.0, 1.001, 0.02):
    g = geom(pm_pT, tk_pT, a)
    acc = (g.argmax(-1) == y).float().mean().item()
    if acc > best_g: best_g = acc; best_ag = a

# Strategy 7: learned 2-param gate (w, b) on pmamba entropy
ent_pm = -(pm_p * pm_p.clamp(min=1e-12).log()).sum(-1)
# normalize entropy
ent_pm_n = (ent_pm - ent_pm.mean()) / ent_pm.std()
# sweep w, b
best_7 = 0
for w in np.arange(-3.0, 3.01, 0.5):
    for b in np.arange(-3.0, 3.01, 0.5):
        alpha = torch.sigmoid(w * ent_pm_n + b)  # alpha = weight on pmamba
        f = alpha.unsqueeze(-1) * pm_p + (1 - alpha.unsqueeze(-1)) * tk_p
        acc = (f.argmax(-1) == y).float().mean().item()
        if acc > best_7: best_7 = acc


print(f"\n=== Fusion strategies ===")
print(f"{'strategy':<30s} {'acc':>8s} {'+vs pm':>8s}")
print(f"{'pmamba solo':<30s} {pm_solo*100:>7.2f}% {'--':>8s}")
print(f"{'tiny solo':<30s} {tk_solo*100:>7.2f}% {(tk_solo-pm_solo)*100:>+8.2f}")
print(f"{'oracle ceiling':<30s} {oracle*100:>7.2f}% {(oracle-pm_solo)*100:>+8.2f}")
print(f"{'':-<30s} {'':->8s} {'':->8s}")
print(f"{'1. alpha raw':<30s} {fuse1*100:>7.2f}% {(fuse1-pm_solo)*100:>+8.2f}  (a={a1:.2f})")
print(f"{'2. alpha + tempcal':<30s} {fuse2*100:>7.2f}% {(fuse2-pm_solo)*100:>+8.2f}  (a={a2:.2f}, T={T_pm:.2f},{T_tk:.2f})")
print(f"{'3. logit-alpha':<30s} {fuse3*100:>7.2f}% {(fuse3-pm_solo)*100:>+8.2f}  (a={a3:.2f})")
print(f"{'4. conf-gate thr':<30s} {best_gfuse*100:>7.2f}% {(best_gfuse-pm_solo)*100:>+8.2f}  (thr={best_thr:.2f})")
print(f"{'5. smooth sigmoid gate':<30s} {best_smooth*100:>7.2f}% {(best_smooth-pm_solo)*100:>+8.2f}  (c={best_params[0]:.2f}, k={best_params[1]})")
print(f"{'6. geom mean + tempcal':<30s} {best_g*100:>7.2f}% {(best_g-pm_solo)*100:>+8.2f}  (a={best_ag:.2f})")
print(f"{'7. entropy-gate sweep':<30s} {best_7*100:>7.2f}% {(best_7-pm_solo)*100:>+8.2f}")

# Per-class: where does tiny fusion HELP
print(f"\n=== Per-class contribution (best strategy: {'2 (cal)' if fuse2 > fuse1 else '1 (raw)'}) ===")
best_pT_p = pm_pT if fuse2 > fuse1 else pm_p
best_tT_p = tk_pT if fuse2 > fuse1 else tk_p
a_star = a2 if fuse2 > fuse1 else a1
f_best = a_star * best_pT_p + (1 - a_star) * best_tT_p
f_right = f_best.argmax(-1) == y
K = 25
print(f"{'class':>5} {'n':>4} {'pm':>6} {'fused':>6} {'delta':>6}")
for c in range(K):
    m = y == c
    if m.sum() == 0: continue
    pm_a = pm_right[m].float().mean().item()*100
    fu_a = f_right[m].float().mean().item()*100
    d = fu_a - pm_a
    flag = "  ^" if d > 0 else ("  v" if d < 0 else "")
    print(f"{c:>5} {m.sum().item():>4} {pm_a:>5.1f}% {fu_a:>5.1f}% {d:>+6.1f}{flag}")

recovered = (f_right & ~pm_right).nonzero(as_tuple=True)[0]
lost = (~f_right & pm_right).nonzero(as_tuple=True)[0]
print(f"\nFusion recovered {len(recovered)} pmamba-failures")
print(f"Fusion broke     {len(lost)} pmamba-correct")
print(f"Net gain:        {(fuse2 if fuse2>fuse1 else fuse1)*100-pm_solo*100:+.2f}pp = {len(recovered)-len(lost)} samples")
