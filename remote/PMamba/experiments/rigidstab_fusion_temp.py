"""Temperature-scale base softmax to soften it before fusion with rigidstab.

Hypothesis: pmamba_base is highly confident even when wrong. Flattening its
distribution lets rigidstab's complementary signal (r=0.120) override on cases
where base is uncertain. Sweep T_base and alpha jointly.
"""
import torch, torch.nn.functional as F, numpy as np

d = np.load('/tmp/rigidstab_fuse.npz')
base_logits = torch.from_numpy(d['base_logits'])
stab_logits = torch.from_numpy(d['stab_logits'])
y = torch.from_numpy(d['labels'])
N = len(y)

base_solo = (base_logits.argmax(-1) == y).float().mean().item()
stab_solo = (stab_logits.argmax(-1) == y).float().mean().item()
print(f"base solo {base_solo*100:.2f}  stab solo {stab_solo*100:.2f}")
print(f"\nT_base sweep with alpha sweep — flatten base, see if stab signal lands\n")
print(f"{'T_base':>7s} {'best_a':>7s} {'fuse':>7s} {'+vs base':>9s}")

# Also try with stab_temp variations (though base is the one to flatten)
results = []
for T_base in [1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]:
    base_p = F.softmax(base_logits / T_base, -1)
    stab_p = F.softmax(stab_logits, -1)
    best_a = 0; best_acc = 0
    for a in np.arange(0.0, 1.001, 0.005):
        f = a * base_p + (1 - a) * stab_p
        acc = (f.argmax(-1) == y).float().mean().item()
        if acc > best_acc: best_acc = acc; best_a = a
    print(f"{T_base:>7.1f} {best_a:>7.3f} {best_acc*100:>6.2f}% {(best_acc-base_solo)*100:>+8.2f}pp")
    results.append((T_base, best_a, best_acc))

best = max(results, key=lambda x: x[2])
print(f"\nBest: T_base={best[0]} alpha={best[1]:.3f} -> {best[2]*100:.2f}%  (+{(best[2]-base_solo)*100:.2f}pp vs base)")

# Compare with previous numbers
print(f"\nReference:")
print(f"  alpha-blend (T=1):   90.46  (+0.62pp)")
print(f"  TinyKNN fuse:        90.66  (+0.83pp)")
print(f"  oracle ceiling:      93.36  (+3.53pp)")
print(f"\nGap to oracle: {(0.9336 - best[2])*100:.2f}pp")

# Sanity: also try logit-space blend
print(f"\n=== Bonus: logit-space blend (no temperature) ===")
best_a = 0; best_acc = 0
for a in np.arange(0.0, 1.001, 0.005):
    f = a * base_logits + (1 - a) * stab_logits
    acc = (f.argmax(-1) == y).float().mean().item()
    if acc > best_acc: best_acc = acc; best_a = a
print(f"logit-blend: alpha={best_a:.3f}  acc={best_acc*100:.2f}%  (+{(best_acc-base_solo)*100:.2f}pp)")
