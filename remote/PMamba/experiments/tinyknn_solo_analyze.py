"""Solo analysis of TinyKNN (82.57% run). No fusion.
Reads /tmp/tinyknn_fuse.npz (saved from tiny_knn_full_analyze.py).
"""
import torch, torch.nn.functional as F, numpy as np

d = np.load('/tmp/tinyknn_fuse.npz')
tk = torch.from_numpy(d['tk_logits'])
y  = torch.from_numpy(d['labels'])
N  = len(y)
K  = 25  # classes

tk_p = F.softmax(tk, -1)
pred = tk_p.argmax(-1)
correct = pred == y
solo = correct.float().mean().item()

# Per-class accuracy + support
per_class_acc = torch.zeros(K)
per_class_n = torch.zeros(K, dtype=torch.long)
for c in range(K):
    mask = y == c
    n = mask.sum().item()
    per_class_n[c] = n
    per_class_acc[c] = correct[mask].float().mean().item() if n > 0 else 0.0

# Confidence stats
top_prob = tk_p.max(-1).values
entropy = -(tk_p * tk_p.clamp(min=1e-12).log()).sum(-1)
correct_conf = top_prob[correct]
wrong_conf = top_prob[~correct]
correct_ent = entropy[correct]
wrong_ent = entropy[~correct]

# Confusion: top-5 most confused class pairs (true, pred) where mismatched
wrong_idx = (~correct).nonzero(as_tuple=True)[0]
conf_pairs = {}
for i in wrong_idx.tolist():
    tc = y[i].item(); pc = pred[i].item()
    key = (tc, pc)
    conf_pairs[key] = conf_pairs.get(key, 0) + 1
top_conf = sorted(conf_pairs.items(), key=lambda x: -x[1])[:10]

print(f"=== TinyKNN solo analysis (82.57% config, N={N}) ===\n")
print(f"Overall accuracy: {solo*100:.2f}%")
print(f"Total correct:    {correct.sum().item()}/{N}")
print(f"Total wrong:      {(~correct).sum().item()}/{N}\n")

print("=== Per-class accuracy ===")
print(f"{'class':>5} {'n':>4} {'acc':>7}")
for c in range(K):
    n = per_class_n[c].item()
    a = per_class_acc[c].item()*100
    bar = '=' * int(a/5)
    print(f"{c:>5} {n:>4} {a:>6.2f}%  {bar}")

best5 = torch.argsort(per_class_acc, descending=True)[:5].tolist()
worst5 = torch.argsort(per_class_acc)[:5].tolist()
print(f"\nBest 5 classes:  " + ", ".join(f"C{c}({per_class_acc[c]*100:.0f}%)" for c in best5))
print(f"Worst 5 classes: " + ", ".join(f"C{c}({per_class_acc[c]*100:.0f}%)" for c in worst5))

print("\n=== Confidence ===")
print(f"Mean top-prob on correct:    {correct_conf.mean():.3f}  (std {correct_conf.std():.3f})")
print(f"Mean top-prob on wrong:      {wrong_conf.mean():.3f}  (std {wrong_conf.std():.3f})")
print(f"Mean entropy on correct:     {correct_ent.mean():.3f}")
print(f"Mean entropy on wrong:       {wrong_ent.mean():.3f}")
# higher entropy on wrong = well-calibrated (good for gating)
gap = wrong_ent.mean().item() - correct_ent.mean().item()
print(f"Entropy gap wrong - correct: {gap:+.3f}  (larger = better calibrated)")

# High-confidence wrong (overconfidence failures)
hi_conf_wrong = (~correct & (top_prob > 0.8)).sum().item()
lo_conf_right = (correct & (top_prob < 0.4)).sum().item()
print(f"\nHigh-conf(>0.8) wrong:  {hi_conf_wrong}  (overconfident mistakes)")
print(f"Low-conf(<0.4) correct: {lo_conf_right}  (hesitant wins)")

print("\n=== Top 10 most-confused (true -> pred) ===")
for (tc, pc), n in top_conf:
    print(f"  C{tc:2d} -> C{pc:2d} : {n}x")

# Top-k (top-3) accuracy
top3 = tk_p.topk(3, dim=-1).indices
top3_correct = (top3 == y.unsqueeze(-1)).any(-1).float().mean().item()
top5 = tk_p.topk(5, dim=-1).indices
top5_correct = (top5 == y.unsqueeze(-1)).any(-1).float().mean().item()
print(f"\nTop-3 accuracy: {top3_correct*100:.2f}%")
print(f"Top-5 accuracy: {top5_correct*100:.2f}%")

# ECE (expected calibration error, 10 bins)
bins = torch.linspace(0, 1, 11)
ece = 0.0
for b in range(10):
    m = (top_prob >= bins[b]) & (top_prob < bins[b+1])
    if m.sum() == 0: continue
    acc_bin = correct[m].float().mean().item()
    conf_bin = top_prob[m].mean().item()
    ece += (m.sum().item() / N) * abs(acc_bin - conf_bin)
print(f"\nECE (10 bins): {ece*100:.2f}%")
