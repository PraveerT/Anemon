"""Fusion analysis: PMamba ep110 + UMDR ep48 (depth-only).

Test set order matches between both — verified by valid.txt vs test_depth_list_full.txt.
"""
import numpy as np
from itertools import product

PMAMBA_NPZ = "/notebooks/PMamba/experiments/work_dir/pmamba_branch/pmamba_test_preds.npz"
UMDR_NPZ = "/notebooks/PMamba/experiments/work_dir/umdr_test_preds.npz"


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def acc(probs, labels):
    return (np.argmax(probs, axis=-1) == labels).mean() * 100


def main():
    pm = np.load(PMAMBA_NPZ)
    um = np.load(UMDR_NPZ)
    pm_probs = pm["probs"]
    pm_labels = pm["labels"]
    um_logits_main = um["logits"]
    um_logits_sum = um["logits_all"]
    um_labels = um["labels"]

    assert (pm_labels == um_labels).all(), "label order mismatch"
    print(f"Test samples: {len(pm_labels)}")

    pm_pred = np.argmax(pm_probs, axis=-1)
    pm_acc = (pm_pred == pm_labels).mean() * 100
    um_main_pred = np.argmax(um_logits_main, axis=-1)
    um_sum_pred = np.argmax(um_logits_sum, axis=-1)
    um_main_acc = (um_main_pred == pm_labels).mean() * 100
    um_sum_acc = (um_sum_pred == pm_labels).mean() * 100

    print(f"\n=== INDIVIDUAL ===")
    print(f"PMamba ep110:        {pm_acc:.2f}")
    print(f"UMDR main (logits):  {um_main_acc:.2f}")
    print(f"UMDR sum (xs+xm+xl): {um_sum_acc:.2f}")

    um_main_probs = softmax(um_logits_main)
    um_sum_probs = softmax(um_logits_sum)

    print(f"\n=== ORACLE (any-correct) ===")
    pm_correct = pm_pred == pm_labels
    um_main_correct = um_main_pred == pm_labels
    um_sum_correct = um_sum_pred == pm_labels
    oracle_main = (pm_correct | um_main_correct).mean() * 100
    oracle_sum = (pm_correct | um_sum_correct).mean() * 100
    oracle_both = (pm_correct | um_main_correct | um_sum_correct).mean() * 100
    print(f"PMamba | UMDR-main:  {oracle_main:.2f}")
    print(f"PMamba | UMDR-sum:   {oracle_sum:.2f}")
    print(f"PMamba | UMDR all:   {oracle_both:.2f}")

    print(f"\n=== AVERAGE FUSION (uniform softmax avg) ===")
    avg_main = (pm_probs + um_main_probs) / 2
    avg_sum = (pm_probs + um_sum_probs) / 2
    avg_both = (pm_probs + um_main_probs + um_sum_probs) / 3
    print(f"PMamba + UMDR-main:  {acc(avg_main, pm_labels):.2f}")
    print(f"PMamba + UMDR-sum:   {acc(avg_sum, pm_labels):.2f}")
    print(f"PMamba + UMDR both:  {acc(avg_both, pm_labels):.2f}")

    print(f"\n=== WEIGHTED FUSION  ===  (PMamba weight w; UMDR-sum weight 1-w)")
    best = (0, 0)
    for w in np.linspace(0.0, 1.0, 21):
        fused = w * pm_probs + (1 - w) * um_sum_probs
        a = acc(fused, pm_labels)
        if a > best[1]:
            best = (w, a)
        if w in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
            print(f"  w={w:.2f}  acc={a:.2f}")
    print(f"Best weighted: w_pmamba={best[0]:.2f}  acc={best[1]:.2f}")

    print(f"\n=== TOP-2 ROUTING (specialist when PMamba uncertain) ===")
    pm_top2 = np.argsort(-pm_probs, axis=-1)[:, :2]
    pm_margin = pm_probs[np.arange(len(pm_probs)), pm_top2[:, 0]] - pm_probs[np.arange(len(pm_probs)), pm_top2[:, 1]]
    for thresh in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        defer = pm_margin < thresh
        cas = pm_pred.copy()
        cas[defer] = um_sum_pred[defer]
        a = (cas == pm_labels).mean() * 100
        print(f"  defer when margin<{thresh:.2f}: {defer.sum()} routed, acc={a:.2f}")


if __name__ == "__main__":
    main()
