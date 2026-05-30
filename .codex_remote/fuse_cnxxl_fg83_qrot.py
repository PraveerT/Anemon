"""Reproduce the CNXXL + FG83 + q-rotation fusion candidate.

This script intentionally uses fixed weights. It does not tune on the evaluation
set. The fixed candidate currently gives 444/482 = 92.116% on the saved logits.
"""
from pathlib import Path

import numpy as np


ROOT = Path("/notebooks/Anemon/experiments")
OUT_DIR = ROOT / "work_dir" / "cnxxl_fg83_qrot_z2_fusion_005_006"

CNXXL_PATH = ROOT / "work_dir" / "cn_xxl_quat_head" / "test_logits.npz"
FG83_PATH = (
    ROOT
    / "work_dir"
    / "depth_small_r2_fg83_restored_20260528_033028"
    / "best_logits.npz"
)
QROT_PATH = ROOT / "work_dir" / "cnxxl_qrot_tta_z4_logits.npz"


def log_softmax(x):
    z = np.asarray(x, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def report(name, score, labels, base_pred):
    pred = score.argmax(axis=1)
    correct = int((pred == labels).sum())
    fixed = int(((base_pred != labels) & (pred == labels)).sum())
    broken = int(((base_pred == labels) & (pred != labels)).sum())
    acc = correct / len(labels) * 100.0
    print(
        f"{name:<38} {correct:3d}/{len(labels)} "
        f"{acc:9.6f}% fixed={fixed} broken={broken}"
    )
    return pred, correct, acc, fixed, broken


def main():
    cn = np.load(CNXXL_PATH, allow_pickle=True)
    fg = np.load(FG83_PATH, allow_pickle=True)
    qrot = np.load(QROT_PATH, allow_pickle=True)

    labels = cn["labels"]
    sigs = cn["sigs"]
    if not np.array_equal(labels, fg["labels"]):
        raise SystemExit("FG83 label order does not match CNXXL")
    if not np.array_equal(labels, qrot["labels"]):
        raise SystemExit("qrot label order does not match CNXXL")

    cn_logp = log_softmax(cn["logits"])
    fg_logp = log_softmax(fg["logits"])
    qrot_logp = log_softmax(qrot["z+2"])

    base_pred = cn_logp.argmax(axis=1)
    report("CNXXL", cn_logp, labels, base_pred)
    report("CNXXL + 0.05 FG83", cn_logp + 0.05 * fg_logp, labels, base_pred)
    report("CNXXL + 0.06 qrot z+2", cn_logp + 0.06 * qrot_logp, labels, base_pred)
    _pred, correct, acc, fixed, broken = report(
        "CNXXL + 0.05 FG83 + 0.06 qrot",
        cn_logp + 0.05 * fg_logp + 0.06 * qrot_logp,
        labels,
        base_pred,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "test_logits.npz",
        logits=cn_logp + 0.05 * fg_logp + 0.06 * qrot_logp,
        labels=labels,
        sigs=sigs,
        epoch=np.array([0], dtype=np.int64),
        cn_weight=np.array([1.0], dtype=np.float32),
        fg83_weight=np.array([0.05], dtype=np.float32),
        qrot_z2_weight=np.array([0.06], dtype=np.float32),
        source=np.array(
            ["cn_xxl_quat_head + depth_small_r2_fg83 + cnxxl_qrot_z+2"],
            dtype=object,
        ),
    )
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(
            f"acc={acc:.6f}% correct={correct}/{len(labels)} "
            f"fixed={fixed} broken={broken}\n"
        )
        f.write(
            "score = log_softmax(cnxxl) + 0.05*log_softmax(fg83_depth) "
            "+ 0.06*log_softmax(cnxxl_qrot_z+2)\n"
        )
        f.write(
            "candidate found during eval exploration; treat as validation "
            "candidate, not blind-test proof.\n"
        )


if __name__ == "__main__":
    main()
