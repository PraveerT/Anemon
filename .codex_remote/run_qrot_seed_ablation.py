import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SEEDS = [31, 37, 43, 47]
ROOT = Path("experiments/work_dir")


def best_npz(path):
    z = np.load(path, allow_pickle=True)
    logits = z["logits"]
    labels = z["labels"]
    top1 = float((logits.argmax(1) == labels).mean() * 100.0)
    order = np.argsort(logits, axis=1)[:, -5:]
    top5 = float(np.mean([labels[i] in order[i] for i in range(len(labels))]) * 100.0)
    epoch = int(z["epoch"][0]) if "epoch" in z.files else None
    return {"top1": top1, "top5": top5, "epoch": epoch}


def run_one(kind, seed):
    if kind == "clean":
        wd = ROOT / f"seedctl_clean_s{seed}"
        extra = []
    elif kind == "qrot":
        wd = ROOT / f"seedctl_qrot_s{seed}"
        extra = [
            "--rot-cycle-weight", "0.02",
            "--rot-aug-ce-weight", "0.0",
            "--rot-cycle-prob", "1.0",
        ]
    else:
        raise ValueError(kind)

    if wd.exists():
        stamp = str(int(time.time()))
        archive = wd.with_name(f"{wd.name}_prev_{stamp}")
        wd.rename(archive)
        print(f"[archive] {wd} -> {archive}", flush=True)
    wd.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-u", "train_corr_qcc_fusion.py",
        "--workdir", str(wd),
        "--frames", "16",
        "--points", "128",
        "--epochs", "260",
        "--batch-size", "64",
        "--workers", "2",
        "--lr", "0.00075",
        "--min-lr", "0.000015",
        "--warmup-epochs", "10",
        "--wd", "0.04",
        "--ema-decay", "0.995",
        "--label-smoothing", "0.08",
        "--qcc-weight", "0.0",
        "--cycle-weight", "0.0",
        "--dropout", "0.30",
        "--point-hidden", "160",
        "--temporal-hidden", "256",
        "--layers", "2",
        "--jitter", "0.006",
        "--point-drop", "0.08",
        "--seed", str(seed),
        "--no-quat-inject",
        "--no-publish-active",
        *extra,
    ]
    (wd / "launch_cmd.txt").write_text(" ".join(cmd) + "\n")
    print(f"[start] {kind} seed={seed} wd={wd}", flush=True)
    with open(wd / "stdout.log", "ab", buffering=0) as out:
        rc = subprocess.call(cmd, cwd="/notebooks/Anemon", stdout=out, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"{kind} seed {seed} failed rc={rc}")
    fused = best_npz(wd / "best_fused_logits.npz")
    branch = best_npz(wd / "best_branch_logits.npz")
    result = {"kind": kind, "seed": seed, "workdir": str(wd), "fused": fused, "branch": branch}
    print("[result] " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    results = []
    out_path = ROOT / "qrot_seed_ablation_results.json"
    for seed in SEEDS:
        results.append(run_one("clean", seed))
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        results.append(run_one("qrot", seed))
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print("[done] wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
