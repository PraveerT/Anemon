"""topk sweep on 10% train subset.

Same recipe as knn_sweep_10pct.py:
- 10% of train clips (seed 42, deterministic)
- pts=96 fixed, framerate=32, batch=8
- 60 ep: LR 0.00012 (ep 0..29) -> 0.000012 (ep 30..59)
- knn=[32,24,48,24] (winner from prior sweep)
- vary topk
"""
import os
import re
import random
import shutil
import subprocess
import time
import yaml
from pathlib import Path


PROC = "/notebooks/PMamba/dataset/Nvidia/Processed"
ORIG_LIST = f"{PROC}/train_depth_list.txt"
ORIG_BACKUP = "/tmp/train_depth_list_orig_for_topk_sweep.bak"
SUBSET_LIST = "/tmp/train_depth_list_10pct.txt"
EXP_ROOT = Path("/notebooks/PMamba/experiments")
WORKDIR_ROOT = EXP_ROOT / "work_dir"
SUMMARY = "/tmp/topk_sweep_summary.txt"


VARIANTS = [
    ("T1_topk2",   2),
    ("T2_topk4",   4),
    ("T3_topk6",   6),
    ("T4_topk8",   8),    # baseline
    ("T5_topk12",  12),
    ("T6_topk16",  16),
    ("T7_topk24",  24),
]


def build_subset():
    if not os.path.exists(ORIG_BACKUP):
        shutil.copy(ORIG_LIST, ORIG_BACKUP)
    lines = open(ORIG_BACKUP).readlines()
    rng = random.Random(42)
    rng.shuffle(lines)
    n_keep = max(1, int(round(0.10 * len(lines))))
    subset = sorted(lines[:n_keep])
    with open(SUBSET_LIST, "w") as f:
        f.writelines(subset)
    shutil.copy(SUBSET_LIST, ORIG_LIST)
    print(f"subset: {n_keep}/{len(lines)} clips (seed=42)")


def restore():
    if os.path.exists(ORIG_BACKUP):
        shutil.copy(ORIG_BACKUP, ORIG_LIST)
        print("restored original train list")


def run_variant(name, topk):
    yaml_path = f"/tmp/topk_sweep_{name}.yaml"
    work_dir = WORKDIR_ROOT / f"topk_sweep_{name}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    cfg = {
        "dataloader": "nvidia_dataloader.NvidiaLoader",
        "phase": "train",
        "num_epoch": 60,
        "work_dir": str(work_dir) + "/",
        "batch_size": 8,
        "test_batch_size": 1,
        "num_worker": 8,
        "device": 0,
        "log_interval": 100,
        "eval_interval": 1,
        "save_interval": 200,
        "framesize": 32,
        "pts_size": 96,
        "dynamic_pts_size": False,
        "strict_load": False,
        "optimizer_args": {
            "optimizer": "Adam",
            "base_lr": 0.00012,
            "step": [30],
            "weight_decay": 0.03,
            "start_epoch": 0,
            "nesterov": False,
        },
        "train_loader_args": {"phase": "train", "framerate": 32},
        "test_loader_args": {"phase": "test", "framerate": 32},
        "model": "models.motion.Motion",
        "model_args": {
            "pts_size": 96,
            "num_classes": 25,
            "knn": [32, 24, 48, 24],
            "topk": topk,
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    log_path = f"/tmp/topk_sweep_{name}.log"
    print(f"\n=== {name} topk={topk} ===")
    t0 = time.time()
    with open(log_path, "w") as logf:
        result = subprocess.run(
            ["python", "-u", "main.py", "--config", yaml_path],
            cwd=str(EXP_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED rc={result.returncode}")
        return None

    text = open(log_path).read().replace("\r", "\n")
    accs = [float(m) for m in re.findall(r"Overall Accuracy:\s+([\d.]+)%", text)]
    if not accs:
        print(f"  no acc found")
        return None
    best = max(accs)
    final = accs[-1]
    print(f"  evals: {len(accs)} | best={best:.2f} | final={final:.2f} | time={elapsed:.0f}s")
    return {"name": name, "topk": topk, "best": best, "final": final, "evals": accs, "time_s": elapsed}


def main():
    try:
        build_subset()
        results = []
        for name, topk in VARIANTS:
            r = run_variant(name, topk)
            if r:
                results.append(r)
                with open(SUMMARY, "w") as f:
                    f.write("name,topk,best,final,evals,time_s\n")
                    for x in sorted(results, key=lambda y: -y["best"]):
                        f.write(f'{x["name"]},{x["topk"]},{x["best"]:.2f},{x["final"]:.2f},{x["evals"]},{x["time_s"]:.0f}\n')
    finally:
        restore()

    print("\n\n=== TOPK SWEEP RESULT ===")
    for x in sorted(results, key=lambda y: -y["best"]):
        print(f"{x['name']:14s}  topk={x['topk']:3d}  best={x['best']:.2f}  final={x['final']:.2f}")

    if results:
        winner = max(results, key=lambda y: y["best"])
        print(f"\nWINNER: {winner['name']}  topk={winner['topk']}  best={winner['best']:.2f}")


if __name__ == "__main__":
    main()
