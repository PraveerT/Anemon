"""downsample sweep on 10% train subset.

Same recipe as knn/topk sweeps:
- 10% of train clips (seed 42), full test (482)
- pts=96 fixed, framerate=32, batch=8
- 60 ep: LR 0.00012 (ep 0..29) -> 0.000012 (ep 30..59)
- knn=[32,24,48,24], topk=<winner from topk sweep>

Vary downsample tuple (3 stage-pair reduction factors).
"""
import os
import re
import random
import shutil
import subprocess
import sys
import time
import yaml
from pathlib import Path


PROC = "/notebooks/PMamba/dataset/Nvidia/Processed"
ORIG_LIST = f"{PROC}/train_depth_list.txt"
ORIG_BACKUP = "/tmp/train_depth_list_orig_for_ds_sweep.bak"
SUBSET_LIST = "/tmp/train_depth_list_10pct.txt"
EXP_ROOT = Path("/notebooks/PMamba/experiments")
WORKDIR_ROOT = EXP_ROOT / "work_dir"
SUMMARY = "/tmp/downsample_sweep_summary.txt"

# pts_size=96 path: 96 -> 96/d0 -> 96/(d0*d1) -> 96/(d0*d1*d2)
# All chosen tuples keep last stage >= 6 points
VARIANTS = [
    ("D1_baseline_2_2_2", [2, 2, 2]),   # 96 -> 48 -> 24 -> 12 (default)
    ("D2_no_ds_1_1_1",    [1, 1, 1]),   # 96 -> 96 -> 96 -> 96
    ("D3_aggressive_4_4_4", [4, 4, 4]), # 96 -> 24 -> 6  -> may break
    ("D4_late_keep_2_2_1", [2, 2, 1]),  # 96 -> 48 -> 24 -> 24
    ("D5_early_keep_1_2_2", [1, 2, 2]), # 96 -> 96 -> 48 -> 24
    ("D6_drop_early_4_2_2", [4, 2, 2]), # 96 -> 24 -> 12 -> 6
    ("D7_drop_mid_2_4_2",   [2, 4, 2]), # 96 -> 48 -> 12 -> 6
    ("D8_smooth_2_2_1",     [2, 2, 1]), # already in D4 - alt
]

# Set TOPK from prior sweep winner (default 8 if not yet known)
TOPK = int(os.environ.get("TOPK_WINNER", "8"))


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


def run_variant(name, downsample):
    yaml_path = f"/tmp/ds_sweep_{name}.yaml"
    work_dir = WORKDIR_ROOT / f"ds_sweep_{name}"
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
            "topk": TOPK,
            "downsample": list(downsample),
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    log_path = f"/tmp/ds_sweep_{name}.log"
    print(f"\n=== {name} downsample={downsample} ===")
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
    return {"name": name, "downsample": list(downsample), "best": best, "final": final, "evals": accs, "time_s": elapsed}


def main():
    print(f"using topk={TOPK}")
    try:
        build_subset()
        results = []
        for name, ds in VARIANTS:
            r = run_variant(name, ds)
            if r:
                results.append(r)
                with open(SUMMARY, "w") as f:
                    f.write("name,downsample,best,final,evals,time_s\n")
                    for x in sorted(results, key=lambda y: -y["best"]):
                        f.write(f'{x["name"]},{x["downsample"]},{x["best"]:.2f},{x["final"]:.2f},{x["evals"]},{x["time_s"]:.0f}\n')
    finally:
        restore()

    print("\n\n=== DOWNSAMPLE SWEEP RESULT ===")
    for x in sorted(results, key=lambda y: -y["best"]):
        print(f"{x['name']:25s}  ds={x['downsample']}  best={x['best']:.2f}  final={x['final']:.2f}")

    if results:
        winner = max(results, key=lambda y: y["best"])
        print(f"\nWINNER: {winner['name']}  downsample={winner['downsample']}  best={winner['best']:.2f}")


if __name__ == "__main__":
    main()
