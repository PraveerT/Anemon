"""knn sweep on 10% train subset.

Setup:
- 10% of train clips (deterministic seed; ~105 clips), full test (482)
- pts_size=96 fixed, framerate=32, batch=8
- 60 epochs total: LR 0.00012 (ep 0..29) → 0.000012 (ep 30..59)
- 9 knn variants spanning small/medium/large/growing/shrinking shapes
- Report best test acc per variant; pick winner

Outputs each run to /tmp/knn_sweep/<name>/ and a summary csv.
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
ORIG_BACKUP = "/tmp/train_depth_list_orig_for_knn_sweep.bak"
SUBSET_LIST = "/tmp/train_depth_list_10pct.txt"
EXP_ROOT = Path("/notebooks/PMamba/experiments")
WORKDIR_ROOT = EXP_ROOT / "work_dir"
SUMMARY = "/tmp/knn_sweep_summary.txt"


VARIANTS = [
    ("K1_baseline",   [32, 24, 48, 24]),
    ("K2_all16",      [16, 16, 16, 16]),
    ("K3_all32",      [32, 32, 32, 32]),
    ("K4_all48",      [48, 48, 48, 48]),
    ("K5_grow",       [16, 24, 48, 64]),
    ("K6_shrink",     [64, 48, 24, 16]),
    ("K7_belly",      [24, 48, 48, 24]),
    ("K8_last_big",   [16, 24, 48, 96]),
    ("K9_small",      [8, 12, 16, 12]),
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
    print(f"subset: {n_keep}/{len(lines)} clips (seed=42), wrote {SUBSET_LIST}")


def restore():
    if os.path.exists(ORIG_BACKUP):
        shutil.copy(ORIG_BACKUP, ORIG_LIST)
        print(f"restored original train list")


def run_variant(name, knn):
    yaml_path = f"/tmp/knn_sweep_{name}.yaml"
    work_dir = WORKDIR_ROOT / f"knn_sweep_{name}"
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
            "knn": list(knn),
            "topk": 8,
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    log_path = f"/tmp/knn_sweep_{name}.log"
    print(f"\n=== {name} knn={knn} ===")
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

    # parse Overall Accuracy lines
    text = open(log_path).read().replace("\r", "\n")
    accs = [float(m) for m in re.findall(r"Overall Accuracy:\s+([\d.]+)%", text)]
    if not accs:
        print(f"  no acc found")
        return None
    best = max(accs)
    final = accs[-1]
    print(f"  evals: {len(accs)} | best={best:.2f} | final={final:.2f} | time={elapsed:.0f}s")
    return {"name": name, "knn": list(knn), "best": best, "final": final, "evals": accs, "time_s": elapsed}


def main():
    try:
        build_subset()
        results = []
        for name, knn in VARIANTS:
            r = run_variant(name, knn)
            if r:
                results.append(r)
                # write summary so far
                with open(SUMMARY, "w") as f:
                    f.write("name,knn,best,final,evals,time_s\n")
                    for x in sorted(results, key=lambda y: -y["best"]):
                        f.write(f'{x["name"]},{x["knn"]},{x["best"]:.2f},{x["final"]:.2f},{x["evals"]},{x["time_s"]:.0f}\n')
    finally:
        restore()

    print("\n\n=== KNN SWEEP RESULT ===")
    for x in sorted(results, key=lambda y: -y["best"]):
        print(f"{x['name']:14s}  knn={x['knn']}  best={x['best']:.2f}  final={x['final']:.2f}")

    if results:
        winner = max(results, key=lambda y: y["best"])
        print(f"\nWINNER: {winner['name']}  knn={winner['knn']}  best={winner['best']:.2f}")


if __name__ == "__main__":
    main()
