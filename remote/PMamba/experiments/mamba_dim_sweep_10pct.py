"""Mamba hidden_dim sweep on 10% train subset.

Same recipe as prior sweeps. Uses winners from prior sweeps:
- knn=[32,24,48,24] (knn sweep winner)
- topk=6 (topk sweep winner)
- downsample passed via env DS_WINNER (default '2,2,2')

Sweeps mamba_hidden_dim ∈ {32, 64, 96, 128, 192, 256, 384}.
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
ORIG_BACKUP = "/tmp/train_depth_list_orig_for_md_sweep.bak"
SUBSET_LIST = "/tmp/train_depth_list_10pct.txt"
EXP_ROOT = Path("/notebooks/PMamba/experiments")
WORKDIR_ROOT = EXP_ROOT / "work_dir"
SUMMARY = "/tmp/mamba_dim_sweep_summary.txt"


VARIANTS = [
    ("M1_dim32",  32),
    ("M2_dim64",  64),
    ("M3_dim96",  96),
    ("M4_dim128", 128),  # baseline
    ("M5_dim192", 192),
    ("M6_dim256", 256),
    ("M7_dim384", 384),
]

TOPK = int(os.environ.get("TOPK_WINNER", "6"))
DS = [int(x) for x in os.environ.get("DS_WINNER", "2,2,2").split(",")]


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


def run_variant(name, dim):
    yaml_path = f"/tmp/md_sweep_{name}.yaml"
    work_dir = WORKDIR_ROOT / f"md_sweep_{name}"
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
            "downsample": DS,
            "mamba_hidden_dim": dim,
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    log_path = f"/tmp/md_sweep_{name}.log"
    print(f"\n=== {name} mamba_hidden_dim={dim} ===")
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
    return {"name": name, "mamba_hidden_dim": dim, "best": best, "final": final, "evals": accs, "time_s": elapsed}


def main():
    print(f"using topk={TOPK}, downsample={DS}")
    try:
        build_subset()
        results = []
        for name, dim in VARIANTS:
            r = run_variant(name, dim)
            if r:
                results.append(r)
                with open(SUMMARY, "w") as f:
                    f.write("name,mamba_hidden_dim,best,final,evals,time_s\n")
                    for x in sorted(results, key=lambda y: -y["best"]):
                        f.write(f'{x["name"]},{x["mamba_hidden_dim"]},{x["best"]:.2f},{x["final"]:.2f},{x["evals"]},{x["time_s"]:.0f}\n')
    finally:
        restore()

    print("\n\n=== MAMBA HIDDEN DIM SWEEP RESULT ===")
    for x in sorted(results, key=lambda y: -y["best"]):
        print(f"{x['name']:10s}  dim={x['mamba_hidden_dim']:4d}  best={x['best']:.2f}  final={x['final']:.2f}")

    if results:
        winner = max(results, key=lambda y: y["best"])
        print(f"\nWINNER: {winner['name']}  mamba_hidden_dim={winner['mamba_hidden_dim']}  best={winner['best']:.2f}")


if __name__ == "__main__":
    main()
