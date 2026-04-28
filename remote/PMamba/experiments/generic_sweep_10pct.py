"""Generic 10%-subset knob sweep driver.

Usage:
  python generic_sweep_10pct.py SWEEP_NAME

SWEEP_NAME ∈ {mamba_num_layers, mamba_output_dim, ms_num_scales, ms_feature_dim}

Reads previously-tuned knobs from env (TOPK_WINNER, DS_WINNER, MAMBA_DIM_WINNER,
MAMBA_LAYERS_WINNER, MAMBA_OUT_WINNER, MS_SCALES_WINNER) so each sweep stacks
on the prior winner.
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
ORIG_BACKUP = "/tmp/train_depth_list_orig_for_generic_sweep.bak"
SUBSET_LIST = "/tmp/train_depth_list_10pct.txt"
EXP_ROOT = Path("/notebooks/PMamba/experiments")
WORKDIR_ROOT = EXP_ROOT / "work_dir"


SWEEPS = {
    "mamba_num_layers": [1, 2, 3, 4, 6],
    "mamba_output_dim": [64, 128, 192, 256, 384, 512],
    "ms_num_scales":    [1, 2, 3, 4, 5],
    "ms_feature_dim":   [8, 16, 32, 64, 128],
}


def fixed_kwargs():
    return {
        "knn": [32, 24, 48, 24],
        "topk": int(os.environ.get("TOPK_WINNER", "6")),
        "downsample": [int(x) for x in os.environ.get("DS_WINNER", "4,4,4").split(",")],
        "mamba_hidden_dim": int(os.environ.get("MAMBA_DIM_WINNER", "32")),
        "mamba_num_layers": int(os.environ.get("MAMBA_LAYERS_WINNER", "2")),
        "mamba_output_dim": int(os.environ.get("MAMBA_OUT_WINNER", "256")),
        "ms_num_scales":    int(os.environ.get("MS_SCALES_WINNER", "4")),
        "ms_feature_dim":   int(os.environ.get("MS_FEATURE_WINNER", "32")),
    }


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


def run_variant(sweep_name, val):
    base = fixed_kwargs()
    base[sweep_name] = val
    name = f"{sweep_name}_{val}"
    yaml_path = f"/tmp/sweep_{name}.yaml"
    work_dir = WORKDIR_ROOT / f"sweep_{name}"
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
            **base,
        },
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    log_path = f"/tmp/sweep_{name}.log"
    print(f"\n=== {name} {sweep_name}={val} ===")
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
        print(f"  FAILED rc={result.returncode}  see {log_path}")
        return None

    text = open(log_path).read().replace("\r", "\n")
    accs = [float(m) for m in re.findall(r"Overall Accuracy:\s+([\d.]+)%", text)]
    if not accs:
        print(f"  no acc found")
        return None
    best = max(accs)
    final = accs[-1]
    print(f"  evals: {len(accs)} | best={best:.2f} | final={final:.2f} | time={elapsed:.0f}s")
    return {"name": name, "val": val, "best": best, "final": final, "evals": accs, "time_s": elapsed}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SWEEPS:
        print(f"usage: python generic_sweep_10pct.py {'|'.join(SWEEPS)}")
        sys.exit(1)
    sweep_name = sys.argv[1]
    values = SWEEPS[sweep_name]

    base = fixed_kwargs()
    print(f"sweep: {sweep_name}  values={values}")
    print(f"fixed: {base}")

    summary = f"/tmp/{sweep_name}_sweep_summary.txt"
    try:
        build_subset()
        results = []
        for v in values:
            r = run_variant(sweep_name, v)
            if r:
                results.append(r)
                with open(summary, "w") as f:
                    f.write(f"name,{sweep_name},best,final,evals,time_s\n")
                    for x in sorted(results, key=lambda y: -y["best"]):
                        f.write(f'{x["name"]},{x["val"]},{x["best"]:.2f},{x["final"]:.2f},{x["evals"]},{x["time_s"]:.0f}\n')
    finally:
        restore()

    print(f"\n\n=== {sweep_name.upper()} SWEEP RESULT ===")
    for x in sorted(results, key=lambda y: -y["best"]):
        print(f"{x['name']:30s}  best={x['best']:.2f}  final={x['final']:.2f}")
    if results:
        winner = max(results, key=lambda y: y["best"])
        print(f"\nWINNER: {sweep_name}={winner['val']}  best={winner['best']:.2f}")


if __name__ == "__main__":
    main()
