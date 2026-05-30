import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path("/notebooks/Anemon")
OUT = ROOT / "experiments/work_dir/depth_qrot_uniform_weight_results.json"


BASE_ARGS = [
    "python", "train_corr_qcc_fusion.py",
    "--frames", "16",
    "--points", "128",
    "--epochs", "180",
    "--batch-size", "64",
    "--workers", "2",
    "--lr", "0.00075",
    "--min-lr", "0.000015",
    "--warmup-epochs", "10",
    "--wd", "0.04",
    "--ema-decay", "0.995",
    "--label-smoothing", "0.08",
    "--qcc-weight", "0",
    "--cycle-weight", "0",
    "--rot-aug-ce-weight", "0",
    "--rot-cycle-prob", "1.0",
    "--rot-mode", "uniform",
    "--dropout", "0.30",
    "--point-hidden", "160",
    "--temporal-hidden", "256",
    "--layers", "2",
    "--jitter", "0.006",
    "--point-drop", "0.08",
    "--seed", "29",
    "--no-publish-active",
]


RUNS = [
    ("depth_qrot_uniform_w001_s29", "0.01"),
    ("depth_qrot_uniform_w003_s29", "0.03"),
]


def summarize(workdir):
    best = Path(workdir) / "best_fused_logits.npz"
    branch = Path(workdir) / "best_branch_logits.npz"
    z = np.load(best, allow_pickle=True)
    logits = z["logits"]
    labels = z["labels"]
    item = {
        "top1": float((logits.argmax(1) == labels).mean() * 100.0),
        "epoch": int(z["epoch"][0]) if "epoch" in z.files else None,
        "workdir": str(workdir),
    }
    if branch.exists():
        b = np.load(branch, allow_pickle=True)
        item["branch_top1"] = float((b["logits"].argmax(1) == b["labels"]).mean() * 100.0)
    return item


def main():
    results = []
    for name, weight in RUNS:
        workdir = ROOT / "experiments/work_dir" / name
        cmd = BASE_ARGS + ["--workdir", str(workdir), "--rot-cycle-weight", weight]
        print("==== RUN", name, flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        item = summarize(workdir)
        item.update({"name": name, "rot_cycle_weight": weight})
        results.append(item)
        OUT.write_text(json.dumps(results, indent=2))
        print(json.dumps(item, indent=2), flush=True)
    print("==== SUMMARY ====", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
