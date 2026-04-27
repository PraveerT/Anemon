"""Run UMDR ep48 (model_best.pth.tar) on full nv test set, save logits.

Output: /notebooks/PMamba/experiments/work_dir/umdr_test_preds.npz with
  logits: (482, 25)  - "Acc_adaptive" head (main logits)
  logits_all: (482, 25)  - xs+xm+xl combo (best on ep48 val)
  labels: (482,)
  paths: list of test/class_XX/subject_X_rX strings
Order matches valid.txt (which matches PMamba's test_depth_list_full.txt).
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/notebooks/MotionRGBD-PAMI")
from config import Config
from lib.model.build import build_model
from lib.datasets.NvGesture import NvData


CKPT = "/notebooks/MotionRGBD-PAMI/Checkpoints/model_best.pth.tar"
SPLITS = "/notebooks/MotionRGBD-PAMI/data/dataset_splits/NvGesture/depth"
DATA = "/notebooks/MotionRGBD-PAMI/nv_data"
CFG = "/notebooks/MotionRGBD-PAMI/config/NvGesture.yml"
OUT = "/notebooks/PMamba/experiments/work_dir/umdr_test_preds.npz"


def make_args():
    a = argparse.Namespace()
    a.config = CFG
    Config(a)
    a.local_rank = 0
    a.distributed = False
    a.world_size = 1
    a.nprocs = 1
    a.gpu = 0
    a.device = "cuda"
    a.num_classes = 25
    a.dataset = "NvGesture"
    a.type = "K"
    a.data = DATA
    a.splits = SPLITS
    a.batch_size = 1
    a.test_batch_size = 1
    a.num_workers = 4
    a.pretrained = ""
    a.model_ema = False
    a.frp = False
    a.MultiLoss = True
    a.eval_only = True
    a.save_output = False
    a.drop_path = 0.1
    a.drop_path_prob = 0.5
    a.drop = 0.0
    a.sample_duration = 16
    a.sample_size = 224
    a.intar_fatcer = 2
    a.w = 4
    a.smprob = 0.3
    a.smixmode = "sm"
    a.shufflemix = 0.2
    a.frp_num = 0
    a.epochs = 50
    a.warmup_epochs = 5
    a.distill = 0.3
    a.temper = 0.6
    a.DC_weight = 0.2
    a.resize_rate = 0.1
    a.translate = 20
    a.color_jitter = 0.4
    a.train_interpolation = "bicubic"
    a.aa = "rand-m9-mstd0.5-inc1"
    a.reprob = 0.0
    a.remode = "pixel"
    a.recount = 1
    a.resplit = False
    a.smoothing = 0.1
    a.replace_prob = 0.25
    a.tempMix = False
    a.MixIntra = False
    a.temporal_consist = False
    a.repeated_aug = True
    a.strong_aug = False
    a.autoaug = False
    a.epoch = 49
    return a


def main():
    args = make_args()
    device = torch.device("cuda")
    model = build_model(args).to(device).eval()

    sd = torch.load(CKPT, map_location=device)
    state = sd.get("model", sd.get("state_dict", sd))
    model.load_state_dict(state, strict=False)
    print(f"loaded {CKPT}")

    splits_path = os.path.join(SPLITS, "valid.txt")
    ds = NvData(args, splits_path, "depth", phase="test")
    print(f"test samples: {len(ds)}")

    BS = 4
    loader = DataLoader(ds, batch_size=BS, num_workers=2, shuffle=False, drop_last=False)

    logits_all = np.zeros((len(ds), 25), dtype=np.float32)
    sumheads_all = np.zeros((len(ds), 25), dtype=np.float32)
    labels = np.zeros(len(ds), dtype=np.int64)
    paths = []
    pos = 0

    with torch.no_grad():
        for bi, (clips, heatmaps, target, v_path) in enumerate(loader):
            clips = clips.to(device, non_blocking=True)
            (lg, xs, xm, xl), _ = model(clips)
            n = clips.size(0)
            logits_all[pos : pos + n] = lg.cpu().numpy()
            sumheads_all[pos : pos + n] = (xs + xm + xl).cpu().numpy()
            labels[pos : pos + n] = target.cpu().numpy()
            paths.extend(list(v_path))
            pos += n
            if bi % 20 == 0:
                running = (np.argmax(logits_all[:pos], axis=-1) == labels[:pos]).mean()
                print(f"  batch {bi}  done {pos}/{len(ds)}  running_acc {running*100:.2f}")

    main_acc = (np.argmax(logits_all, axis=-1) == labels).mean()
    sum_acc = (np.argmax(sumheads_all, axis=-1) == labels).mean()
    print(f"\nMain head (logits)  acc: {main_acc*100:.2f}")
    print(f"xs+xm+xl head      acc: {sum_acc*100:.2f}")

    np.savez(
        OUT,
        logits=logits_all,
        logits_all=sumheads_all,
        labels=labels,
        paths=np.array(paths),
    )
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
