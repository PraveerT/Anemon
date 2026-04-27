"""Prepare UMDR-Net's expected data layout from our extracted nvGesture_v1.

For each clip in the train.txt / valid.txt splits, extract 80 frames
(within the v2 .lst gesture window) from sk_depth/ and sk_color/ directories,
symlink them to:
  <DATA_ROOT>/depth/<phase>/<class>/<subject>/{000000..000079}.jpg
  <DATA_ROOT>/rgb/<phase>/<class>/<subject>/{000000..000079}.jpg
"""
import os
import re
import sys
from multiprocessing import Pool, cpu_count

V2_LST_TRAIN = "/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1/nvgesture_train_correct_cvpr2016_v2.lst"
V2_LST_TEST = "/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1/nvgesture_test_correct_cvpr2016_v2.lst"
SRC_VIDEO_DATA = "/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1/Video_data"
DATA_ROOT = "/notebooks/MotionRGBD-PAMI/nv_data"
N_FRAMES = 80


def parse_v2(line):
    sp = re.split(r"[ \t\n\r]+", line.strip())
    rel = sp[0].split(":", 1)[1].lstrip("./").rstrip("/")
    color_field = sp[2].split(":")
    sframe = int(color_field[2])
    eframe = int(color_field[3])
    label = int(sp[-1].split(":")[-1]) - 1
    return rel, sframe, eframe, label


def make_links(args):
    rel, s, e, label, phase = args
    # rel: Video_data/class_01/subject3_r0
    parts = rel.split("/")
    cls_subject = "/".join(parts[1:])                           # class_01/subject3_r0
    src_color = os.path.join(SRC_VIDEO_DATA, cls_subject, "sk_color")
    src_depth = os.path.join(SRC_VIDEO_DATA, cls_subject, "sk_depth")

    dst_rgb = os.path.join(DATA_ROOT, "rgb", phase, cls_subject)
    dst_depth = os.path.join(DATA_ROOT, "depth", phase, cls_subject)
    os.makedirs(dst_rgb, exist_ok=True)
    os.makedirs(dst_depth, exist_ok=True)

    n_avail = e - s
    if n_avail < N_FRAMES:
        # short clip: pad by repeating last frame indices (loader will subsample)
        sel = list(range(s, e)) + [e - 1] * (N_FRAMES - n_avail)
    else:
        # uniformly take N_FRAMES from [s, e)
        sel = [s + int(i * n_avail / N_FRAMES) for i in range(N_FRAMES)]

    fails = 0
    for j, src_idx in enumerate(sel):
        src_color_jpg = os.path.join(src_color, f"img{src_idx + 1:06d}.jpg")
        src_depth_jpg = os.path.join(src_depth, f"img{src_idx + 1:06d}.jpg")
        dst_color_jpg = os.path.join(dst_rgb, f"{j:06d}.jpg")
        dst_depth_jpg = os.path.join(dst_depth, f"{j:06d}.jpg")
        if not os.path.exists(dst_color_jpg):
            try:
                os.symlink(src_color_jpg, dst_color_jpg)
            except FileExistsError:
                pass
        if not os.path.exists(dst_depth_jpg):
            try:
                os.symlink(src_depth_jpg, dst_depth_jpg)
            except FileExistsError:
                pass
        # check files actually exist
        if not os.path.exists(src_color_jpg):
            fails += 1
    return rel, fails


def main():
    splits = {
        "train": V2_LST_TRAIN,
        "test": V2_LST_TEST,
    }

    for phase, lst_path in splits.items():
        lines = open(lst_path).readlines()
        print(f"{phase}: {len(lines)} clips")
        args_list = []
        for line in lines:
            rel, s, e, label = parse_v2(line)
            args_list.append((rel, s, e, label, phase))
        with Pool(processes=max(1, cpu_count() // 2)) as pool:
            results = pool.map(make_links, args_list)
        bad = sum(1 for _, f in results if f > 0)
        print(f"  {phase} done. clips with missing source frames: {bad}")


if __name__ == "__main__":
    main()
