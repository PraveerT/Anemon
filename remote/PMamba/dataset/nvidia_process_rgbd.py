"""RGB-D preprocessing for NVGesture using sk_wrapped (depth aligned to color)
+ sk_color (RGB).

Reads:
  /notebooks/PMamba/dataset/Nvidia/nvgesture_{train,test}_correct_cvpr2016_v2.lst
  /notebooks/PMamba/dataset/Nvidia/Video_data/.../sk_wrapped.avi (symlink)
  /notebooks/PMamba/dataset/Nvidia/Video_data/.../sk_color.avi   (symlink)

Writes:
  /notebooks/PMamba/dataset/Nvidia/Processed/{train,test}/.../sk_wrapped.avi/{idx}_rgbd_label_NN.npy
  shape (32, pts_size, 11) int  — channels: u, v, d, t, x, y, z, t, R, G, B
  /notebooks/PMamba/dataset/Nvidia/Processed/{train,test}_rgbd_list.txt
"""
import os
import re
import copy
import cv2
import numpy as np
import utils
from multiprocessing import Pool, cpu_count

PREFIX = "./Nvidia"
SENSOR_DEPTH = "sk_depth"     # native depth (matches baseline 89.83 geometry)
SENSOR_COLOR = "sk_wrapped"   # color reprojected onto depth-sensor frame; pixel-aligned to sk_depth
PTS_SIZE = 512
FRAME_RATE_OUT = 32           # number of frames sampled from each clip


def _parse_v2_line(line):
    sp = re.split(r"[ \t\n\r]+", line.strip())
    # path:./Video_data/...    depth:sk_depth:138:218    color:sk_color:138:218
    rel = sp[0].split(":", 1)[1]                        # ./Video_data/class_01/subject13_r0
    rel = rel.lstrip("./").rstrip("/")                  # Video_data/class_01/subject13_r0
    color_field = sp[2].split(":")
    sframe = int(color_field[2])
    eframe = int(color_field[3])
    label = int(sp[-1].split(":")[-1]) - 1
    return rel, sframe, eframe, label


def _load_avi_window(path, start, end, gray):
    """Load frames [start, end) from an .avi. Returns (T_native, H, W, C)."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ok, fr = cap.read()
        if not ok:
            break
        if gray:
            if fr.ndim == 3:
                fr = fr[..., 0:1]
        else:
            fr = cv2.resize(fr, (320, 240))            # match depth resolution
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(fr)
    cap.release()
    return np.stack(frames, axis=0) if frames else None


def process_one(args):
    idx, rel, sframe, eframe, label, train_or_test = args
    depth_path = os.path.join(PREFIX, rel, SENSOR_DEPTH + ".avi")
    color_path = os.path.join(PREFIX, rel, SENSOR_COLOR + ".avi")

    depth = _load_avi_window(depth_path, sframe, eframe, gray=True)        # (T, 240, 320, 1)
    color = _load_avi_window(color_path, sframe, eframe, gray=False)       # (T, 240, 320, 3)
    if depth is None or color is None:
        print(f"[fail] {idx} {rel} could not load")
        return

    # uniform sub-sample to FRAME_RATE_OUT frames
    ind = utils.key_frame_sampling(len(depth), FRAME_RATE_OUT)
    depth = depth[ind]
    color = color[ind]

    pts = np.zeros((FRAME_RATE_OUT, PTS_SIZE, 11), dtype=np.int32)
    for t in range(FRAME_RATE_OUT):
        frame = depth[t, :, :, 0]
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        _, thresh = cv2.threshold(frame_blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = utils.save_largest_label(thresh)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel)

        masked = frame * thresh
        uvdt = utils.points_sampling(utils.generate_points(masked, t), PTS_SIZE)  # (N, 4) [u, v, d, t]
        xyz_t = utils.uvd2xyz_sherc(copy.deepcopy(uvdt))                          # (N, 4) [x, y, z, t]

        # utils returns [row, col, depth, t]; color frame is (T, H, W, 3) → index [t, row, col, :]
        row = uvdt[:, 0].astype(np.int64)
        col = uvdt[:, 1].astype(np.int64)
        row = np.clip(row, 0, color.shape[1] - 1)
        col = np.clip(col, 0, color.shape[2] - 1)
        rgb = color[t, row, col, :]                                               # (N, 3) uint8

        pts[t, :, 0:4] = uvdt
        pts[t, :, 4:8] = xyz_t
        pts[t, :, 8:11] = rgb.astype(np.int32)

    # save under sk_depth.avi/ subdir (same as baseline _pts.npy location, distinct name)
    save_dir = (PREFIX + "/Processed/" + train_or_test
                + "/" + rel.split("Video_data/", 1)[1] + "/" + SENSOR_DEPTH + ".avi")
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{str(idx).zfill(4)}_rgbd_label_{str(label).zfill(2)}.npy"
    np.save(save_path, pts)
    return idx, save_path, label


def main():
    splits = {
        "train": f"{PREFIX}/nvgesture_train_correct_cvpr2016_v2.lst",
        "test":  f"{PREFIX}/nvgesture_test_correct_cvpr2016_v2.lst",
    }

    for split_name, lst_path in splits.items():
        lines = open(lst_path).readlines()
        print(f"{split_name}: {len(lines)} clips")
        args_list = []
        for idx, line in enumerate(lines):
            rel, s, e, label = _parse_v2_line(line)
            args_list.append((idx, rel, s, e, label, split_name))

        with Pool(processes=max(1, cpu_count() // 2)) as pool:
            results = pool.map(process_one, args_list)

        list_path = f"{PREFIX}/Processed/{split_name}_rgbd_list.txt"
        with open(list_path, "w") as f:
            for r in results:
                if r is None:
                    continue
                idx, save_path, label = r
                f.write(f"{str(idx).zfill(4)}\t{save_path}\t{str(label).zfill(2)}\n")
        print(f"  wrote {list_path}")


if __name__ == "__main__":
    main()
