"""Build alternate point-cloud dataset using complement frames.

Original nvidia_process.py uses key_frame_sampling(N, 32) which picks 32
frames from the N (~80) trimmed gesture frames. This leaves ~48 unused.

This script picks 32 frames from the COMPLEMENT (i.e., the 48 not picked
originally), uniformly spaced, and processes them through the same Otsu/
points pipeline. Output saved as `_alt_pts.npy` next to the original.

Train-only — test set is unchanged.
"""
import re
import cv2
import copy
import sys
import utils
import numpy as np
from multiprocessing import Pool, cpu_count


def process_one(args):
    idx, npy_path_full, pts_size = args
    parts = re.split(r"[ \t\n\r:]+", npy_path_full)
    npy_path = parts[1]
    label = parts[2]
    full = npy_path                                      # already starts with ./Nvidia/...
    # nvidia_process.py uses np.load with "../dataset/<rel_path[1:]>_pts.npy" but the depth.npy itself
    # lives at "../dataset/<rel_path[1:]>". We're mirroring that.
    src_path = "../dataset/" + npy_path[1:]              # the 80-frame depth.npy
    try:
        depth_video = np.load(src_path)
    except Exception as e:
        print(f"[fail] {idx} {src_path}: {e}")
        return None
    n = len(depth_video)
    orig_idx = utils.key_frame_sampling(n, 32)
    orig_set = set(orig_idx)
    complement = [i for i in range(n) if i not in orig_set]
    if len(complement) < 32:
        # too few unused frames (very short clip); fall back to original sampling
        alt_real = utils.key_frame_sampling(n, 32)
    else:
        alt_local = utils.key_frame_sampling(len(complement), 32)
        alt_real = [complement[i] for i in alt_local]
    depth_alt = depth_video[alt_real]

    pts = np.zeros((32, pts_size, 8), dtype=int)
    for i in range(32):
        frame = depth_alt[i, :, :, 0]
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        _, thresh = cv2.threshold(frame_blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = utils.save_largest_label(thresh)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel)
        pts[i, :, :4] = utils.points_sampling(utils.generate_points(frame * thresh, i), pts_size)
        pts[i, :, 4:8] = utils.uvd2xyz_sherc(copy.deepcopy(pts[i, :, :4]))

    save_path = src_path[:-4] + "_alt_pts"
    np.save(save_path, pts)
    return idx, npy_path, label


def main():
    pts_size = 512
    prefix = "./Nvidia"
    train_list_path = f"{prefix}/Processed/train_depth_list.txt"
    train_lines = open(train_list_path).readlines()
    print(f"processing {len(train_lines)} alt train clips")

    args_list = [(i, line, pts_size) for i, line in enumerate(train_lines)]
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_one, args_list)

    # write the alt list pointing to the new files
    alt_list_path = f"{prefix}/Processed/train_depth_alt_list.txt"
    with open(alt_list_path, "w") as f:
        for r in results:
            if r is None:
                continue
            idx, npy_path, label = r
            alt_path = npy_path[:-4] + "_alt.npy"
            f.write(f"{idx}\t{alt_path}\t{label}\n")
    print(f"wrote {alt_list_path}: {sum(1 for r in results if r)} entries")

    # produce combined 2x list
    combined = open(train_list_path).readlines() + open(alt_list_path).readlines()
    combined_path = f"{prefix}/Processed/train_depth_2x_list.txt"
    with open(combined_path, "w") as f:
        f.writelines(combined)
    print(f"wrote {combined_path}: {len(combined)} entries")


if __name__ == "__main__":
    main()
