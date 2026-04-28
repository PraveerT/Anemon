"""Build per-frame dual-quaternion cache for all NV clips.

For each clip:
  - load _pts.npy  (T, N, 8)  — xyz at channels 4..7
  - per frame pair (t-1, t): Kabsch SVD → R, T
  - encode SE(3) as dual quaternion: qr (4) + qd (4)
  - frame 0: identity (qr=(1,0,0,0), qd=0)
  - save (T, 8) as <pts_path>_dq.npy

Channels of dq: [qr_w, qr_x, qr_y, qr_z, qd_w, qd_x, qd_y, qd_z]
"""
import os
import sys
import glob
import numpy as np
import multiprocessing as mp


PROC = "/notebooks/PMamba/dataset/Nvidia/Processed"


def kabsch(src, tgt):
    """Rigid alignment: R src + t = tgt. Both (N, 3). Returns (R, t)."""
    src_mu = src.mean(0)
    tgt_mu = tgt.mean(0)
    sc = src - src_mu
    tc = tgt - tgt_mu
    H = sc.T @ tc
    U, _, Vt = np.linalg.svd(H + 1e-9 * np.eye(3))
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ U.T
    t = tgt_mu - R @ src_mu
    return R, t


def rot_to_quat(R):
    """3x3 rotation -> unit quat (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def build_dq_for_clip(pts_path):
    out_path = pts_path.replace("_pts.npy", "_dq.npy")
    if os.path.exists(out_path):
        return out_path, "skip"
    arr = np.load(pts_path)
    xyz = arr[..., 4:7].astype(np.float64)  # (T, N, 3)
    T = xyz.shape[0]
    dq = np.zeros((T, 8), dtype=np.float32)
    dq[0, 0] = 1.0  # identity quat for frame 0
    for ti in range(1, T):
        R, t = kabsch(xyz[ti - 1], xyz[ti])
        qr = rot_to_quat(R)
        t_quat = np.array([0.0, t[0], t[1], t[2]], dtype=np.float32)
        qd = 0.5 * quat_mul(t_quat, qr)
        dq[ti, :4] = qr
        dq[ti, 4:] = qd
    np.save(out_path, dq)
    return out_path, "wrote"


def main():
    paths = []
    for split in ("train", "test"):
        for f in glob.iglob(f"{PROC}/{split}/class_*/subject*_r*/sk_depth.avi/*_pts.npy"):
            paths.append(f)
    print(f"clips: {len(paths)}")

    n_workers = max(1, mp.cpu_count() // 2)
    written = 0
    skipped = 0
    with mp.Pool(n_workers) as pool:
        for i, (out_path, status) in enumerate(pool.imap_unordered(build_dq_for_clip, paths, chunksize=4)):
            if status == "wrote":
                written += 1
            else:
                skipped += 1
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(paths)} (wrote {written}, skipped {skipped})")
    print(f"done. wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
