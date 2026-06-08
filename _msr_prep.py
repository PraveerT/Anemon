"""MSRAction3D depth .bin -> cached point-cloud sequences.
.bin = header[nframes, ncols=320, nrows=240] then int32 depth. Per frame: take
foreground (depth>0) pixels as (x=col, y=row, z=depth), sample N points; pick T
frames evenly; per-clip normalize (center + unit std). Cache all clips to one npz.
Labels: action a01..a20 -> 0..19; subject s01..s10. Cross-subject split standard.
"""
import os, glob, numpy as np

SRC = '/notebooks/wt9191/msr_raw/Depth'
T, N = 24, 512
rng = np.random.RandomState(0)


def read_bin(p):
    a = np.fromfile(p, dtype=np.int32)
    nf, nc, nr = int(a[0]), int(a[1]), int(a[2])
    d = a[3:3 + nf * nr * nc].reshape(nf, nr, nc).astype(np.float32)
    return d


def clip_pc(p):
    d = read_bin(p)
    nf = d.shape[0]
    fidx = np.linspace(0, nf - 1, T).astype(int)
    frames = []
    for fi in fidx:
        dm = d[fi]
        ys, xs = np.nonzero(dm)
        if len(xs) == 0:
            frames.append(np.zeros((N, 3), np.float32)); continue
        zs = dm[ys, xs]
        pts = np.stack([xs, ys, zs], 1).astype(np.float32)
        rep = len(pts) < N
        sel = rng.choice(len(pts), N, replace=rep)
        frames.append(pts[sel])
    seq = np.stack(frames)                          # (T,N,3)
    flat = seq.reshape(-1, 3)
    seq = (seq - flat.mean(0)) / (flat.std() + 1e-6)
    return seq.astype(np.float32)


files = sorted(glob.glob(os.path.join(SRC, '*.bin')))
X, A, S = [], [], []
for i, p in enumerate(files):
    b = os.path.basename(p)
    act = int(b[1:3]) - 1
    sub = int(b[5:7])
    try:
        X.append(clip_pc(p)); A.append(act); S.append(sub)
    except Exception as e:
        print('skip', b, e, flush=True)
    if (i + 1) % 100 == 0:
        print('processed', i + 1, '/', len(files), flush=True)
X = np.stack(X); A = np.array(A); S = np.array(S)
np.savez_compressed('/notebooks/wt9191/experiments/msr_pc.npz',
                    X=X, action=A, subject=S)
print('SAVED', X.shape, 'actions', len(set(A.tolist())), 'subjects', sorted(set(S.tolist())), flush=True)
print('DONE', flush=True)
