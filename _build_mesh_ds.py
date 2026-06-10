"""Build the dense depth-mesh dataset for ALL NVGesture clips (train + test).

For each clip listed in {train,test}_depth_list.txt: look up its depth frame range from the
official .lst, decode sk_depth.avi, sample 32 frames evenly over that range, take the largest
connected-component foreground per frame, and store the surface vertices (u,v,depth, int16)
ragged with per-frame offsets. Faces are recomputed at load (see mesh_ds.mesh_faces).

Writes {stem}_mesh.npz next to each {stem}_pts.npy. Resumable: skips clips already done.
Smoke mode: set env MESH_LIMIT=N to process only the first N missing clips."""
import os, re, time, numpy as np, cv2
os.chdir('/notebooks/Manta/experiments')

RAW = '/notebooks/Manta/dataset_full/nvGesture_v1.1/nvGesture_v1'
LSTS = [f'{RAW}/nvgesture_train_correct_cvpr2016_v2.lst',
        f'{RAW}/nvgesture_test_correct_cvpr2016_v2.lst']
LISTS = ['../dataset/Nvidia/Processed/train_depth_list.txt',
         '../dataset/Nvidia/Processed/test_depth_list.txt']
T = 32
LIMIT = int(os.environ.get('MESH_LIMIT', '0'))

ranges = {}
for lp in LSTS:
    for line in open(lp):
        m = re.search(r'path:\./Video_data/(\S+)', line)
        d = re.search(r'depth:sk_depth:(\d+):(\d+)', line)
        if m and d:
            ranges[m.group(1)] = (int(d.group(1)), int(d.group(2)))
print('ranges parsed', len(ranges), flush=True)


def largest_cc(mask):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return mask
    return lab == 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))


def frame_verts(g):
    m = largest_cc(g > 0); ys, xs = np.where(m)
    if len(ys) < 4:
        return np.zeros((0, 3), np.int16)
    return np.stack([xs, ys, g[ys, xs]], 1).astype(np.int16)   # u, v, depth


def clip_key(stem):
    parts = stem.split('/')
    i = next(k for k, p in enumerate(parts) if p.startswith('class_'))
    return parts[i] + '/' + parts[i + 1]


r = re.compile(r'[ \t\n\r:]+')
done = skip = err = 0; t0 = time.time()
for listfile in LISTS:
    for line in open(listfile):
        parts = r.split(line)
        if len(parts) < 3:
            continue
        stem = parts[1][1:-4]
        label = int(parts[-2])
        out = f'../dataset/{stem}_mesh.npz'
        if os.path.exists(out):
            skip += 1; continue
        key = clip_key(stem)
        if key not in ranges:
            err += 1; print('NORANGE', key, flush=True); continue
        F0, F1 = ranges[key]
        avi = f'{RAW}/Video_data/{key}/sk_depth.avi'
        cap = cv2.VideoCapture(avi); allf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            allf.append(fr[:, :, 0])
        cap.release()
        if not allf:
            err += 1; print('NOFRAMES', avi, flush=True); continue
        allf = np.array(allf)
        ids = np.clip(np.linspace(F0, F1, T).astype(int), 0, len(allf) - 1)
        vs = [frame_verts(allf[i].astype(np.int32)) for i in ids]
        vptr = np.zeros(T + 1, np.int32)
        for k in range(T):
            vptr[k + 1] = vptr[k] + len(vs[k])
        verts = np.concatenate(vs, 0).astype(np.int16) if vptr[-1] > 0 else np.zeros((0, 3), np.int16)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, verts=verts, vptr=vptr,
                            label=np.int64(label), frame_ids=ids.astype(np.int32))
        done += 1
        if done % 50 == 0:
            print('done %d skip %d err %d | %.0fs | last %s ~%d verts/frame'
                  % (done, skip, err, time.time() - t0, key, vptr[-1] // T), flush=True)
        if LIMIT and done >= LIMIT:
            print('LIMIT %d reached' % LIMIT, flush=True)
            print('FINISHED done %d skip %d err %d in %.0fs' % (done, skip, err, time.time() - t0), flush=True)
            raise SystemExit
print('FINISHED done %d skip %d err %d in %.0fs' % (done, skip, err, time.time() - t0), flush=True)
