"""Cache step for the mesh-GNN: regrid each clip's dense mesh verts to ~BUDGET points/frame
on a common per-clip lattice (stride chosen from mean density), store compact int16 (u,v,depth)
+ per-frame offsets + label. Edges (mesh + kNN) are rebuilt in RAM by the trainer, so the
cache stays lean and budget/k can change without recaching. One mesh_cache_{split}.pt per
split, in the SAME order as {split}_depth_list.txt (so test preds align with hard_idx.npy)."""
import os, re, numpy as np, torch
os.chdir('/notebooks/Manta/experiments')

BUDGET = 512
LIMIT = int(os.environ.get('CACHE_LIMIT', '0'))
LISTS = [('train', '../dataset/Nvidia/Processed/train_depth_list.txt'),
         ('test', '../dataset/Nvidia/Processed/test_depth_list.txt')]
r = re.compile(r'[ \t\n\r:]+')


def regrid(frames):
    allv = np.concatenate([f for f in frames if len(f)], 0) if any(len(f) for f in frames) else np.zeros((0, 3))
    if len(allv) == 0:
        return None, 1
    umin, vmin = int(allv[:, 0].min()), int(allv[:, 1].min())
    meanN = np.mean([len(f) for f in frames if len(f)])
    s = max(1, int(round(np.sqrt(meanN / BUDGET))))
    out = []
    for f in frames:
        if len(f) < 4:
            out.append(np.zeros((0, 3), np.int16)); continue
        u = f[:, 0] - umin; v = f[:, 1] - vmin; d = f[:, 2]
        if s > 1:
            k = (u % s == 0) & (v % s == 0); u, v, d = u[k] // s, v[k] // s, d[k]
        key = u.astype(np.int64) * 100000 + v
        _, idx = np.unique(key, return_index=True)
        out.append(np.stack([u[idx], v[idx], d[idx]], 1).astype(np.int16))
    return out, s


for split, lf in LISTS:
    clips = []; strides = []
    for line in open(lf):
        p = r.split(line)
        if len(p) < 3:
            continue
        stem = p[1][1:-4]; label = int(p[-2])
        z = np.load(f'../dataset/{stem}_mesh.npz'); verts = z['verts']; vptr = z['vptr']
        frames = [verts[vptr[t]:vptr[t + 1]].astype(np.int64) for t in range(len(vptr) - 1)]
        out, s = regrid(frames); strides.append(s)
        if out is None:
            clips.append({'x': np.zeros((0, 3), np.int16), 'fptr': np.zeros(33, np.int32), 'y': label}); continue
        fptr = np.zeros(len(out) + 1, np.int32)
        for t in range(len(out)):
            fptr[t + 1] = fptr[t] + len(out[t])
        x = np.concatenate(out, 0).astype(np.int16) if fptr[-1] > 0 else np.zeros((0, 3), np.int16)
        clips.append({'x': x, 'fptr': fptr, 'y': label})
        if LIMIT and len(clips) >= LIMIT:
            break
    torch.save(clips, f'mesh_cache_{split}.pt')
    nv = [c['fptr'][-1] for c in clips]
    print(split, 'clips', len(clips), '| stride', int(np.median(strides)),
          '| verts/clip mean', int(np.mean(nv)), '~per frame', int(np.mean(nv) / 32),
          '| max verts/frame', int(max((np.diff(c['fptr']).max() if c['fptr'][-1] > 0 else 0) for c in clips)), flush=True)
print('DONE', flush=True)
