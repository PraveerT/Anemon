"""Loader for the dense depth-mesh dataset (one {stem}_mesh.npz per clip).

Each file stores, for 32 frames, the largest-connected-component foreground surface
vertices (u, v, depth) as int16 plus per-frame offsets `vptr`. The triangle faces are
the grid triangulation of those (u,v) lattice points (each foreground 2x2 quad -> 2
triangles, pruned where the depth jump is large); they are deterministic from the
vertices, so they are recomputed here rather than stored. This means each file doubles
as a full-resolution point cloud (verts) and a mesh (verts + mesh_faces(verts))."""
import numpy as np


def mesh_faces(verts, depth_jump=20):
    """Rebuild the triangle faces (M,3 int32) for one frame's verts (N,3 = u,v,depth)."""
    if len(verts) < 4:
        return np.zeros((0, 3), np.int32)
    u = verts[:, 0].astype(np.int64); v = verts[:, 1].astype(np.int64)
    umin, vmin = int(u.min()), int(v.min())
    W = int(u.max()) - umin + 2; H = int(v.max()) - vmin + 2
    idx = -np.ones((H, W), np.int64); idx[v - vmin, u - umin] = np.arange(len(verts))
    m = idx >= 0
    val = m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1] & m[1:, 1:]; rr, cc = np.where(val)
    f = np.concatenate([np.stack([idx[rr, cc], idx[rr, cc + 1], idx[rr + 1, cc + 1]], 1),
                        np.stack([idx[rr, cc], idx[rr + 1, cc + 1], idx[rr + 1, cc]], 1)], 0).astype(np.int32)
    d = verts[:, 2].astype(np.int32); a, b, c = d[f[:, 0]], d[f[:, 1]], d[f[:, 2]]
    dj = np.stack([abs(a - b), abs(b - c), abs(c - a)], 1).max(1)
    return f[dj <= depth_jump]


def load_clip(path, with_faces=True):
    """Return (frames, label, frame_ids). frames = list of (verts(N,3), faces(M,3)|None)."""
    z = np.load(path)
    verts, vptr = z['verts'], z['vptr']
    frames = []
    for t in range(len(vptr) - 1):
        v = verts[vptr[t]:vptr[t + 1]]
        frames.append((v, mesh_faces(v) if with_faces else None))
    return frames, int(z['label']), z['frame_ids']
