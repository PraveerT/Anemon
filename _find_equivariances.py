"""Scan candidate symmetries for clean twin structure (the prerequisite for a
useful equivariance). Same method that worked for reflection: apply transform ->
teacher penultimate features -> mutual nearest-twin matching. Reports, per
transform, the mutual cross-pairs and how 'twinned' vs 'invariant' it is.
Transforms: time-reversal, 180-rot (uv), vertical-flip(v), speed-2x (temporal
subsample+repeat). Train data only.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch
from torch.utils.data import DataLoader
from nvidia_dataloader import NvidiaLoader
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead

dev = 'cuda'; C = 25
margs = dict(num_classes=25, pts_size=172, topk=8, knn=[32, 24, 48, 24],
             multi_scale_num_scales=5, lxl_hidden_dim=256, lxl_mlp_dim=512,
             lxl_num_layers=4, lxl_dropout=0.3, lxl_bidirectional=True,
             lxl_residual_scale=0.7)
m = MotionCleanestLinXLQuatHead(**margs).to(dev)
ck = torch.load('/notebooks/Manta/experiments/work_dir/pgcnet_quat_head/best_model.pt', map_location=dev)
sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
m.load_state_dict(sd, strict=False); m.eval(); m.pts_size = 172


def feat(x):
    coords = m._sample_points(x)
    fea3 = m._encode_sampled_points(coords)
    return m.global_bn(m.pool5(m.stage5(fea3))).flatten(1)


def t_reverse(x):                         # reverse frame order (time-reversal)
    return torch.flip(x, dims=[1])


def t_rot180(x):                          # 180 deg in u,v about per-clip center
    x = x.clone()
    for ch in (0, 1):
        a = x[..., ch]; am = a.mean(dim=(1, 2), keepdim=True)
        x[..., ch] = 2 * am - a
    return x


def t_vflip(x):                           # vertical flip (v=ch1) fixed axis
    x = x.clone()
    v = x[..., 1]; vm = v.mean(dim=(1, 2), keepdim=True)
    x[..., 1] = 2 * vm - v
    return x


def t_speed(x):                           # 2x speed: take every other frame, repeat
    idx = (torch.arange(x.shape[1], device=x.device) // 2) % x.shape[1]
    return x[:, idx]


TRANSFORMS = {'time_reverse': t_reverse, 'rot180': t_rot180,
              'vflip': t_vflip, 'speed2x': t_speed}

ds = NvidiaLoader(framerate=32, phase='train', datatype='depth')
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

realf = torch.zeros(C, 1024, device=dev); nc = torch.zeros(C, device=dev)
tf = {k: torch.zeros(C, 1024, device=dev) for k in TRANSFORMS}
with torch.no_grad():
    for x, y, _ in dl:
        x = x.to(dev); y = y.to(dev)
        fr = feat(x)
        ff = {k: feat(fn(x)) for k, fn in TRANSFORMS.items()}
        for k in range(len(y)):
            c = int(y[k]); realf[c] += fr[k]; nc[c] += 1
            for kk in TRANSFORMS:
                tf[kk][c] += ff[kk][k]
realf /= nc[:, None].clamp_min(1)
rn = torch.nn.functional.normalize(realf, dim=1)
for kk in TRANSFORMS:
    mn = torch.nn.functional.normalize(tf[kk] / nc[:, None].clamp_min(1), dim=1)
    S = (mn @ rn.t()).cpu().numpy()
    twin = S.argmax(1)
    selfsim = float(np.mean(S[np.arange(C), np.arange(C)]))
    mutual = sorted(set(tuple(sorted((i, int(twin[i])))) for i in range(C)
                        if twin[i] != i and twin[int(twin[i])] == i))
    nself = int(sum(twin[i] == i for i in range(C)))
    print('%-12s self-sim %.3f  self-mapped %d/25  MUTUAL cross-pairs: %s'
          % (kk, selfsim, nself, mutual), flush=True)
print('DONE', flush=True)
