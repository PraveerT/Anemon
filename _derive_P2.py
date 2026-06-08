"""Find a TRUSTWORTHY mirror-twin map. Fix: mirror about a FIXED per-clip axis
(whole trajectory flips -> a real left<->right gesture), and match twins by teacher
PENULTIMATE-FEATURE cosine similarity (robust; no argmax confusion-sink). Reports
mutual cross-pairs (the real chirality twins) with purity. Train data only.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, torch
from torch.utils.data import DataLoader
from nvidia_dataloader import NvidiaLoader
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead

dev = 'cuda'
margs = dict(num_classes=25, pts_size=172, topk=8, knn=[32, 24, 48, 24],
             multi_scale_num_scales=5, lxl_hidden_dim=256, lxl_mlp_dim=512,
             lxl_num_layers=4, lxl_dropout=0.3, lxl_bidirectional=True,
             lxl_residual_scale=0.7)
m = MotionCleanestLinXLQuatHead(**margs).to(dev)
ck = torch.load('/notebooks/Manta/experiments/work_dir/pgcnet_quat_head/best_model.pt', map_location=dev)
sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
m.load_state_dict(sd, strict=False); m.eval(); m.pts_size = 172
C = 25


def mirror_fixed(x):                 # flip u about ONE per-clip axis (whole traj)
    x = x.clone()
    u = x[..., 0]
    um = u.mean(dim=(1, 2), keepdim=True)            # clip-constant axis
    x[..., 0] = 2 * um - u
    return x


def feat(x):                         # teacher penultimate (B,1024), eval->linspace
    coords = m._sample_points(x)
    fea3 = m._encode_sampled_points(coords)
    g = m.global_bn(m.pool5(m.stage5(fea3))).flatten(1)
    return g


def logits_pred(x):
    return m(x).argmax(1)


ds = NvidiaLoader(framerate=32, phase='train', datatype='depth')
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
realf = torch.zeros(C, 1024, device=dev); mirf = torch.zeros(C, 1024, device=dev)
nc = torch.zeros(C, device=dev)
cnt = np.zeros((C, C), np.int64)
with torch.no_grad():
    for x, y, _ in dl:
        x = x.to(dev); y = y.to(dev)
        xm = mirror_fixed(x)
        fr = feat(x); fm = feat(xm)
        pm = logits_pred(xm).cpu().numpy()
        for k in range(len(y)):
            c = int(y[k]); realf[c] += fr[k]; mirf[c] += fm[k]; nc[c] += 1
            cnt[c, int(pm[k])] += 1
realf /= nc[:, None].clamp_min(1); mirf /= nc[:, None].clamp_min(1)
rn = torch.nn.functional.normalize(realf, dim=1)
mn = torch.nn.functional.normalize(mirf, dim=1)
S = (mn @ rn.t()).cpu().numpy()                      # S[i,j]=sim(mirror i, real j)
twin = S.argmax(1)
# self-similarity vs twin-similarity: is the mirror closer to another class than self?
selfsim = S[np.arange(C), np.arange(C)]
twinsim = S[np.arange(C), twin]
mutual = [(i, int(twin[i])) for i in range(C)
          if twin[i] != i and twin[int(twin[i])] == i]
argmax_twin = cnt.argmax(1)
print('feature-sim twin[i]      :', twin.tolist(), flush=True)
print('argmax-logit twin[i]     :', argmax_twin.tolist(), flush=True)
print('MUTUAL cross-pairs (feat):', sorted(set(tuple(sorted(p)) for p in mutual)), flush=True)
print('classes where mirror!=self (twinsim>selfsim+0.02):',
      [i for i in range(C) if twin[i] != i and twinsim[i] > selfsim[i] + 0.02], flush=True)
np.save('/notebooks/wt9191/experiments/mirror_P2.npy', twin)
print('saved mirror_P2.npy  (mean self-sim %.3f)' % float(selfsim.mean()), flush=True)
