"""Derive the mirror-twin class permutation P from the 91.08 teacher.
Mirror each training gesture (flip u about per-frame centroid), run the teacher,
and record which class the mirrored gesture looks like. P[i] = argmax_j count of
teacher(mirror(class i)) == j. A reflection-equivariant model should satisfy
f(mirror x) = P f(x). Saves P.npy + reports involution quality.
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


def mirror(x):                      # x (B,T,N,4); flip u (ch0) about per-frame centroid
    x = x.clone()
    u = x[..., 0]
    um = u.mean(dim=2, keepdim=True)
    x[..., 0] = 2 * um - u
    return x


ds = NvidiaLoader(framerate=32, phase='train', datatype='depth')
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
C = 25
cnt = np.zeros((C, C), dtype=np.int64)        # cnt[true, pred_on_mirror]
with torch.no_grad():
    for x, y, _ in dl:
        x = x.to(dev)
        p = m(mirror(x)).argmax(1).cpu().numpy()
        for t, pp in zip(np.asarray(y), p):
            cnt[int(t), int(pp)] += 1

P = cnt.argmax(1)
row = cnt.sum(1) + 1e-9
peak = cnt[np.arange(C), P] / row             # how dominant the twin is per class
invol = int(sum(P[P[i]] == i for i in range(C)))
selfm = int(sum(P[i] == i for i in range(C)))
np.save('/notebooks/wt9191/experiments/mirror_P.npy', P)
print('P =', P.tolist(), flush=True)
print('per-class twin purity (min/mean):', round(float(peak.min()), 2), round(float(peak.mean()), 2), flush=True)
print('involution P[P[i]]==i :', invol, '/25   self-mapped (symmetric):', selfm, flush=True)
print('SAVED mirror_P.npy', flush=True)
