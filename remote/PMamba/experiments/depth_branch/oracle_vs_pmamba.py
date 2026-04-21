"""Oracle: PMamba (pts=256, epoch110) vs DepthCNNLSTM (v1 or v2) @ given epoch.

Usage: python depth_branch/oracle_vs_pmamba.py <version> <depth_epoch>
  version = v1 (depth 1ch) | v2 (depth+tops 4ch)
  depth_epoch = int | 'best'
Default: v1 best.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')

import numpy as np
import torch

from models.motion import Motion
from nvidia_dataloader import NvidiaLoader
from depth_branch.model import DepthCNNLSTM, DepthCNNLSTMQCC, DepthCNNLSTMPartQCC
from depth_branch.dataloader import DepthVideoLoader


VERSION = sys.argv[1] if len(sys.argv) > 1 else 'v1'
_ARG = sys.argv[2] if len(sys.argv) > 2 else 'best'
DEPTH_CKPT = 'best_model' if _ARG == 'best' else f'epoch{_ARG}_model'
DEPTH_TAG = _ARG
USE_TOPS = VERSION in ('v2', 'v3', 'v4')
IN_CHANNELS = 4 if USE_TOPS else 1
QCC_KIND = {'v3': 'global', 'v4': 'part'}.get(VERSION)
PTS = 256
N_TTA_PMAMBA = 3

# --- load PMamba ---
pm = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()
cp = torch.load('work_dir/pmamba_branch/epoch110_model.pt', map_location='cpu')
pm.load_state_dict(cp.get('model_state_dict', cp), strict=False); pm.eval()

# --- load DepthCNNLSTM (plain, global-QCC, or partwise-QCC) ---
if QCC_KIND == 'global':
    dm_cls = DepthCNNLSTMQCC
elif QCC_KIND == 'part':
    dm_cls = DepthCNNLSTMPartQCC
else:
    dm_cls = DepthCNNLSTM
dm_kwargs = dict(
    num_classes=25, in_channels=IN_CHANNELS, feat_dim=256,
    lstm_hidden=256, lstm_layers=2, bidirectional=True, dropout=0.3,
)
if QCC_KIND == 'global':
    dm_kwargs.update(dict(qcc_hidden=64, qcc_weight=0.1))
elif QCC_KIND == 'part':
    dm_kwargs.update(dict(qcc_hidden=64, qcc_weight=0.1, part_grid=(2, 3)))
dm = dm_cls(**dm_kwargs).cuda()
cd = torch.load(f'work_dir/depth_branch/{VERSION}/{DEPTH_CKPT}.pt', map_location='cpu')
dm.load_state_dict(cd.get('model_state_dict', cd), strict=False); dm.eval()
print(f"loaded PMamba@e110 + depth_{VERSION}@{DEPTH_TAG} (in_ch={IN_CHANNELS})")

# --- loaders ---
pml = NvidiaLoader(framerate=32, phase='test')
dpl = DepthVideoLoader(framerate=32, phase='test', img_size=112, use_tops=USE_TOPS)
n = len(pml)
assert len(dpl) == n, f"loader len mismatch: {n} vs {len(dpl)}"


def to_cuda(s):
    if isinstance(s, torch.Tensor): return s.unsqueeze(0).cuda()
    return torch.from_numpy(s).unsqueeze(0).cuda()


labels = np.zeros(n, int); pm_c = np.zeros(n, bool); d_c = np.zeros(n, bool)
pm_lg = []; d_lg = []
with torch.no_grad():
    for i in range(n):
        sp, lab, _ = pml[i]
        op = torch.stack([pm(to_cuda(sp)) for _ in range(N_TTA_PMAMBA)]).mean(0)
        labels[i] = int(lab); pm_c[i] = (op.argmax(1).item() == int(lab)); pm_lg.append(op.cpu())

        sd, _, _ = dpl[i]
        od = dm(sd.unsqueeze(0).cuda())
        d_c[i] = (od.argmax(1).item() == int(lab)); d_lg.append(od.cpu())

        if i % 100 == 0: print(f"  {i}/{n}")

pm_lg = torch.cat(pm_lg, 0); d_lg = torch.cat(d_lg, 0)
pm_acc = pm_c.mean() * 100; d_acc = d_c.mean() * 100
oracle = pm_c | d_c
print()
print(f"PMamba:         {pm_acc:.2f}%")
print(f"depth_{VERSION}@{DEPTH_TAG}:   {d_acc:.2f}%")
print(f"Oracle:         {oracle.mean()*100:.2f}%")
print(f"Both correct:   {(pm_c & d_c).sum()}")
print(f"Only PMamba:    {(pm_c & ~d_c).sum()}")
print(f"Only depth:     {(~pm_c & d_c).sum()}")
print(f"Both wrong:     {(~pm_c & ~d_c).sum()}")
print(f"Headroom:       +{oracle.mean()*100 - pm_acc:.2f}pp")

pp = torch.softmax(pm_lg, 1); dp = torch.softmax(d_lg, 1)
lt = torch.tensor(labels, dtype=torch.long)
bfa = 0; ba = 0
for ai in range(0, 105, 5):
    a = ai / 100
    fp = (a * pp + (1 - a) * dp).argmax(1)
    acc = (fp == lt).sum().item() / n * 100
    if acc > bfa: bfa = acc; ba = a
print(f"Best fusion: alpha={ba:.2f} -> {bfa:.2f}% (+{bfa - pm_acc:.2f}pp)")
