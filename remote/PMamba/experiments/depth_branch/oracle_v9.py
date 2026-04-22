"""Oracle: PMamba (e110) vs depth_v9 (CE-reweighted CNN-LSTM)."""
import sys, os
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

import numpy as np
import torch

from models.motion import Motion
from nvidia_dataloader import NvidiaLoader
from depth_branch.model import DepthCNNLSTM
from depth_branch.dataloader import DepthVideoLoader

ARG = sys.argv[1] if len(sys.argv) > 1 else 'best'
CKPT = 'best_model' if ARG == 'best' else f'epoch{ARG}_model'
TAG = ARG
PTS = 256
N_TTA = 3

pm = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()
cp = torch.load('work_dir/pmamba_branch/epoch110_model.pt', map_location='cpu')
pm.load_state_dict(cp.get('model_state_dict', cp), strict=False); pm.eval()

dm = DepthCNNLSTM(
    num_classes=25, in_channels=4, feat_dim=256,
    lstm_hidden=256, lstm_layers=2, bidirectional=True, dropout=0.3,
    rigidity_dim=0, rigidity_aux_dim=0, clip_reweight_beta=2.0,
).cuda()
cd = torch.load(f'work_dir/depth_branch/v9/{CKPT}.pt', map_location='cpu')
dm.load_state_dict(cd.get('model_state_dict', cd), strict=False); dm.eval()
print(f"loaded PMamba@e110 + depth_v9@{TAG}")

pml = NvidiaLoader(framerate=32, phase='test')
dpl = DepthVideoLoader(framerate=32, phase='test', img_size=112, use_tops=True, use_rigidity=True, rigidity_norm_scale=1.0)
n = len(pml)

def to_cuda(s):
    if isinstance(s, torch.Tensor): return s.unsqueeze(0).cuda()
    return torch.from_numpy(s).unsqueeze(0).cuda()

labels = np.zeros(n, int); pm_c = np.zeros(n, bool); d_c = np.zeros(n, bool)
pm_lg = []; d_lg = []
with torch.no_grad():
    for i in range(n):
        sp, lab, _ = pml[i]
        op = torch.stack([pm(to_cuda(sp)) for _ in range(N_TTA)]).mean(0)
        labels[i] = int(lab); pm_c[i] = (op.argmax(1).item() == int(lab)); pm_lg.append(op.cpu())
        (dt, rt), _, _ = dpl[i]
        od = dm((dt.unsqueeze(0).cuda(), rt.unsqueeze(0).cuda()))
        d_c[i] = (od.argmax(1).item() == int(lab)); d_lg.append(od.cpu())
        if i % 100 == 0: print(f"  {i}/{n}")

pm_lg = torch.cat(pm_lg, 0); d_lg = torch.cat(d_lg, 0)
pm_acc = pm_c.mean()*100; d_acc = d_c.mean()*100
oracle = pm_c | d_c
print()
print(f"PMamba:           {pm_acc:.2f}%")
print(f"depth_v9@{TAG}:   {d_acc:.2f}%")
print(f"Oracle:           {oracle.mean()*100:.2f}%")
print(f"Both correct:     {(pm_c & d_c).sum()}")
print(f"Only PMamba:      {(pm_c & ~d_c).sum()}")
print(f"Only depth:       {(~pm_c & d_c).sum()}")
print(f"Both wrong:       {(~pm_c & ~d_c).sum()}")
print(f"Headroom:         +{oracle.mean()*100 - pm_acc:.2f}pp")

pp = torch.softmax(pm_lg, 1); dp = torch.softmax(d_lg, 1)
lt = torch.tensor(labels, dtype=torch.long)
bfa = 0; ba = 0
for ai in range(0, 105, 5):
    a = ai/100
    fp = (a*pp + (1-a)*dp).argmax(1)
    acc = (fp==lt).sum().item()/n*100
    if acc > bfa: bfa = acc; ba = a
print(f"Best fusion: alpha={ba:.2f} -> {bfa:.2f}% (+{bfa-pm_acc:.2f}pp)")
