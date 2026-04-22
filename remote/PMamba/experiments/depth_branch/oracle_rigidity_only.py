"""Oracle: PMamba (e110) vs RigidityOnlyClassifier (~12.66% solo)."""
import sys, os
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

import numpy as np
import torch

from models.motion import Motion
from nvidia_dataloader import NvidiaLoader
from depth_branch.model import RigidityOnlyClassifier
from depth_branch.dataloader import DepthVideoLoader

VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'stats'    # 'stats' (K=6) or 'pp' (sorted P=256)
ARG = sys.argv[2] if len(sys.argv) > 2 else 'best'
CKPT = 'best_model' if ARG == 'best' else f'epoch{ARG}_model'
WORKDIR = 'rigidity_only_pp' if VARIANT == 'pp' else 'rigidity_only'
RIG_DIM = 256 if VARIANT == 'pp' else 6
PER_POINT = VARIANT == 'pp'
PTS = 256
N_TTA = 3

pm = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()
cp = torch.load('work_dir/pmamba_branch/epoch110_model.pt', map_location='cpu')
pm.load_state_dict(cp.get('model_state_dict', cp), strict=False); pm.eval()

hidden = 128 if PER_POINT else 64
rm = RigidityOnlyClassifier(num_classes=25, rigidity_dim=RIG_DIM, hidden=hidden, lstm_layers=2, dropout=0.3).cuda()
cr = torch.load(f'work_dir/depth_branch/{WORKDIR}/{CKPT}.pt', map_location='cpu')
rm.load_state_dict(cr.get('model_state_dict', cr), strict=False); rm.eval()
print(f"loaded PMamba@e110 + rigidity_only[{VARIANT}]@{ARG}")

pml = NvidiaLoader(framerate=32, phase='test')
dpl = DepthVideoLoader(framerate=32, phase='test', img_size=112,
                       use_tops=False, use_rigidity=True,
                       rigidity_per_point=PER_POINT, rigidity_norm_scale=1.0)
n = len(pml)

def to_cuda(s):
    if isinstance(s, torch.Tensor): return s.unsqueeze(0).cuda()
    return torch.from_numpy(s).unsqueeze(0).cuda()

labels = np.zeros(n, int); pm_c = np.zeros(n, bool); r_c = np.zeros(n, bool)
pm_lg = []; r_lg = []
with torch.no_grad():
    for i in range(n):
        sp, lab, _ = pml[i]
        op = torch.stack([pm(to_cuda(sp)) for _ in range(N_TTA)]).mean(0)
        labels[i] = int(lab); pm_c[i] = (op.argmax(1).item() == int(lab)); pm_lg.append(op.cpu())
        (dt, rt), _, _ = dpl[i]
        od = rm((dt.unsqueeze(0).cuda(), rt.unsqueeze(0).cuda()))
        r_c[i] = (od.argmax(1).item() == int(lab)); r_lg.append(od.cpu())
        if i % 100 == 0: print(f"  {i}/{n}")

pm_lg = torch.cat(pm_lg, 0); r_lg = torch.cat(r_lg, 0)
pm_acc = pm_c.mean()*100; r_acc = r_c.mean()*100
oracle = pm_c | r_c
print()
print(f"PMamba:               {pm_acc:.2f}%")
print(f"rigidity_only[{VARIANT}]:   {r_acc:.2f}%")
print(f"Oracle:               {oracle.mean()*100:.2f}%")
print(f"Both correct:         {(pm_c & r_c).sum()}")
print(f"Only PMamba:          {(pm_c & ~r_c).sum()}")
print(f"Only rigidity:        {(~pm_c & r_c).sum()}")
print(f"Both wrong:           {(~pm_c & ~r_c).sum()}")
print(f"Headroom:             +{oracle.mean()*100 - pm_acc:.2f}pp")

pp = torch.softmax(pm_lg, 1); rp = torch.softmax(r_lg, 1)
lt = torch.tensor(labels, dtype=torch.long)
bfa = 0; ba = 0
for ai in range(0, 105, 5):
    a = ai/100
    fp = (a*pp + (1-a)*rp).argmax(1)
    acc = (fp==lt).sum().item()/n*100
    if acc > bfa: bfa = acc; ba = a
print(f"Best fusion: alpha={ba:.2f} -> {bfa:.2f}% (+{bfa-pm_acc:.2f}pp)")
