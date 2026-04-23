"""Oracle: PMamba vs RigidityAttentionBearingQCCFeatureMotion."""
import sys, os
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

import numpy as np
import torch

from models.motion import Motion
from models.reqnn_motion import RigidityAttentionBearingQCCFeatureMotion
from nvidia_dataloader import NvidiaLoader, NvidiaQuaternionQCCParityLoader


ARG = sys.argv[1] if len(sys.argv) > 1 else 'best'
CKPT = 'best_model' if ARG == 'best' else f'epoch{ARG}_model'
TAG = ARG
PTS = 256
N_TTA = 3

pm = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()
cp = torch.load('work_dir/pmamba_branch/epoch110_model.pt', map_location='cpu')
pm.load_state_dict(cp.get('model_state_dict', cp), strict=False); pm.eval()

rm = RigidityAttentionBearingQCCFeatureMotion(
    num_classes=25, pts_size=PTS, hidden_dims=[64, 256], dropout=0.05,
    edgeconv_k=20, merge_eps=1e-6, so3_weight=0.0, rotation_sigma=0.3,
    bearing_knn_k=10, qcc_weight=0.1, qcc_variant='contrastive',
).cuda()
cr = torch.load(f'work_dir/quaternion_rigidity_attn_v1/{CKPT}.pt', map_location='cpu')
sd = cr.get('model_state_dict', cr)
rm.load_state_dict(sd, strict=False); rm.eval()
print(f"loaded PMamba@e110 + attn_v1@{TAG} | tau={rm.rig_tau.item():.4f} alpha={rm.rig_alpha.item():.4f}")

pml = NvidiaLoader(framerate=32, phase='test')
ql = NvidiaQuaternionQCCParityLoader(framerate=32, phase='test', return_correspondence=True, assignment_mode='hungarian')
n = len(pml)

def to_cuda(s):
    if isinstance(s, dict):
        return {k: (v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else
                    torch.from_numpy(v).unsqueeze(0).cuda() if hasattr(v, 'shape') else v)
                for k, v in s.items()}
    if isinstance(s, torch.Tensor): return s.unsqueeze(0).cuda()
    return torch.from_numpy(s).unsqueeze(0).cuda()

labels = np.zeros(n, int); pm_c = np.zeros(n, bool); r_c = np.zeros(n, bool)
pm_lg = []; r_lg = []
with torch.no_grad():
    for i in range(n):
        sp, lab, _ = pml[i]
        op = torch.stack([pm(to_cuda(sp)) for _ in range(N_TTA)]).mean(0)
        labels[i] = int(lab); pm_c[i] = (op.argmax(1).item() == int(lab)); pm_lg.append(op.cpu())
        sq, _, _ = ql[i]
        oq = torch.stack([rm(to_cuda(sq)) for _ in range(N_TTA)]).mean(0)
        r_c[i] = (oq.argmax(1).item() == int(lab)); r_lg.append(oq.cpu())
        if i % 100 == 0: print(f"  {i}/{n}")

pm_lg = torch.cat(pm_lg, 0); r_lg = torch.cat(r_lg, 0)
pm_acc = pm_c.mean() * 100; r_acc = r_c.mean() * 100
oracle = pm_c | r_c
print()
print(f"PMamba:           {pm_acc:.2f}%")
print(f"attn_v1@{TAG}:   {r_acc:.2f}%")
print(f"Oracle:           {oracle.mean()*100:.2f}%")
print(f"Both correct:     {(pm_c & r_c).sum()}")
print(f"Only PMamba:      {(pm_c & ~r_c).sum()}")
print(f"Only attn:        {(~pm_c & r_c).sum()}")
print(f"Both wrong:       {(~pm_c & ~r_c).sum()}")
print(f"Headroom:         +{oracle.mean()*100 - pm_acc:.2f}pp")

pp = torch.softmax(pm_lg, 1); rp = torch.softmax(r_lg, 1)
lt = torch.tensor(labels, dtype=torch.long)
bfa = 0; ba = 0
for ai in range(0, 105, 5):
    a = ai / 100
    fp = (a * pp + (1 - a) * rp).argmax(1)
    acc = (fp == lt).sum().item() / n * 100
    if acc > bfa: bfa = acc; ba = a
print(f"Best fusion: alpha={ba:.2f} -> {bfa:.2f}% (+{bfa - pm_acc:.2f}pp)")
