"""Quick oracle: PMamba vs velpolar epoch90 checkpoint only."""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
import numpy as np
from models.motion import Motion
from models.reqnn_motion import VelocityPolarBearingQCCFeatureMotion
from nvidia_dataloader import NvidiaLoader, NvidiaQuaternionQCCParityLoader

PTS = 256
N_TTA = 3

pmamba = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()
cp = torch.load('work_dir/pmamba_branch/epoch110_model.pt', map_location='cpu')
pmamba.load_state_dict(cp.get('model_state_dict', cp), strict=False)
pmamba.eval()

vp = VelocityPolarBearingQCCFeatureMotion(
    num_classes=25, pts_size=PTS, hidden_dims=[64, 256], dropout=0.05,
    edgeconv_k=20, merge_eps=1e-6, so3_weight=0.0, rotation_sigma=0.3,
    bearing_knn_k=10, qcc_weight=0.1, qcc_variant='contrastive',
).cuda()
cq = torch.load('work_dir/quaternion_branch_v2_velpolar/epoch90_model.pt', map_location='cpu')
vp.load_state_dict(cq.get('model_state_dict', cq), strict=False)
vp.eval()
print("both loaded")

pml = NvidiaLoader(framerate=32, phase='test')
ql = NvidiaQuaternionQCCParityLoader(framerate=32, phase='test', return_correspondence=True)
n = len(pml)

def to_cuda(s):
    if isinstance(s, dict):
        return {k: (v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else
                    torch.from_numpy(v).unsqueeze(0).cuda() if hasattr(v, 'shape') else v)
                for k, v in s.items()}
    if isinstance(s, torch.Tensor): return s.unsqueeze(0).cuda()
    return torch.from_numpy(s).unsqueeze(0).cuda()

labels = np.zeros(n, int); pm_c = np.zeros(n, bool); q_c = np.zeros(n, bool)
pm_lg = []; q_lg = []
with torch.no_grad():
    for i in range(n):
        sp, lab, _ = pml[i]
        op = torch.stack([pmamba(to_cuda(sp)) for _ in range(N_TTA)]).mean(0)
        labels[i] = int(lab); pm_c[i] = (op.argmax(1).item() == int(lab)); pm_lg.append(op.cpu())
        sq, _, _ = ql[i]
        oq = torch.stack([vp(to_cuda(sq)) for _ in range(N_TTA)]).mean(0)
        q_c[i] = (oq.argmax(1).item() == int(lab)); q_lg.append(oq.cpu())
        if i % 100 == 0: print(f"  {i}/{n}")

pm_lg = torch.cat(pm_lg, 0); q_lg = torch.cat(q_lg, 0)
pm_acc = pm_c.mean() * 100; q_acc = q_c.mean() * 100
oracle = pm_c | q_c
print()
print(f"PMamba:    {pm_acc:.2f}%  ({pm_c.sum()}/{n})")
print(f"velpolar90:{q_acc:.2f}%  ({q_c.sum()}/{n})")
print(f"Oracle:    {oracle.mean()*100:.2f}%  ({oracle.sum()}/{n})")
print(f"Both correct: {(pm_c & q_c).sum()}")
print(f"Only PMamba:  {(pm_c & ~q_c).sum()}")
print(f"Only velpolar:{(~pm_c & q_c).sum()}")
print(f"Both wrong:   {(~pm_c & ~q_c).sum()}")
print(f"Headroom: +{oracle.mean()*100 - pm_acc:.2f}pp")

pp = torch.softmax(pm_lg, 1); qp = torch.softmax(q_lg, 1)
lt = torch.tensor(labels, dtype=torch.long)
bfa = 0; ba = 0
for ai in range(0, 105, 5):
    a = ai / 100
    fp = (a * pp + (1 - a) * qp).argmax(1)
    acc = (fp == lt).sum().item() / n * 100
    if acc > bfa: bfa = acc; ba = a
print(f"Best fusion: α={ba:.2f} -> {bfa:.2f}% (+{bfa - pm_acc:.2f}pp)")
