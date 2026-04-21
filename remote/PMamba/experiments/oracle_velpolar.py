"""Oracle: PMamba vs VelocityPolar (velpolar). Runs all saved checkpoints."""
import sys, glob
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
import numpy as np
from models.motion import Motion
from models.reqnn_motion import VelocityPolarBearingQCCFeatureMotion
from nvidia_dataloader import NvidiaLoader, NvidiaQuaternionQCCParityLoader

PTS = 256
N_TTA = 3
PMAMBA_CKPT = 'work_dir/pmamba_branch/epoch110_model.pt'
ckpts = sorted(
    glob.glob('work_dir/quaternion_branch_v2_velpolar/epoch*_model.pt'),
    key=lambda p: int(p.split('epoch')[1].split('_')[0]),
)
print(f"Checkpoints: {[c.split('/')[-1] for c in ckpts]}")

pmamba = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()
cp = torch.load(PMAMBA_CKPT, map_location='cpu')
pmamba.load_state_dict(cp.get('model_state_dict', cp), strict=False)
pmamba.eval()
print("pmamba loaded")

quat = VelocityPolarBearingQCCFeatureMotion(
    num_classes=25, pts_size=PTS, hidden_dims=[64, 256], dropout=0.05,
    edgeconv_k=20, merge_eps=1e-6, so3_weight=0.0, rotation_sigma=0.3,
    bearing_knn_k=10, qcc_weight=0.1, qcc_variant='contrastive',
).cuda()

pmloader = NvidiaLoader(framerate=32, phase='test')
qloader = NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=True,
)
n = len(pmloader)
print(f"Test: {n}")

def to_cuda(s):
    if isinstance(s, dict):
        return {k: (v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else
                    torch.from_numpy(v).unsqueeze(0).cuda() if hasattr(v, 'shape') else v)
                for k, v in s.items()}
    if isinstance(s, torch.Tensor): return s.unsqueeze(0).cuda()
    return torch.from_numpy(s).unsqueeze(0).cuda()

labels = np.zeros(n, int); pm_correct = np.zeros(n, bool); pm_logits = []
with torch.no_grad():
    for i in range(n):
        s, lab, _ = pmloader[i]
        t = to_cuda(s)
        out = torch.stack([pmamba(t) for _ in range(N_TTA)]).mean(dim=0)
        labels[i] = int(lab)
        pm_correct[i] = (out.argmax(dim=1).item() == int(lab))
        pm_logits.append(out.cpu())
        if i % 100 == 0: print(f"  pmamba {i}/{n}")
pm_logits = torch.cat(pm_logits, dim=0)
print(f"PMamba: {pm_correct.mean()*100:.2f}%")

best_acc = 0; best_ep = None; best_correct = None; best_logits = None
for c in ckpts:
    ep = int(c.split('epoch')[1].split('_')[0])
    ck = torch.load(c, map_location='cpu')
    quat.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    quat.eval()
    correct = np.zeros(n, bool); logits_list = []
    with torch.no_grad():
        for i in range(n):
            s, lab, _ = qloader[i]
            t = to_cuda(s)
            out = torch.stack([quat(t) for _ in range(N_TTA)]).mean(dim=0)
            correct[i] = (out.argmax(dim=1).item() == int(lab))
            logits_list.append(out.cpu())
    acc = correct.mean() * 100
    print(f"  velpolar ep{ep:3d}: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc; best_ep = ep; best_correct = correct
        best_logits = torch.cat(logits_list, dim=0)

print(f"\nBest velpolar: ep{best_ep} at {best_acc:.2f}%")
pm_acc = pm_correct.mean() * 100
oracle = pm_correct | best_correct
both_w = ~pm_correct & ~best_correct
only_p = pm_correct & ~best_correct
only_q = ~pm_correct & best_correct
print()
print('='*50)
print(f"PMamba:       {pm_acc:.2f}%  ({pm_correct.sum()}/{n})")
print(f"velpolar:     {best_acc:.2f}%  ({best_correct.sum()}/{n})")
print(f"Oracle:       {oracle.mean()*100:.2f}%  ({oracle.sum()}/{n})")
print('='*50)
print(f"Both correct: {(pm_correct & best_correct).sum()}")
print(f"Only PMamba:  {only_p.sum()}")
print(f"Only velpolar:{only_q.sum()}")
print(f"Both wrong:   {both_w.sum()}")
print(f"Headroom:     +{oracle.mean()*100 - pm_acc:.2f}pp")

pp = torch.softmax(pm_logits, dim=1)
qp = torch.softmax(best_logits, dim=1)
labels_t = torch.tensor(labels, dtype=torch.long)
bfa = 0; ba = 0
for ai in range(0, 105, 5):
    a = ai / 100
    fp = (a * pp + (1 - a) * qp).argmax(dim=1)
    acc = (fp == labels_t).sum().item() / n * 100
    if acc > bfa: bfa = acc; ba = a
print(f"\nBest fusion: alpha={ba:.2f} -> {bfa:.2f}% (+{bfa - pm_acc:.2f}pp)")
