"""Oracle: PMamba epoch110 vs polar model (quaternion_branch_v2_polar)."""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
import numpy as np

from models.motion import Motion
from models.reqnn_motion import PolarBearingQCCFeatureMotion
from nvidia_dataloader import NvidiaLoader, NvidiaQuaternionQCCParityLoader

PTS = 256
N_TTA = 3
PMAMBA_CKPT = 'work_dir/pmamba_branch/epoch110_model.pt'

import glob
polar_ckpts = sorted(
    glob.glob('work_dir/quaternion_branch_v2_polar/epoch*_model.pt'),
    key=lambda p: int(p.split('epoch')[1].split('_')[0]),
)
# Evaluate each saved polar ckpt to find best; then oracle.
print(f"Polar checkpoints: {[c.split('/')[-1] for c in polar_ckpts[-5:]]}")

pmamba_model = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()
ckpt_p = torch.load(PMAMBA_CKPT, map_location='cpu')
state_p = ckpt_p.get('model_state_dict', ckpt_p.get('model', ckpt_p))
pmamba_model.load_state_dict(state_p, strict=False)
pmamba_model.eval()
print("PMamba loaded")

polar_model = PolarBearingQCCFeatureMotion(
    num_classes=25, pts_size=PTS, hidden_dims=[64, 256], dropout=0.05,
    edgeconv_k=20, merge_eps=1e-6, so3_weight=0.0, rotation_sigma=0.3,
    bearing_knn_k=10, qcc_weight=0.1, qcc_variant='contrastive',
).cuda()

pmamba_loader = NvidiaLoader(framerate=32, phase='test')
polar_loader = NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=True,
)
n = len(pmamba_loader)
print(f"Test: {n}")

def to_cuda(s):
    if isinstance(s, dict):
        return {k: (v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else
                    torch.from_numpy(v).unsqueeze(0).cuda() if hasattr(v, 'shape') else v)
                for k, v in s.items()}
    if isinstance(s, torch.Tensor):
        return s.unsqueeze(0).cuda()
    return torch.from_numpy(s).unsqueeze(0).cuda()

# Precompute PMamba predictions once.
pmamba_logits, pmamba_correct, labels = [], np.zeros(n, bool), np.zeros(n, int)
with torch.no_grad():
    for i in range(n):
        s, lab, _ = pmamba_loader[i]
        t = to_cuda(s)
        out = torch.stack([pmamba_model(t) for _ in range(N_TTA)]).mean(dim=0)
        pred = out.argmax(dim=1).item()
        labels[i] = int(lab)
        pmamba_correct[i] = (pred == int(lab))
        pmamba_logits.append(out.cpu())
        if i % 100 == 0:
            print(f"  pmamba {i}/{n}")
all_pmamba_logits = torch.cat(pmamba_logits, dim=0)
pmamba_acc = pmamba_correct.mean() * 100
print(f"PMamba: {pmamba_acc:.2f}%")

# Evaluate each polar checkpoint, track best.
best_acc = 0; best_ep = None; best_logits = None; best_correct = None
for c in polar_ckpts:
    ep = int(c.split('epoch')[1].split('_')[0])
    ckpt_q = torch.load(c, map_location='cpu')
    state_q = ckpt_q.get('model_state_dict', ckpt_q.get('model', ckpt_q))
    polar_model.load_state_dict(state_q, strict=False)
    polar_model.eval()
    correct = np.zeros(n, bool); logits_list = []
    with torch.no_grad():
        for i in range(n):
            s, lab, _ = polar_loader[i]
            t = to_cuda(s)
            out = torch.stack([polar_model(t) for _ in range(N_TTA)]).mean(dim=0)
            pred = out.argmax(dim=1).item()
            correct[i] = (pred == int(lab))
            logits_list.append(out.cpu())
    acc = correct.mean() * 100
    print(f"  polar ep{ep:3d}: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_ep = ep
        best_correct = correct
        best_logits = torch.cat(logits_list, dim=0)

print()
print(f"Best polar: ep{best_ep} at {best_acc:.2f}%")

# Oracle + fusion with best polar.
oracle = pmamba_correct | best_correct
only_p = pmamba_correct & ~best_correct
only_q = ~pmamba_correct & best_correct
both_w = ~pmamba_correct & ~best_correct
print()
print('='*50)
print(f"PMamba acc:   {pmamba_acc:.2f}% ({pmamba_correct.sum()}/{n})")
print(f"Polar acc:    {best_acc:.2f}% ({best_correct.sum()}/{n})")
print(f"Oracle:       {oracle.mean()*100:.2f}% ({oracle.sum()}/{n})")
print('='*50)
print(f"Both correct: {(pmamba_correct & best_correct).sum()}")
print(f"Only PMamba:  {only_p.sum()}")
print(f"Only polar:   {only_q.sum()}")
print(f"Both wrong:   {both_w.sum()}")
print(f"Headroom:     +{oracle.mean()*100 - pmamba_acc:.2f}pp")

pmamba_probs = torch.softmax(all_pmamba_logits, dim=1)
polar_probs = torch.softmax(best_logits, dim=1)
labels_t = torch.tensor(labels, dtype=torch.long)
best_fa = 0; best_alpha = 0
for ai in range(0, 105, 5):
    a = ai / 100
    fp = (a * pmamba_probs + (1 - a) * polar_probs).argmax(dim=1)
    acc = (fp == labels_t).sum().item() / n * 100
    if acc > best_fa: best_fa = acc; best_alpha = a
print()
print(f"Best fusion: alpha={best_alpha:.2f} -> {best_fa:.2f}%  (+{best_fa - pmamba_acc:.2f}pp over PMamba)")
