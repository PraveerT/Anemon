"""Overlap and oracle analysis: PMamba vs Quaternion (80.29%) vs Quaternion+TTA (81.54%).

Compares per-sample correctness across three models, computes:
  - Individual accuracies
  - Pairwise oracle (best-of-pair upper bound)
  - Disagreement counts (only-A correct, only-B correct, both correct, both wrong)
  - Late-fusion accuracy sweep
"""
import sys
import os
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
import torch
import torch.nn.functional as F
import numpy as np

from models.motion import Motion
from models.reqnn_motion import BearingQCCFeatureMotion
from nvidia_dataloader import NvidiaLoader, NvidiaQuaternionQCCParityLoader


def to_cuda_input(sample):
    if isinstance(sample, dict):
        out = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.unsqueeze(0).cuda()
            elif isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v).unsqueeze(0).cuda()
            else:
                out[k] = v
        return out
    return sample.unsqueeze(0).cuda()


# ============================================================
# Load PMamba
# ============================================================
pmamba_model = Motion(
    num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
).cuda()
ckpt_p = torch.load(
    'work_dir/pmamba_branch/epoch110_model.pt', map_location='cpu',
)
pmamba_model.load_state_dict(
    ckpt_p.get('model_state_dict', ckpt_p), strict=False,
)
pmamba_model.eval()
print('PMamba loaded', flush=True)


# ============================================================
# Load Quaternion (qcc_variant=grounded_cycle, the 80.29% checkpoint)
# pts_size=256 because dynamic ramp put it there during training
# ============================================================
quat_model = BearingQCCFeatureMotion(
    num_classes=25,
    pts_size=256,
    hidden_dims=[64, 256],
    dropout=0.05,
    edgeconv_k=20,
    merge_eps=1e-6,
    so3_weight=0.0,
    rotation_sigma=0.3,
    bearing_knn_k=10,
    qcc_weight=0.1,
    qcc_variant='grounded_cycle',
).cuda()
ckpt_q = torch.load(
    'work_dir/quaternion_branch/epoch112_model.pt', map_location='cpu',
)
quat_model.load_state_dict(
    ckpt_q.get('model_state_dict', ckpt_q), strict=False,
)
quat_model.eval()
quat_model.pts_size = 256  # ensure runtime sampler uses 256
print('Quaternion loaded', flush=True)


# ============================================================
# Loaders
# ============================================================
pmamba_loader = NvidiaLoader(framerate=32, phase='test')
quat_loader = NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=False,
)
n = len(pmamba_loader)
print(f'Test samples: {n}', flush=True)


# ============================================================
# Per-sample inference
# ============================================================
pmamba_ok = np.zeros(n, dtype=bool)
quat_det_ok = np.zeros(n, dtype=bool)   # deterministic linspace
quat_tta_ok = np.zeros(n, dtype=bool)   # random TTA=10
labs = np.zeros(n, dtype=np.int64)

p_logits = []
q_det_logits = []
q_tta_logits = []

# Helper for forcing random sampling at TTA time
original_sampler = quat_model._sample_point_indices


def random_sampler(point_count, device):
    sample_size = min(quat_model.pts_size, point_count)
    if sample_size == point_count:
        return None
    return torch.randperm(point_count, device=device)[:sample_size]


with torch.no_grad():
    for i in range(n):
        sp, lp, _ = pmamba_loader[i]
        inp_p = to_cuda_input(sp)
        out_p = pmamba_model(inp_p)

        sq, lq, _ = quat_loader[i]
        inp_q = to_cuda_input(sq)

        # Quaternion deterministic (linspace sampling)
        quat_model._sample_point_indices = original_sampler
        out_q_det = quat_model(inp_q)

        # Quaternion random TTA=10
        quat_model._sample_point_indices = random_sampler
        out_q_tta = torch.stack(
            [F.softmax(quat_model(inp_q), dim=-1) for _ in range(10)],
        ).mean(0)
        quat_model._sample_point_indices = original_sampler

        label = int(lp)
        labs[i] = label
        pmamba_ok[i] = (out_p.argmax(1).item() == label)
        quat_det_ok[i] = (out_q_det.argmax(1).item() == label)
        quat_tta_ok[i] = (out_q_tta.argmax(1).item() == label)

        p_logits.append(out_p.cpu())
        q_det_logits.append(out_q_det.cpu())
        q_tta_logits.append(out_q_tta.cpu())  # already softmax

        if i % 50 == 0:
            print(f'  {i}/{n}...', flush=True)

p_logits = torch.cat(p_logits)
q_det_logits = torch.cat(q_det_logits)
q_tta_probs = torch.cat(q_tta_logits)
labels_t = torch.from_numpy(labs)


# ============================================================
# Individual + Oracle
# ============================================================
print()
print('=' * 60)
print(f'Individual accuracies:')
print(f'  PMamba:           {pmamba_ok.sum():3d}/{n} = {pmamba_ok.mean()*100:.2f}%')
print(f'  Quat (det):       {quat_det_ok.sum():3d}/{n} = {quat_det_ok.mean()*100:.2f}%')
print(f'  Quat (TTA=10):    {quat_tta_ok.sum():3d}/{n} = {quat_tta_ok.mean()*100:.2f}%')

print()
print('-' * 60)
print('Pairwise oracle (best-of-pair upper bound):')
oracle_det = pmamba_ok | quat_det_ok
oracle_tta = pmamba_ok | quat_tta_ok
print(f'  PMamba | Quat(det):    {oracle_det.sum():3d}/{n} = {oracle_det.mean()*100:.2f}%')
print(f'  PMamba | Quat(TTA):    {oracle_tta.sum():3d}/{n} = {oracle_tta.mean()*100:.2f}%')

print()
print('-' * 60)
print('Disagreement breakdown vs PMamba:')
for name, ok in [('Quat(det)', quat_det_ok), ('Quat(TTA)', quat_tta_ok)]:
    only_p = pmamba_ok & ~ok
    only_q = ~pmamba_ok & ok
    both = pmamba_ok & ok
    neither = ~pmamba_ok & ~ok
    print(f'  {name}:')
    print(f'    Both correct:    {both.sum():3d}')
    print(f'    Only PMamba:     {only_p.sum():3d}')
    print(f'    Only {name:9s}: {only_q.sum():3d}  (unique correct)')
    print(f'    Both wrong:      {neither.sum():3d}')


# ============================================================
# Late fusion sweep
# ============================================================
print()
print('-' * 60)
print('Late fusion: alpha * PMamba + (1-alpha) * Quat')

pp = F.softmax(p_logits, dim=-1)
qd = F.softmax(q_det_logits, dim=-1)
qt = q_tta_probs  # already in prob space

best_det = (0.0, 0.0)
best_tta = (0.0, 0.0)
print(f'  {"alpha":>6}  {"+ Quat(det)":>12}  {"+ Quat(TTA)":>12}')
for ai in range(0, 105, 5):
    a = ai / 100.0
    fused_det = a * pp + (1 - a) * qd
    fused_tta = a * pp + (1 - a) * qt
    acc_det = (fused_det.argmax(dim=-1) == labels_t).float().mean().item() * 100
    acc_tta = (fused_tta.argmax(dim=-1) == labels_t).float().mean().item() * 100
    print(f'  {a:>6.2f}  {acc_det:>11.2f}%  {acc_tta:>11.2f}%')
    if acc_det > best_det[1]:
        best_det = (a, acc_det)
    if acc_tta > best_tta[1]:
        best_tta = (a, acc_tta)

print()
print(f'  Best PMamba + Quat(det):  alpha={best_det[0]:.2f} -> {best_det[1]:.2f}%')
print(f'  Best PMamba + Quat(TTA):  alpha={best_tta[0]:.2f} -> {best_tta[1]:.2f}%')
print()
print('DONE', flush=True)
