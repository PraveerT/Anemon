"""Oracle + fusion analysis: PMamba vs v26a (xyz+tops).

Runs both trained models on the test set, measures:
  - Individual accuracy
  - Oracle upper bound (either correct)
  - Overlap breakdown (both right, only one right, both wrong)
  - Late-fusion sweep across softmax/logit weights

Use after v26a training completes. Reads:
  work_dir/pmamba_branch/epoch110_model.pt
  work_dir/quaternion_tops_xyz_v26a/epoch140_model.pt
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')

import torch
import numpy as np

from models.motion import Motion
from models.reqnn_motion import TopsXYZInputMotion
from nvidia_dataloader import NvidiaLoader, NvidiaQuaternionQCCParityLoader

PMAMBA_CKPT = 'work_dir/pmamba_branch/epoch110_model.pt'
QUAT_CKPT = 'work_dir/quaternion_tops_xyz_v26a/epoch140_model.pt'
PTS = 256
N_TTA = 3

# --- Load PMamba ---
pmamba_model = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()
ckpt_p = torch.load(PMAMBA_CKPT, map_location='cpu')
state_p = ckpt_p.get('model_state_dict', ckpt_p.get('model', ckpt_p))
missing, unexpected = pmamba_model.load_state_dict(state_p, strict=False)
pmamba_model.eval()
print(f"PMamba loaded. missing={len(missing)} unexpected={len(unexpected)}")

# --- Load Quaternion v26a (xyz + tops) ---
quat_model = TopsXYZInputMotion(
    num_classes=25, pts_size=PTS, hidden_dims=[64, 256], dropout=0.05,
    edgeconv_k=20, merge_eps=1e-6, so3_weight=0.0, rotation_sigma=0.3,
    bearing_knn_k=10, qcc_weight=0.0, qcc_variant='contrastive',
).cuda()
ckpt_q = torch.load(QUAT_CKPT, map_location='cpu')
state_q = ckpt_q.get('model_state_dict', ckpt_q.get('model', ckpt_q))
missing_q, unexpected_q = quat_model.load_state_dict(state_q, strict=False)
quat_model.eval()
print(f"v26a loaded. missing={len(missing_q)} unexpected={len(unexpected_q)}")

# --- Test loaders ---
pmamba_loader = NvidiaLoader(framerate=32, phase='test')
quat_loader = NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=True,
    assignment_mode='hungarian',
)
n_test = len(pmamba_loader)
assert len(quat_loader) == n_test, "Loader length mismatch"
print(f"Test samples: {n_test}")

pmamba_correct = np.zeros(n_test, dtype=bool)
quat_correct = np.zeros(n_test, dtype=bool)
pmamba_preds = np.zeros(n_test, dtype=int)
quat_preds = np.zeros(n_test, dtype=int)
labels = np.zeros(n_test, dtype=int)
all_pmamba_logits = []
all_quat_logits = []

def to_cuda(sample):
    if isinstance(sample, dict):
        return {k: (v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else
                    torch.from_numpy(v).unsqueeze(0).cuda() if hasattr(v, 'shape') else v)
                for k, v in sample.items()}
    if isinstance(sample, torch.Tensor):
        return sample.unsqueeze(0).cuda()
    return torch.from_numpy(sample).unsqueeze(0).cuda()

with torch.no_grad():
    for i in range(n_test):
        sample_p, label_p, _ = pmamba_loader[i]
        inp_p = to_cuda(sample_p)
        outs_p = [pmamba_model(inp_p) for _ in range(N_TTA)]
        out_p = torch.stack(outs_p).mean(dim=0)
        pred_p = out_p.argmax(dim=1).item()

        sample_q, label_q, _ = quat_loader[i]
        inp_q = to_cuda(sample_q)
        outs_q = [quat_model(inp_q) for _ in range(N_TTA)]
        out_q = torch.stack(outs_q).mean(dim=0)
        pred_q = out_q.argmax(dim=1).item()

        label = int(label_p)
        labels[i] = label
        pmamba_preds[i] = pred_p
        quat_preds[i] = pred_q
        pmamba_correct[i] = (pred_p == label)
        quat_correct[i] = (pred_q == label)
        all_pmamba_logits.append(out_p.cpu())
        all_quat_logits.append(out_q.cpu())

        if i % 50 == 0:
            print(f"  {i}/{n_test}  pmamba={pmamba_correct[:i+1].mean()*100:.1f}%  quat={quat_correct[:i+1].mean()*100:.1f}%")

all_pmamba_logits = torch.cat(all_pmamba_logits, dim=0)
all_quat_logits = torch.cat(all_quat_logits, dim=0)

pmamba_acc = pmamba_correct.mean() * 100
quat_acc = quat_correct.mean() * 100
oracle_correct = pmamba_correct | quat_correct
oracle_acc = oracle_correct.mean() * 100
both_correct = pmamba_correct & quat_correct
both_wrong = ~pmamba_correct & ~quat_correct
only_pmamba = pmamba_correct & ~quat_correct
only_quat = ~pmamba_correct & quat_correct

print()
print('='*56)
print(f"PMamba accuracy:     {pmamba_acc:6.2f}% ({pmamba_correct.sum()}/{n_test})")
print(f"v26a accuracy:       {quat_acc:6.2f}% ({quat_correct.sum()}/{n_test})")
print(f"Oracle (either):     {oracle_acc:6.2f}% ({oracle_correct.sum()}/{n_test})")
print('='*56)
print(f"Both correct:        {both_correct.sum():4d}  ({both_correct.mean()*100:5.1f}%)")
print(f"Only PMamba:         {only_pmamba.sum():4d}  ({only_pmamba.mean()*100:5.1f}%)")
print(f"Only v26a:           {only_quat.sum():4d}  ({only_quat.mean()*100:5.1f}%)")
print(f"Both wrong:          {both_wrong.sum():4d}  ({both_wrong.mean()*100:5.1f}%)")
print('='*56)
print(f"Complementarity (exactly-one): {only_pmamba.sum() + only_quat.sum()}")
print(f"Headroom over best single:     +{oracle_acc - max(pmamba_acc, quat_acc):.2f}pp")

print()
print('='*56)
print("Late Fusion: alpha * PMamba_softmax + (1-alpha) * v26a_softmax")
print('='*56)
pmamba_probs = torch.softmax(all_pmamba_logits, dim=1)
quat_probs = torch.softmax(all_quat_logits, dim=1)
labels_t = torch.tensor(labels, dtype=torch.long)
best_alpha = 0
best_fusion_acc = 0
for alpha_int in range(0, 105, 5):
    alpha = alpha_int / 100.0
    fused = alpha * pmamba_probs + (1 - alpha) * quat_probs
    fused_preds = fused.argmax(dim=1)
    correct = (fused_preds == labels_t).sum().item()
    acc = correct / n_test * 100
    print(f"  alpha={alpha:4.2f}  acc={acc:6.2f}%  ({correct}/{n_test})")
    if acc > best_fusion_acc:
        best_fusion_acc = acc
        best_alpha = alpha

print()
print(f"Best late fusion: alpha={best_alpha:.2f} -> {best_fusion_acc:.2f}%")
print(f"Lift over PMamba: +{best_fusion_acc - pmamba_acc:.2f}pp")
print(f"Lift over v26a:   +{best_fusion_acc - quat_acc:.2f}pp")
print(f"Oracle ceiling:   {oracle_acc:.2f}%")
print(f"Fusion vs oracle: {(best_fusion_acc - quat_acc) / (oracle_acc - quat_acc + 1e-9) * 100:.1f}% of available headroom captured")
