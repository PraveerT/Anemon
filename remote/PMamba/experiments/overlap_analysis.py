import sys, os
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
import torch, numpy as np

from models.motion import Motion
from nvidia_dataloader import NvidiaLoader, NvidiaQuaternionQCCParityLoader
from models.reqnn_motion import BearingQCCFeatureMotion


def to_cuda_input(sample):
    """Move all values to CUDA, converting numpy arrays to tensors first."""
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


pmamba_model = Motion(num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8).cuda()
ckpt_p = torch.load('work_dir/pmamba_branch/epoch110_model.pt', map_location='cpu')
pmamba_model.load_state_dict(ckpt_p.get('model_state_dict', ckpt_p), strict=False)
pmamba_model.eval()
print('PMamba loaded', flush=True)

old_quat = BearingQCCFeatureMotion(
    num_classes=25, pts_size=256, hidden_dims=[64, 256],
    dropout=0.05, edgeconv_k=20, merge_eps=1e-6,
    so3_weight=0.0, rotation_sigma=0.3, bearing_knn_k=10, qcc_weight=0.1,
).cuda()
ckpt_old = torch.load('work_dir/quaternion_branch/epoch112_model.pt', map_location='cpu')
old_quat.load_state_dict(ckpt_old.get('model_state_dict', ckpt_old), strict=False)
old_quat.eval()
print('Old quat loaded', flush=True)

new_quat = BearingQCCFeatureMotion(
    num_classes=25, pts_size=256, hidden_dims=[64, 256],
    dropout=0.05, edgeconv_k=20, merge_eps=1e-6,
    so3_weight=0.0, rotation_sigma=0.3, bearing_knn_k=10,
    qcc_weight=0.1, qcc_variant='prediction',
).cuda()
ckpt_new = torch.load('work_dir/quaternion_corr_fixed_finetune/epoch130_model.pt', map_location='cpu')
new_quat.load_state_dict(ckpt_new.get('model_state_dict', ckpt_new), strict=False)
new_quat.eval()
print('New quat loaded', flush=True)

# Loaders
pmamba_loader = NvidiaLoader(framerate=32, phase='test')
# Old quat: no correspondence
old_quat_loader = NvidiaQuaternionQCCParityLoader(framerate=32, phase='test', return_correspondence=False)
# New quat: WITH correspondence (trained with it)
new_quat_loader = NvidiaQuaternionQCCParityLoader(framerate=32, phase='test', return_correspondence=True)
n = len(pmamba_loader)
print(f'Test: {n}', flush=True)

# Full overlap analysis
pmamba_ok = np.zeros(n, dtype=bool)
old_ok = np.zeros(n, dtype=bool)
new_ok = np.zeros(n, dtype=bool)
labs = np.zeros(n, dtype=int)
p_logits, o_logits, n_logits = [], [], []

with torch.no_grad():
    for i in range(n):
        # PMamba (TTA=3 as standard)
        sp, lp, _ = pmamba_loader[i]
        inp_p = to_cuda_input(sp)
        out_p = torch.stack([pmamba_model(inp_p) for _ in range(3)]).mean(0)

        # Old quat (no corr, TTA=1 — deterministic linspace)
        so, lo, _ = old_quat_loader[i]
        inp_o = to_cuda_input(so)
        out_o = old_quat(inp_o)

        # New quat (with corr, TTA=10 — needs averaging over random fallback)
        sn, ln, _ = new_quat_loader[i]
        inp_n = to_cuda_input(sn)
        out_n = torch.stack([new_quat(inp_n) for _ in range(10)]).mean(0)

        label = int(lp)
        labs[i] = label
        pmamba_ok[i] = (out_p.argmax(1).item() == label)
        old_ok[i] = (out_o.argmax(1).item() == label)
        new_ok[i] = (out_n.argmax(1).item() == label)
        p_logits.append(out_p.cpu())
        o_logits.append(out_o.cpu())
        n_logits.append(out_n.cpu())
        if i % 50 == 0:
            print(f'  {i}/{n}...', flush=True)

p_logits = torch.cat(p_logits)
o_logits = torch.cat(o_logits)
n_logits = torch.cat(n_logits)

print(f'\n============================================================')
print(f'PMamba:     {pmamba_ok.sum()}/{n} = {pmamba_ok.mean()*100:.2f}%')
print(f'Old Quat:   {old_ok.sum()}/{n} = {old_ok.mean()*100:.2f}%')
print(f'New Quat:   {new_ok.sum()}/{n} = {new_ok.mean()*100:.2f}%')

oracle_old = pmamba_ok | old_ok
oracle_new = pmamba_ok | new_ok
only_p_old = pmamba_ok & ~old_ok
only_old = ~pmamba_ok & old_ok
only_p_new = pmamba_ok & ~new_ok
only_new = ~pmamba_ok & new_ok

print(f'\nPMamba + OLD Quat:')
print(f'  Oracle:        {oracle_old.sum()}/{n} = {oracle_old.mean()*100:.2f}%')
print(f'  Only PMamba:   {only_p_old.sum()}')
print(f'  Only Old Quat: {only_old.sum()}')
print(f'  Both wrong:    {(~pmamba_ok & ~old_ok).sum()}')

print(f'\nPMamba + NEW Quat:')
print(f'  Oracle:        {oracle_new.sum()}/{n} = {oracle_new.mean()*100:.2f}%')
print(f'  Only PMamba:   {only_p_new.sum()}')
print(f'  Only New Quat: {only_new.sum()}')
print(f'  Both wrong:    {(~pmamba_ok & ~new_ok).sum()}')

# Late fusion
labels_t = torch.tensor(labs, dtype=torch.long)
pp = torch.softmax(p_logits, 1)
op = torch.softmax(o_logits, 1)
np_ = torch.softmax(n_logits, 1)
best_oa, best_oac, best_na, best_nac = 0, 0, 0, 0
print(f'\nLate fusion: alpha*PMamba + (1-a)*Quat')
print(f'{"a":>5} {"OLD":>8} {"NEW":>8}')
for ai in range(0, 105, 5):
    a = ai/100.
    ao = ((a*pp+(1-a)*op).argmax(1)==labels_t).float().mean().item()*100
    an = ((a*pp+(1-a)*np_).argmax(1)==labels_t).float().mean().item()*100
    print(f'{a:>5.2f} {ao:>7.2f}% {an:>7.2f}%')
    if ao > best_oac: best_oac, best_oa = ao, a
    if an > best_nac: best_nac, best_na = an, a
print(f'\nBest OLD fusion: a={best_oa:.2f} -> {best_oac:.2f}%')
print(f'Best NEW fusion: a={best_na:.2f} -> {best_nac:.2f}%')
print('DONE', flush=True)
