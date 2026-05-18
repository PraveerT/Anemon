"""Frozen-mechanism diagnostic on the ORIGINAL PMamba baseline (Motion with
MambaTemporalEncoder). Tests whether Mamba itself is decorative on this
backbone, the same way RD/AttRD/BDN-Q/Lie-group all turned out to be.

Setup:
  - Load Motion (the base PMamba class, 90.04% test acc per memory)
  - Load ep115 checkpoint from work_dir/pmamba_branch/
  - Three eval modes:
      normal      : Mamba layers compute normally
      zeroed_mamba: each Mamba layer returns zeros (residual still applies)
      no_residual : Mamba computes normally but residual is dropped (would
                    have to retrain; SKIP for now -- just for symmetry)

If zeroed_mamba acc ≈ normal acc -> Mamba is decorative.
If zeroed_mamba acc drops significantly -> Mamba is contributing.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion import Motion, MambaTemporalEncoder
from mamba_ssm.modules.mamba_simple import Mamba


def eval_test(model, name, pts_size=256):
    ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
    loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    correct = 0; total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].cuda().float()
            y = batch[1].cuda().long().ravel()
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    acc = 100 * correct / total
    print(f'[eval {name}] acc = {acc:.2f}% ({correct}/{total})')
    return acc


def build_and_load():
    ckpt_path = 'work_dir/pmamba_branch/epoch115_model.pt'
    print(f'[ckpt] {ckpt_path}')
    model = Motion(
        num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
        multi_scale_num_scales=5,
    ).cuda()
    state = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()
    return model


orig_mamba_forward = Mamba.forward


def make_zeroed_mamba_forward():
    """Return a forward fn that yields zeros of the same shape as input."""
    def zeroed_forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)
    return zeroed_forward


print('=' * 70)
print('ORIGINAL PMAMBA (Mamba temporal encoder) FROZEN DIAGNOSTIC')
print('=' * 70)

# 1. Normal eval
Mamba.forward = orig_mamba_forward
m = build_and_load()
print()
acc_normal = eval_test(m, 'normal Mamba')

# 2. Zeroed Mamba layers (residual still preserves spatial)
print()
print('--- patching Mamba.forward to return zeros ---')
Mamba.forward = make_zeroed_mamba_forward()
m = build_and_load()
print()
acc_zero = eval_test(m, 'zeroed Mamba (residual intact)')

# Restore
Mamba.forward = orig_mamba_forward

print()
print('=' * 70)
print(f'Normal Mamba:               {acc_normal:.2f}%')
print(f'Zeroed Mamba (residual on): {acc_zero:.2f}%')
print(f'Mamba contribution:         {acc_normal - acc_zero:+.2f} pp')
if abs(acc_normal - acc_zero) < 1.0:
    print('-> Mamba is DECORATIVE; < 1pp contribution')
elif acc_normal - acc_zero > 5.0:
    print('-> Mamba is LOAD-BEARING; > 5pp contribution')
else:
    print('-> Mixed: 1-5pp contribution')
print('=' * 70)
