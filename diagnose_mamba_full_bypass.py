"""Are the QuaternionLinear input/output projections in MambaTemporalEncoder
load-bearing, or are they decorative like the Mamba layers themselves?

Previous diagnostic zeroed Mamba.forward and got 90.04 (unchanged).
That left the input/output QuaternionLinear projections active inside the
encoder. This test goes further: replace the ENTIRE MambaTemporalEncoder
forward with the identity (fea3 passes through unchanged).

If accuracy stays at 90.04 -> the whole encoder is decorative, the cleanest
architecture is truly nn.Identity. Reproduce with no temporal encoder at all.

If accuracy drops -> the QuaternionLinear bottleneck (256->128->256) was
doing real work. The "cleanest" model needs those projections.
"""
import os, sys, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion import Motion, MambaTemporalEncoder
from mamba_ssm.modules.mamba_simple import Mamba


def eval_test(model, name):
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
    model = Motion(
        num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
        multi_scale_num_scales=5,
    ).cuda()
    state = torch.load('work_dir/pmamba_branch/epoch115_model.pt', map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()
    return model


orig_encoder_forward = MambaTemporalEncoder.forward


# 1. Baseline: normal Mamba (sanity, should give 90.04)
print('=' * 70)
print('PMAMBA ENCODER FULL BYPASS DIAGNOSTIC')
print('=' * 70)
m = build_and_load()
print()
acc_normal = eval_test(m, 'normal Mamba encoder')

# 2. Zero only the Mamba layers (previous test, residual + in/out proj active)
def zero_mamba_layers_forward(self, x):
    return torch.zeros_like(x)
print()
print('--- Mamba layers only -> 0 (in/out proj active) ---')
Mamba.forward = zero_mamba_layers_forward
m = build_and_load()
acc_zeromamba = eval_test(m, 'zeroed Mamba layers')
Mamba.forward = lambda self, x, *a, **k: orig_encoder_forward.__globals__['Mamba'].forward(self, x, *a, **k)  # restore via globals

# Just to be safe, reset Mamba.forward via reload
import importlib, mamba_ssm.modules.mamba_simple as mm; importlib.reload(mm)
from mamba_ssm.modules.mamba_simple import Mamba as Mamba2
Mamba2.forward  # noqa
# we now go further

# 3. Replace the entire encoder forward with identity passthrough
def identity_encoder_forward(self, x):
    # x is (B, C, T, N). Encoder output dim = 256, same as input. Just return x.
    return x

print()
print('--- ENTIRE MambaTemporalEncoder -> identity passthrough ---')
MambaTemporalEncoder.forward = identity_encoder_forward
m = build_and_load()
acc_identity = eval_test(m, 'encoder = identity (no in/out proj, no Mamba)')

# Restore
MambaTemporalEncoder.forward = orig_encoder_forward

print()
print('=' * 70)
print(f'Normal:                              {acc_normal:.2f}%')
print(f'Zero Mamba (in/out proj active):    {acc_zeromamba:.2f}%')
print(f'Identity encoder (no proj at all): {acc_identity:.2f}%')
print()
print(f'Mamba layers contribution:           {acc_normal - acc_zeromamba:+.2f} pp')
print(f'In/out proj contribution:            {acc_zeromamba - acc_identity:+.2f} pp')
print(f'Total encoder contribution:          {acc_normal - acc_identity:+.2f} pp')
print('=' * 70)
