"""Diagnostic for BDN-Q soft-residual ep30: is the temporal block contributing?

With residual_scale=0.1, the per-layer composition is:
    x = block(x) + 0.1 * residual

We compare:
    normal:        block produces its trained output
    zeroed block:  override block(x) -> 0  => x = 0 + 0.1 * residual = 0.1*residual

If accuracy drops with zeroed block, the block is contributing.
If accuracy stays the same, the block is decorative even at λ=0.1.
"""
import os, sys, math, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
import torch.nn.functional as F
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_bdn_q import MotionBDeltaQ, BDeltaQBlock


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


def build_and_load(ep):
    WORKDIR = 'work_dir/pmamba_baseline_bdnq_softres'
    ckpt_path = f'{WORKDIR}/epoch{ep}_model.pt'
    print(f'[ckpt] {ckpt_path}')
    model = MotionBDeltaQ(
        num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
        multi_scale_num_scales=5,
        bdnq_hidden_dim=128, bdnq_num_layers=2, bdnq_num_heads=4,
        bdnq_n_q=4, bdnq_n_v=8, bdnq_buffer_size=1, bdnq_dropout=0.3,
        bdnq_bidirectional=True, bdnq_residual_scale=0.1,
    ).cuda()
    state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()
    return model


orig_forward = BDeltaQBlock.forward
def zeroed_forward(self, x):
    return torch.zeros_like(self.o_proj(torch.zeros(x.shape[0], x.shape[1], self.H * self.n_v * 4, device=x.device, dtype=x.dtype)))


EP = 30
print(f'BDN-Q soft-residual (λ=0.1) diagnostic at ep{EP}')
print('=' * 60)
BDeltaQBlock.forward = orig_forward
m = build_and_load(EP)
print()
acc_normal = eval_test(m, 'normal block')

print()
print('--- patching block to output zeros ---')
BDeltaQBlock.forward = zeroed_forward
m = build_and_load(EP)
print()
acc_zero = eval_test(m, 'block output = 0')
BDeltaQBlock.forward = orig_forward

print()
print('=' * 60)
print(f'Normal block:        {acc_normal:.2f}%')
print(f'Zeroed block:        {acc_zero:.2f}%')
print(f'Delta (block contrib): {acc_normal - acc_zero:+.2f} pp')
if acc_normal - acc_zero > 5:
    print('-> Block is LOAD-BEARING; > 5pp contribution')
elif acc_normal - acc_zero > 1:
    print('-> Block contributes 1-5pp')
else:
    print('-> Block is DECORATIVE; < 1pp contribution')
print('=' * 60)
