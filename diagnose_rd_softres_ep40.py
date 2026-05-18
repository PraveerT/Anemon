"""Diagnostic on RD-softres (rd_residual_scale=0.7) at ep40.
pts_size matches the training-time dynamic value at ep40 (=110).

Tests:
  normal       : block computes normally (in-training eval was 84.65%)
  zeroed block : RealDeltaNetBlock.forward returns zeros
                 -> encoder output = 0 + 0.7 * residual (scaled spatial)

If accuracy drops a lot when block=0 -> block IS load-bearing at inference
even with softres trained ckpt.
If accuracy stays close -> block still inference-decorative; the 0.7 residual
carries the signal alone.
"""
import os, sys, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_realdeltanet import MotionRealDeltaNet, RealDeltaNetBlock


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
    EP = 40
    WORKDIR = 'work_dir/pmamba_baseline_rd_softres'
    ckpt_path = f'{WORKDIR}/epoch{EP}_model.pt'
    print(f'[ckpt] {ckpt_path}')
    model = MotionRealDeltaNet(
        num_classes=25, pts_size=110, knn=[32, 24, 48, 24], topk=8,
        multi_scale_num_scales=5,
        rd_hidden_dim=128, rd_num_layers=2, rd_num_heads=4,
        rd_n_q=4, rd_n_v=8, rd_dropout=0.3, rd_bidirectional=True,
        rd_residual_scale=0.7,
    ).cuda()
    state = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()
    return model


orig_forward = RealDeltaNetBlock.forward

def zeroed_forward(self, x):
    B, T, D = x.shape
    return torch.zeros(B, T, D, device=x.device, dtype=x.dtype)


print('=' * 70)
print('RD-softres (residual_scale=0.7) FROZEN-BLOCK DIAGNOSTIC at ep40')
print('  pts_size=110 (matches training-time dynamic at ep40)')
print('=' * 70)

RealDeltaNetBlock.forward = orig_forward
m = build_and_load()
print()
acc_normal = eval_test(m, 'normal RD block')

print()
print('--- patching RealDeltaNetBlock to output zeros ---')
RealDeltaNetBlock.forward = zeroed_forward
m = build_and_load()
print()
acc_zero = eval_test(m, 'block output = 0')

RealDeltaNetBlock.forward = orig_forward

print()
print('=' * 70)
print(f'Normal RD block:   {acc_normal:.2f}%')
print(f'Zeroed block:      {acc_zero:.2f}%')
print(f'Block contribution: {acc_normal - acc_zero:+.2f} pp')
if abs(acc_normal - acc_zero) < 1.0:
    print('-> Block is INFERENCE-DECORATIVE (even with rs=0.7)')
elif acc_normal - acc_zero > 5.0:
    print('-> Block is INFERENCE-LOAD-BEARING (residual_scale=0.7 succeeded)')
else:
    print('-> Mixed: block contributes 1-5pp at inference')
print('=' * 70)
