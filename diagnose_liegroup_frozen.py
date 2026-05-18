"""Diagnostic: does the SO(3) state recurrence actually contribute, or is the
network ignoring it?

Loads Lie-group V1 ep109 (89.42% normal). For each LieGroupBlock in the model,
monkey-patches the forward pass to freeze the state at identity: skip the
omega -> exp -> Hamilton update entirely so q_t = q_0 = [1,0,0,0] for all t.

The sandwich-product read q_t * v_t * q_t^{-1} then degenerates to v_t (since
the identity rotation maps any vector to itself), so the read is just the
learned v_t projection.

If accuracy stays at ~89.4% -> Lie-group state is decorative.
If accuracy drops substantially -> state is load-bearing.

Also reports the trained omega magnitudes from a normal forward pass for
context.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_liegroup import (
    MotionLieGroup, LieGroupBlock, quat_normalize, quat_mul,
    omega_to_quat, quat_rotate,
)


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


def build_and_load_model():
    EP = 109
    WORKDIR = 'work_dir/pmamba_baseline_liegroup'
    ckpt_path = f'{WORKDIR}/epoch{EP}_model.pt'
    model = MotionLieGroup(
        num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
        multi_scale_num_scales=5,
        lg_hidden_dim=128, lg_num_layers=2, lg_num_heads=4,
        lg_n_states=8, lg_dropout=0.3, lg_bidirectional=True, lg_omega_scale=0.5,
    ).cuda()
    state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()
    return model


# ----------------------------------------------------------------------------
# 1. Capture omega magnitudes from a *normal* forward pass on a single batch
# ----------------------------------------------------------------------------
def probe_omegas(model):
    ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
    loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    x = next(iter(loader))[0].cuda().float()

    # Hook into each LieGroupBlock's forward to capture omega
    captured = []
    orig_forward = LieGroupBlock.forward

    def hook_forward(self, x_in):
        # Re-implement forward up to omega so we can intercept it.
        B, T, D = x_in.shape
        H, S = self.H, self.S
        omega_in = self.omega_proj(x_in)
        v_in     = self.v_proj(x_in)
        if self.use_short_conv:
            cat = torch.cat([omega_in, v_in], dim=-1).transpose(1, 2)
            cat = self.short_conv(cat)[..., :T].transpose(1, 2)
            half = H * S * 3
            omega_in, v_in = cat[..., :half], cat[..., half:]
        omega = omega_in.view(B, T, H, S, 3)
        omega = self.omega_scale * torch.tanh(omega)
        captured.append(omega.detach().abs().cpu())
        # Continue with original forward (re-call after restoring)
        return orig_forward(self, x_in)

    LieGroupBlock.forward = hook_forward
    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        LieGroupBlock.forward = orig_forward

    if captured:
        all_omega = torch.cat([c.flatten() for c in captured])
        print(f'[omega stats over {len(captured)} blocks, one sample]')
        print(f'  abs(omega).mean = {all_omega.mean():.4f}')
        print(f'  abs(omega).max  = {all_omega.max():.4f}')
        print(f'  abs(omega).std  = {all_omega.std():.4f}')
        # Quantiles
        qs = torch.quantile(all_omega, torch.tensor([0.5, 0.9, 0.99]))
        print(f'  median: {qs[0]:.4f}  p90: {qs[1]:.4f}  p99: {qs[2]:.4f}')
        print(f'  max possible per omega_scale=0.5: 0.5000')
        # Per-step rotation angle in radians = ||omega||
        norm_omega = torch.cat([c.norm(dim=-1).flatten() for c in captured])
        print(f'  per-step rotation angle (rad): mean={norm_omega.mean():.4f}  '
              f'max={norm_omega.max():.4f}  '
              f'(0.5 rad = {0.5*180/3.14159:.1f}deg)')


# ----------------------------------------------------------------------------
# 2. Monkey-patch LieGroupBlock.forward to freeze state at identity
# ----------------------------------------------------------------------------
def patch_freeze_state():
    """Replace LieGroupBlock.forward with a version that sets q_state = identity
    for all t, skipping the omega->exp->Hamilton update entirely."""

    def frozen_forward(self, x):
        import torch
        import torch.nn.functional as F
        B, T, D = x.shape
        H, S = self.H, self.S

        omega_in = self.omega_proj(x)
        v_in     = self.v_proj(x)
        if self.use_short_conv:
            cat = torch.cat([omega_in, v_in], dim=-1).transpose(1, 2)
            cat = self.short_conv(cat)[..., :T].transpose(1, 2)
            half = H * S * 3
            omega_in, v_in = cat[..., :half], cat[..., half:]

        # Skip omega; freeze q_state at identity (w=1, xyz=0)
        v_q  = v_in.view(B, T, H, S, 3)
        gate = torch.sigmoid(self.gate_proj(x)).view(B, T, H, S, 1)

        q_identity = torch.zeros(B, H, S, 4, device=x.device, dtype=x.dtype)
        q_identity[..., 0] = 1.0

        outs = []
        for t in range(T):
            # Sandwich product with identity quaternion = v_t unchanged
            u = quat_rotate(q_identity, v_q[:, t])     # equivalent to v_q[:, t]
            outs.append(u * gate[:, t])

        y = torch.stack(outs, dim=1).reshape(B, T, H * S * 3)
        return self.o_proj(self.dropout(y))

    LieGroupBlock.forward = frozen_forward


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
print('=' * 70)
print('LIE-GROUP V1 FROZEN-STATE DIAGNOSTIC')
print('=' * 70)

# (a) Probe omega magnitudes on a normal model
model = build_and_load_model()
probe_omegas(model)

# (b) Normal evaluation (sanity check: should reproduce 89.42)
print()
acc_normal = eval_test(model, 'normal Lie-group')

# (c) Frozen-state evaluation
print()
print('--- patching forward: q_t = identity for all t ---')
patch_freeze_state()
model_frozen = build_and_load_model()  # re-load to ensure patched forward used
acc_frozen = eval_test(model_frozen, 'frozen q_t = identity')

print()
print('=' * 70)
print(f'Verdict:')
print(f'  Normal Lie-group:  {acc_normal:.2f}%')
print(f'  Frozen state:      {acc_frozen:.2f}%')
print(f'  Delta:             {acc_normal - acc_frozen:+.2f} pp')
if abs(acc_normal - acc_frozen) < 1.0:
    print('  -> Lie-group state is DECORATIVE; rotation step contributes < 1pp')
elif acc_normal - acc_frozen > 5.0:
    print('  -> Lie-group state is LOAD-BEARING; frozen state loses > 5pp')
else:
    print('  -> Mixed signal: state contributes 1-5pp')
print('=' * 70)
