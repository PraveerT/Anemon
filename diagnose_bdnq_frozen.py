"""Diagnostic: does BDN-Q's buffer-attn / eject-into-state mechanism actually
contribute, or is the network ignoring it?

BDN-Q has two mechanisms layered on top of the RD-style write recurrence:
  (1) Short FIFO attention buffer over the last W (k_t, v_t) pairs
      -> buf_out = softmax(q_t . K_buf) . V_buf
  (2) Eject-into-delta-state: oldest buffer entry rolls into S via rank-1
      delta update on overflow
      -> delta_out = q_t^T S_t

  yt = buf_out + delta_out

We ablate each path independently and report the test accuracy delta.

Variants:
  normal              full BDN-Q (89.00 expected at ep108 / 87.76 at train-best)
  no_buffer           yt = delta_out only (skip buffer attention)
  no_delta            yt = buf_out only (skip delta-state read)
  no_eject            never write to S on buffer overflow (S stays 0); buffer
                      still works as a window. Tests if the eject mechanism
                      itself matters
  no_buf_no_delta     yt = 0 read (return only o_proj of v projection? we keep
                      v on the path via buf_out's V_stack but zero scores).
                      Tests if either read is doing anything at all.
"""
import os, sys, math, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

from torch.utils.data import DataLoader
import torch.nn.functional as F
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
    WORKDIR = 'work_dir/pmamba_baseline_bdnq'
    ckpt_path = f'{WORKDIR}/epoch{ep}_model.pt'
    print(f'[ckpt] {ckpt_path}')
    model = MotionBDeltaQ(
        num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
        multi_scale_num_scales=5,
        bdnq_hidden_dim=128, bdnq_num_layers=2, bdnq_num_heads=4,
        bdnq_n_q=4, bdnq_n_v=8, bdnq_buffer_size=1, bdnq_dropout=0.3,
        bdnq_bidirectional=True,
    ).cuda()
    state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()
    return model


orig_forward = BDeltaQBlock.forward


def make_patched_forward(mode):
    """Returns a BDeltaQBlock.forward that ablates one of the BDN-Q paths.
    mode in {'normal', 'no_buffer', 'no_delta', 'no_eject', 'no_buf_no_delta'}.
    """
    def patched_forward(self, x):
        B, T, D = x.shape
        H, nq, nv, W = self.H, self.n_q, self.n_v, self.W

        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            s1 = H * nq * 4
            s2 = s1 + H * nq * 4
            q, k, v = qkv[..., :s1], qkv[..., s1:s2], qkv[..., s2:]

        q = q.view(B, T, H, nq, 4)
        k = k.view(B, T, H, nq, 4)
        v = v.view(B, T, H, nv, 4)
        k_flat = k.reshape(B, T, H, nq * 4)
        k_flat = F.normalize(k_flat, dim=-1)
        k = k_flat.view(B, T, H, nq, 4)
        q = F.silu(q)

        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, nq)
        S = torch.zeros(B, H, nv, nq, 4, device=x.device, dtype=x.dtype)

        K_buf, V_buf, P_buf = [], [], []
        ys = []

        for t in range(T):
            kt = k[:, t]; vt = v[:, t]
            beta_t = beta[:, t]

            kt_flat = kt.reshape(B, H, nq * 4)
            kt_rot_flat = self._rope_flat(kt_flat, t)
            kt_rot = kt_rot_flat.view(B, H, nq, 4)

            if len(K_buf) >= W:
                kt_old = K_buf.pop(0); vt_old = V_buf.pop(0); P_buf.pop(0)
                if mode != 'no_eject':
                    Sk_old = (S * kt_old.unsqueeze(2)).sum(dim=3)
                    err = vt_old - Sk_old
                    upd = err.unsqueeze(3) * kt_old.unsqueeze(2)
                    beta_bc = beta_t.view(B, H, 1, nq, 1)
                    S = S + beta_bc * upd

            K_buf.append(kt_rot); V_buf.append(vt); P_buf.append(t)

            # Buffer attention
            q_flat = q[:, t].reshape(B, H, nq * 4)
            q_rot_flat = self._rope_flat(q_flat, t)
            K_stack_flat = torch.stack([kb.reshape(B, H, nq*4) for kb in K_buf], dim=2)
            V_stack = torch.stack(V_buf, dim=2)
            scores = torch.einsum('bhd,bhld->bhl', q_rot_flat, K_stack_flat) / math.sqrt(nq * 4)
            attn = self.attn_dropout(F.softmax(scores, dim=-1))
            buf_out = torch.einsum('bhl,bhlve->bhve', attn, V_stack)

            # Delta-state read
            delta_out = (S * q[:, t].unsqueeze(2)).sum(dim=3)

            if mode == 'normal':
                yt = buf_out + delta_out
            elif mode == 'no_buffer':
                yt = delta_out
            elif mode == 'no_delta':
                yt = buf_out
            elif mode == 'no_eject':
                # buffer + (delta_out from S that never got written to) = buf_out + 0
                yt = buf_out + delta_out  # delta_out will be ~0 since S stays zero
            elif mode == 'no_buf_no_delta':
                yt = torch.zeros_like(buf_out)
            else:
                raise ValueError(mode)

            ys.append(yt)

        y = torch.stack(ys, dim=1).reshape(B, T, H * nv * 4)
        return self.o_proj(self.dropout(y))
    return patched_forward


# ----------------------------------------------------------------------------
# Run all variants
# ----------------------------------------------------------------------------
print('=' * 70)
print('BDN-Q FROZEN-MECHANISM DIAGNOSTIC (ep108, train-best)')
print('=' * 70)

EP = 108
results = {}
for mode in ['normal', 'no_buffer', 'no_delta', 'no_eject', 'no_buf_no_delta']:
    BDeltaQBlock.forward = make_patched_forward(mode)
    model = build_and_load(EP)
    print()
    results[mode] = eval_test(model, f'mode={mode}')

BDeltaQBlock.forward = orig_forward
print()
print('=' * 70)
print('Summary:')
for mode, acc in results.items():
    print(f'  {mode:20s} {acc:.2f}%')
print()
delta_buf  = results['normal'] - results['no_buffer']
delta_dlt  = results['normal'] - results['no_delta']
delta_ejt  = results['normal'] - results['no_eject']
delta_both = results['normal'] - results['no_buf_no_delta']
print(f'  buffer contribution      = {delta_buf:+.2f} pp')
print(f'  delta-state contribution = {delta_dlt:+.2f} pp')
print(f'  eject-into-state contrib = {delta_ejt:+.2f} pp')
print(f'  total temporal read      = {delta_both:+.2f} pp')
print('=' * 70)
