"""Diagnostic: does AttRD's attention-read mechanism actually contribute?

AttRD architecture (per motion_attrd.py):
  Write side: standard RD delta recurrence, building B_acc[tau] = memory state at tau.
  Read side (NEW vs RD): attention over the FULL sequence of memory states:
    V_pair[t, tau] = sum_{n_q} q[t] * B_acc[tau]
    attn[t, tau]   = softmax_tau(read_q[t] . read_k[tau] / sqrt(d_r))
    Y[t]           = sum_tau attn[t, tau] * V_pair[t, tau]

  Standard RD would take only the diagonal: Y[t] = V_pair[t, t] = q[t]^T S[t].

This diagnostic replaces AttRD's softmax-over-tau attention with the identity
attention (one-hot at tau=t), so the attention-read reduces to the standard RD
point-read. Also runs a uniform-attention variant.

If AttRD frozen-to-diagonal == AttRD original  -> attention mechanism is decorative.
If AttRD frozen drops a lot                    -> attention is load-bearing.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import nvidia_dataloader
from models.motion_attrd import MotionAttRD, AttRDBlock


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


def build_and_load_attrd():
    EP = 120
    WORKDIR = 'work_dir/pmamba_baseline_attrd'
    ckpt_path = f'{WORKDIR}/epoch{EP}_model.pt'
    model = MotionAttRD(
        num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
        multi_scale_num_scales=5,
        ar_hidden_dim=128, ar_num_layers=2, ar_num_heads=4,
        ar_n_q=4, ar_n_v=8, ar_d_read=32,
        ar_dropout=0.3, ar_bidirectional=True,
    ).cuda()
    state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()
    return model


def make_patched_forward(attn_mode):
    """Return an AttRDBlock.forward that uses attn_mode in {'identity','uniform'}
    in place of the softmax over the read scores."""
    def patched_forward(self, x):
        from models.motion_attrd import emul, econj, rmatmul
        B, T, D = x.shape
        H, n_q, n_v, d_r = self.num_heads, self.n_q, self.n_v, self.d_read

        q_proj = self.q_proj(x); k_proj = self.k_proj(x); v_proj = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            s1 = H * n_q * 4
            s2 = s1 + H * n_q * 4
            q_proj, k_proj, v_proj = qkv[..., :s1], qkv[..., s1:s2], qkv[..., s2:]

        q = q_proj.view(B, T, H, n_q, 4)
        k = k_proj.view(B, T, H, n_q, 4)
        v = v_proj.view(B, T, H, n_v, 4)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-9)
        q = F.silu(q)

        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, n_q)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H, n_q)

        k_c = econj(k)
        M_kk = emul(k.unsqueeze(-2), k_c.unsqueeze(-3))
        eye_q = torch.zeros(n_q, n_q, 4, device=x.device, dtype=x.dtype)
        eye_q[torch.arange(n_q), torch.arange(n_q), 0] = 1.0
        eye_q = eye_q.expand(B, T, H, n_q, n_q, 4)
        beta_i = beta.unsqueeze(-1).unsqueeze(-1)
        alpha_i = alpha.unsqueeze(-1).unsqueeze(-1)
        A = alpha_i * (eye_q - beta_i * M_kk)

        v_c = econj(v)
        kv = emul(k.unsqueeze(-2), v_c.unsqueeze(-3))
        B_acc = beta.unsqueeze(-1).unsqueeze(-1) * kv
        A_acc = A
        ident_A = eye_q[:, :1]
        zero_B = torch.zeros_like(B_acc[:, :1])
        log_T = max(1, math.ceil(math.log2(max(T, 2))))
        for level in range(log_T):
            step = 1 << level
            if step >= T: break
            earlier_A = torch.cat([ident_A.expand(-1, step, -1, -1, -1, -1),
                                    A_acc[:, :T-step]], dim=1)
            earlier_B = torch.cat([zero_B.expand(-1, step, -1, -1, -1, -1),
                                    B_acc[:, :T-step]], dim=1)
            A_new = rmatmul(A_acc, earlier_A)
            B_new = rmatmul(A_acc, earlier_B) + B_acc
            A_acc, B_acc = A_new, B_new

        V_pair = torch.einsum('bthqc,bshqvc->btshvc', q, B_acc)   # (B,T,T,H,nv,4)

        # Build the patched attention. attn shape per AttRD's einsum 'bths,btshvc->bthvc'
        # is (B, T_t, H, T_tau) — sum over T_tau.
        if attn_mode == 'identity':
            # one-hot at tau=t -> reduces to standard RD point read (diagonal in t,s)
            eye = torch.eye(T, device=x.device, dtype=x.dtype)         # (T_t, T_tau)
            attn = eye.view(1, T, 1, T).expand(B, T, H, T).contiguous()
        elif attn_mode == 'uniform':
            attn = torch.full((B, T, H, T), 1.0 / T, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(attn_mode)

        Y = torch.einsum('bths,btshvc->bthvc', attn, V_pair)
        y = Y.reshape(B, T, H * n_v * 4)
        return self.o_proj(self.dropout(y))
    return patched_forward


# ----------------------------------------------------------------------------
# Run all three: normal AttRD, identity attn (== RD point read), uniform attn
# ----------------------------------------------------------------------------
print('=' * 70)
print('ATTRD ATTENTION-READ DIAGNOSTIC')
print('=' * 70)

orig_forward = AttRDBlock.forward

# 1. Normal AttRD
AttRDBlock.forward = orig_forward
m = build_and_load_attrd()
print()
acc_normal = eval_test(m, 'normal AttRD (softmax attn read)')

# 2. Identity attn (== RD point read)
AttRDBlock.forward = make_patched_forward('identity')
m = build_and_load_attrd()
print()
acc_identity = eval_test(m, 'identity attn (= RD point read, diagonal)')

# 3. Uniform attn (== state averaging)
AttRDBlock.forward = make_patched_forward('uniform')
m = build_and_load_attrd()
print()
acc_uniform = eval_test(m, 'uniform 1/T attn (state averaging)')

AttRDBlock.forward = orig_forward
print()
print('=' * 70)
print('Verdict:')
print(f'  Normal AttRD (softmax attn):  {acc_normal:.2f}%')
print(f'  Frozen to identity (== RD):   {acc_identity:.2f}%')
print(f'  Frozen to uniform 1/T:        {acc_uniform:.2f}%')
print(f'  Delta normal vs identity:     {acc_normal - acc_identity:+.2f} pp')
if abs(acc_normal - acc_identity) < 1.0:
    print('  -> Attention read is DECORATIVE; reduces to RD point read with no loss')
elif acc_normal - acc_identity > 5.0:
    print('  -> Attention read is LOAD-BEARING; identity attn loses > 5pp')
else:
    print('  -> Mixed signal: attention contributes 1-5pp')
print('=' * 70)
