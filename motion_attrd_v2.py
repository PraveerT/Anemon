"""AttRD-v2: state-conditioned attention-read + output gate.

Improvements over AttRD (89.00 train-best / 89.63 test-best, 91.91 in DSN+AttRD pair):

1. **State-conditioned read keys**: read_k_tau is projected from the MEMORY STATE
   B_acc[tau] (the actual DeltaRule-built state), not from the input x_tau.
   This makes the attention scoring select states based on what is *stored*
   in them, not what was *fed in* at that timestep — a stronger architectural
   distinction from RD's pure input-conditioned read.

2. **Output gate** (Mamba/SSM style): y = sigmoid(W_g x_t) * o_proj(Y_attn).
   Adds per-token selective gating on the output of the attention read,
   giving AttRD-v2 a write-side-untouched but read-side-selective mechanism.

3. **Larger d_read** (32 -> 64): more capacity in attention scoring.

The write recurrence (RD DeltaRule) is unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


def emul(p, q):
    return p * q


def econj(q):
    return q


def rmatmul(A, B):
    A_e = A.unsqueeze(-2)
    B_e = B.unsqueeze(-4)
    H = emul(A_e, B_e)
    return H.sum(dim=-3)


class AttRDv2Block(nn.Module):
    def __init__(self, d_model, num_heads=4, n_q=4, n_v=8, d_read=64, dropout=0.1,
                 use_short_conv=True, conv_size=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_q = n_q
        self.n_v = n_v
        self.d_read = d_read
        H = num_heads

        # Write-side projections (RD-identical)
        self.q_proj = nn.Linear(d_model, H * n_q * 4, bias=False)
        self.k_proj = nn.Linear(d_model, H * n_q * 4, bias=False)
        self.v_proj = nn.Linear(d_model, H * n_v * 4, bias=False)
        self.beta_proj = nn.Linear(d_model, H * n_q)
        self.alpha_proj = nn.Linear(d_model, H * n_q)

        # Read-side: q from input, k from STATE (per RD head)
        # State summary is flattened (n_q * n_v * 4) per head.
        self.read_q_proj = nn.Linear(d_model, H * d_read, bias=False)
        self.read_k_proj_state = nn.Linear(n_q * n_v * 4, d_read, bias=False)

        # Output gate (per-head, per-value-channel)
        self.output_gate = nn.Linear(d_model, H * n_v * 4)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            ch = H * n_q * 4 * 2 + H * n_v * 4
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(H * n_v * 4, d_model, bias=False)

    def forward(self, x):
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

        # --- Write recurrence (RD-style, unchanged) ---
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
            if step >= T:
                break
            earlier_A = torch.cat([ident_A.expand(-1, step, -1, -1, -1, -1),
                                    A_acc[:, :T-step]], dim=1)
            earlier_B = torch.cat([zero_B.expand(-1, step, -1, -1, -1, -1),
                                    B_acc[:, :T-step]], dim=1)
            A_new = rmatmul(A_acc, earlier_A)
            B_new = rmatmul(A_acc, earlier_B) + B_acc
            A_acc, B_acc = A_new, B_new

        # B_acc shape: (B, T, H, n_q, n_v, 4) — memory state at each tau

        # --- State-conditioned attention READ ---
        # 1) Per-(t, tau) point-read result
        V_pair = torch.einsum('bthqc,bshqvc->btshvc', q, B_acc)  # (B, T, T, H, n_v, 4)

        # 2) Attention: q from x, k from STATE (B_acc flattened per RD head)
        rq = self.read_q_proj(x).view(B, T, H, d_r)
        state_flat = B_acc.reshape(B, T, H, n_q * n_v * 4)
        rk = self.read_k_proj_state(state_flat)            # (B, T, H, d_read)
        scores = torch.einsum('bthd,bshd->bths', rq, rk) / math.sqrt(d_r)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # 3) Weighted sum over tau
        Y = torch.einsum('bths,btshvc->bthvc', attn, V_pair)
        y = Y.reshape(B, T, H * n_v * 4)

        # --- Output gate ---
        gate = torch.sigmoid(self.output_gate(x))          # (B, T, H*n_v*4)
        y = y * gate
        y = self.dropout(y)
        return self.o_proj(y)


class AttRDv2TemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, output_dim=None, num_layers=2,
                 num_heads=4, n_q=4, n_v=8, d_read=64, dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_blocks = nn.ModuleList([
            AttRDv2Block(hidden_dim, num_heads, n_q, n_v, d_read, dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_blocks = nn.ModuleList([
                AttRDv2Block(hidden_dim, num_heads, n_q, n_v, d_read, dropout)
                for _ in range(num_layers)
            ])
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def _stack(self, x, layers, norms):
        for blk, norm in zip(layers, norms):
            r = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x)
            x = x + r
        return x

    def forward(self, x):
        Bz, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(Bz * N, T, C)
        x = self.input_proj(x)
        fwd = self._stack(x, self.fwd_blocks, self.fwd_norms)
        out = fwd
        if self.bidirectional:
            bwd = self._stack(x.flip(1), self.bwd_blocks, self.bwd_norms).flip(1)
            out = out + bwd
        out = self.final_norm(out)
        out = self.output_proj(out)
        out = out.reshape(Bz, N, T, self.output_dim).permute(0, 3, 2, 1)
        return out


class MotionAttRDv2(Motion):
    """PMamba spatial trunk + AttRD-v2 (state-conditioned read + output gate)."""
    def __init__(self, *args, av2_hidden_dim=128, av2_num_layers=2, av2_num_heads=4,
                 av2_n_q=4, av2_n_v=8, av2_d_read=64, av2_dropout=0.3,
                 av2_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = AttRDv2TemporalEncoder(
            in_channels=in_c, hidden_dim=av2_hidden_dim, output_dim=out_d,
            num_layers=av2_num_layers, num_heads=av2_num_heads, n_q=av2_n_q,
            n_v=av2_n_v, d_read=av2_d_read, dropout=av2_dropout,
            bidirectional=av2_bidirectional,
        )
