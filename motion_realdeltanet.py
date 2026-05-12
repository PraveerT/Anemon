"""Real-valued DeltaNet ablation of QHDeltaNet.

Same architecture and parameter count as QHDeltaNet, but replaces Hamilton
products with element-wise products (treats 4-component features as 4
INDEPENDENT real channels with no cross-component mixing).

Tests whether the quaternion Hamilton-product structure is the source of
QHDeltaNet's signal, or whether the architectural skeleton (DeltaNet
recurrence + parallel scan) is sufficient on its own.

Replacements vs QHDeltaNet:
  qmul(p, q) -> elementwise p * q
  qconj(q)   -> q (identity; no sign flip on imaginary parts)
  hmatmul    -> standard real matmul with last-axis broadcasting
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


def emul(p, q):
    """Elementwise product over last dim (4-component). NO Hamilton structure."""
    return p * q


def econj(q):
    """Identity — no conjugation in real ablation."""
    return q


def rmatmul(A, B):
    """Standard real matmul over last-2 dims, with elementwise mixing along last axis.

    A: (..., m, k, 4), B: (..., k, n, 4) -> (..., m, n, 4)
    Sum over k (matrix structure preserved), elementwise over 4-component.
    """
    A_e = A.unsqueeze(-2)            # (..., m, k, 1, 4)
    B_e = B.unsqueeze(-4)            # (..., 1, k, n, 4)
    H = emul(A_e, B_e)               # (..., m, k, n, 4)
    return H.sum(dim=-3)             # sum over k


class RealDeltaNetBlock(nn.Module):
    """DeltaNet block, real-valued ablation (no Hamilton products)."""
    def __init__(self, d_model, num_heads=4, n_q=4, n_v=8, dropout=0.1,
                 use_short_conv=True, conv_size=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_q = n_q
        self.n_v = n_v
        H = num_heads

        self.q_proj = nn.Linear(d_model, H * n_q * 4, bias=False)
        self.k_proj = nn.Linear(d_model, H * n_q * 4, bias=False)
        self.v_proj = nn.Linear(d_model, H * n_v * 4, bias=False)
        self.beta_proj = nn.Linear(d_model, H * n_q)
        self.alpha_proj = nn.Linear(d_model, H * n_q)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            ch = H * n_q * 4 * 2 + H * n_v * 4
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(H * n_v * 4, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, n_q, n_v = self.num_heads, self.n_q, self.n_v

        q_proj = self.q_proj(x); k_proj = self.k_proj(x); v_proj = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            split1 = H * n_q * 4
            split2 = split1 + H * n_q * 4
            q_proj, k_proj, v_proj = qkv[..., :split1], qkv[..., split1:split2], qkv[..., split2:]

        q = q_proj.view(B, T, H, n_q, 4)
        k = k_proj.view(B, T, H, n_q, 4)
        v = v_proj.view(B, T, H, n_v, 4)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-9)
        q = F.silu(q)

        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, n_q)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H, n_q)

        # M_kk[..., t, i, j] = emul(k[..., t, i], econj(k[..., t, j]))  -- elementwise
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

        log_T = max(1, math.ceil(math.log2(T)))
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

        Y = emul(q.unsqueeze(-2), B_acc).sum(dim=-3)   # (B, T, H, n_v, 4)
        y = Y.reshape(B, T, H * n_v * 4)
        y = self.dropout(y)
        return self.o_proj(y)


# Re-use QHDeltaNet's encoder/wrapper structure with the real-valued block plugged in.
# We import the existing module hierarchy and just substitute the block class.

import models.motion_qhdeltanet as _qhmod


class RealDeltaNetTemporalEncoder(_qhmod.QHDeltaNetTemporalEncoder if hasattr(_qhmod, 'QHDeltaNetTemporalEncoder') else nn.Module):
    """Temporal encoder: identical to QHDeltaNetTemporalEncoder but uses RealDeltaNetBlock.

    Falls back to a minimal local definition if the upstream class name differs.
    """
    def __init__(self, in_channels, hidden_dim=192, output_dim=None, num_layers=2,
                 num_heads=4, n_q=4, n_v=8, dropout=0.3, bidirectional=True):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_blocks = nn.ModuleList([
            RealDeltaNetBlock(hidden_dim, num_heads, n_q, n_v, dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_blocks = nn.ModuleList([
                RealDeltaNetBlock(hidden_dim, num_heads, n_q, n_v, dropout)
                for _ in range(num_layers)
            ])
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def _stack(self, x, layers, norms):
        for blk, norm in zip(layers, norms):
            residual = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x)
            x = x + residual
        return x

    def forward(self, x):
        # x: (B, C, T, N) -> (B*N, T, C) -> proj to hidden
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


class MotionRealDeltaNet(Motion):
    """PMamba with RealDeltaNet (no Hamilton products) replacing the temporal step."""
    def __init__(self, *args, rd_hidden_dim=128, rd_num_layers=2, rd_num_heads=4,
                 rd_n_q=4, rd_n_v=8, rd_dropout=0.3, rd_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = RealDeltaNetTemporalEncoder(
            in_channels=in_c, hidden_dim=rd_hidden_dim, output_dim=out_d,
            num_layers=rd_num_layers, num_heads=rd_num_heads, n_q=rd_n_q,
            n_v=rd_n_v, dropout=rd_dropout, bidirectional=rd_bidirectional,
        )
