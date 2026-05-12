"""Quaternion Hamilton DeltaNet (QHDelta): true quaternion algebra in the recurrence.

Unlike QDeltaNet which uses Euclidean inner products on quaternion-shaped vectors
(equivalent to 4-component real channels in the chunkwise solve), this version uses
Hamilton products throughout — non-commutative quaternion multiplication that defines
the actual rotation algebra.

Recurrence per timestep (per head, matrix state of quaternions S ∈ H^(n_q × n_v)):
    K_S      = k̄ᵀ ⊙_H S            # Hamilton-contract k_conj with S along row index
    K_S_K    = k ⊙_H K_S(outer)    # rank-1 outer update via Hamilton
    S_t = α_t · (S_{t-1} - β_t · K_S_K) + β_t · k_t ⊗_H v̄_t
    y_t = q_t ⊙_H S_t              # quaternion attention readout

Non-commutative product → no chunkwise telescoping, must use per-step Python loop.
T=32 is short enough that per-step is acceptable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


def qmul(p, q):
    """Hamilton product of quaternions, last dim is 4-component."""
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], dim=-1)


def qconj(q):
    """Quaternion conjugate (negate imaginary parts)."""
    out = q.clone()
    out[..., 1:] *= -1
    return out


class QHDeltaNetBlock(nn.Module):
    """Hamilton-product quaternion DeltaNet, per-step recurrence."""
    def __init__(self, d_model, num_heads=4, n_q=8, n_v=16, dropout=0.1, use_short_conv=True, conv_size=4):
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
        self.o_proj = nn.Linear(H * n_q * n_v * 4, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        H, n_q, n_v = self.num_heads, self.n_q, self.n_v

        q_proj = self.q_proj(x); k_proj = self.k_proj(x); v_proj = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            split1 = H * n_q * 4
            split2 = split1 + H * n_q * 4
            q_proj, k_proj, v_proj = qkv[..., :split1], qkv[..., split1:split2], qkv[..., split2:]

        # Reshape to (B, T, H, n_q, 4) and (B, T, H, n_v, 4)
        q = q_proj.view(B, T, H, n_q, 4)
        k = k_proj.view(B, T, H, n_q, 4)
        v = v_proj.view(B, T, H, n_v, 4)

        # L2-normalize k per quaternion (each k_ch is unit quaternion → defines unit reflection)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-9)
        # Activation on q (silu element-wise)
        q = F.silu(q)

        # Per-channel β, α gates in (0, 1)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, n_q)        # write strength
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H, n_q)      # forget rate

        # State: quaternion matrix S ∈ H^(n_q × n_v), per (B, H)
        S = torch.zeros(B, H, n_q, n_v, 4, device=x.device, dtype=x.dtype)
        outs = []

        for t in range(T):
            k_t = k[:, t]            # (B, H, n_q, 4)
            v_t = v[:, t]            # (B, H, n_v, 4)
            q_t = q[:, t]            # (B, H, n_q, 4)
            b_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)    # (B, H, n_q, 1, 1)
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            # Compute K_S = k̄ᵀ ⊙_H S along row index
            # k_conj: (B, H, n_q, 4); broadcast across n_v: (B, H, n_q, 1, 4)
            # S: (B, H, n_q, n_v, 4)
            # Hamilton-multiply elementwise per (i, j), then sum over i
            k_c = qconj(k_t).unsqueeze(-2)                      # (B, H, n_q, 1, 4)
            kS = qmul(k_c.expand(-1, -1, -1, n_v, -1), S)       # (B, H, n_q, n_v, 4)
            kS_sum = kS.sum(dim=-3)                              # (B, H, n_v, 4) sum over n_q

            # Outer: k ⊗_H kS_sum  → (B, H, n_q, n_v, 4)
            k_exp = k_t.unsqueeze(-2).expand(-1, -1, -1, n_v, -1)
            kS_exp = kS_sum.unsqueeze(-3).expand(-1, -1, n_q, -1, -1)
            K_S_K = qmul(k_exp, kS_exp)                          # rank-1 quaternion outer

            # Outer: k ⊗_H v̄_t → (B, H, n_q, n_v, 4)
            v_c = qconj(v_t).unsqueeze(-3).expand(-1, -1, n_q, -1, -1)
            kv = qmul(k_exp, v_c)

            # Update: S = α (S - β K_S_K) + β kv
            S = a_t * (S - b_t * K_S_K) + b_t * kv

            # Output: q ⊙_H S contracted (Hamilton-multiply q[i] with S[i, :], sum over i)
            q_exp = q_t.unsqueeze(-2).expand(-1, -1, -1, n_v, -1)
            qS = qmul(q_exp, S)                                  # (B, H, n_q, n_v, 4)
            outs.append(qS)

        Y = torch.stack(outs, dim=1)                              # (B, T, H, n_q, n_v, 4)
        # Flatten to project: H * n_q * n_v * 4
        y = Y.reshape(B, T, H * n_q * n_v * 4)
        y = self.dropout(y)
        return self.o_proj(y)


class QHDeltaNetTemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=192, output_dim=None, num_layers=2,
                 num_heads=4, n_q=8, n_v=16,
                 use_short_conv=True, conv_size=4, dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_layers = nn.ModuleList([
            QHDeltaNetBlock(hidden_dim, num_heads=num_heads, n_q=n_q, n_v=n_v,
                            use_short_conv=use_short_conv, conv_size=conv_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                QHDeltaNetBlock(hidden_dim, num_heads=num_heads, n_q=n_q, n_v=n_v,
                                use_short_conv=use_short_conv, conv_size=conv_size, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        proj_in = 2 * hidden_dim if bidirectional else hidden_dim
        self.final_norm = nn.LayerNorm(proj_in)
        self.output_proj = nn.Linear(proj_in, self.output_dim)

    def _stack(self, x, layers, norms):
        for blk, norm in zip(layers, norms):
            r = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x) + r
        return x

    def forward(self, x):
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)
        fwd = self._stack(x, self.fwd_layers, self.fwd_norms)
        if self.bidirectional:
            bwd = self._stack(torch.flip(x, dims=[1]), self.bwd_layers, self.bwd_norms)
            bwd = torch.flip(bwd, dims=[1])
            x = torch.cat([fwd, bwd], dim=-1)
        else:
            x = fwd
        x = self.final_norm(x)
        x = self.output_proj(x)
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        return x


class MotionQHDeltaNet(Motion):
    def __init__(self, *args, qh_hidden_dim=192, qh_num_layers=2, qh_num_heads=4,
                 qh_n_q=8, qh_n_v=16, qh_dropout=0.3, qh_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = QHDeltaNetTemporalEncoder(
            in_channels=in_c,
            hidden_dim=qh_hidden_dim,
            output_dim=out_d,
            num_layers=qh_num_layers,
            num_heads=qh_num_heads,
            n_q=qh_n_q,
            n_v=qh_n_v,
            dropout=qh_dropout,
            bidirectional=qh_bidirectional,
        )
