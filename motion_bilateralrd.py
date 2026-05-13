"""Bilateral Real-DeltaNet (BRD): RD over BOTH temporal (T) and spatial (N) axes.

Standard RD applies the DeltaRule recurrence only along T (time), treating the
N points as a parallel batch. Bilateral RD adds a second RD stream that scans
along N (spatial), capturing which points depend on which within a frame.

Spatial dim N has no canonical order, so the N-stream is always bidirectional.

The two streams are summed (or concatenated) and projected to the output.

At the post-spatial-encoder shape (B, D, T_enc=4, N_enc=8), both axes are tiny,
so the extra N-RD scan is essentially free compared to the spatial encoder.

Story: DeltaRule recurrence is a sequence operator; the sequence dim need not
be time. Applied along N, it learns a per-frame spatial dependency graph at
zero extra encoder cost. Two delta-recurrence axes = bilateral.
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


class RealDeltaNetBlock(nn.Module):
    """Identical to motion_realdeltanet.RealDeltaNetBlock. Inlined so this file
    is self-contained and so the same block can be used for both T and N axes."""
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

        Y = emul(q.unsqueeze(-2), B_acc).sum(dim=-3)
        y = Y.reshape(B, T, H * n_v * 4)
        y = self.dropout(y)
        return self.o_proj(y)


class BilateralRDTemporalEncoder(nn.Module):
    """Two parallel RD streams: one over T, one over N. Sum-fused.

    Input:  (Bz, C, T, N)
    Output: (Bz, output_dim, T, N)
    """
    def __init__(self, in_channels, hidden_dim=128, output_dim=None, num_layers=2,
                 num_heads=4, n_q=4, n_v=8, dropout=0.3, t_bidirectional=True,
                 fuse='sum'):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.t_bidirectional = t_bidirectional
        self.fuse = fuse

        # Separate input projections (the two streams have different "sentence" semantics)
        self.input_proj_t = nn.Linear(in_channels, hidden_dim)
        self.input_proj_n = nn.Linear(in_channels, hidden_dim)

        # T-axis blocks (optionally bidirectional)
        self.t_fwd_blocks = nn.ModuleList([
            RealDeltaNetBlock(hidden_dim, num_heads, n_q, n_v, dropout)
            for _ in range(num_layers)
        ])
        self.t_fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if t_bidirectional:
            self.t_bwd_blocks = nn.ModuleList([
                RealDeltaNetBlock(hidden_dim, num_heads, n_q, n_v, dropout)
                for _ in range(num_layers)
            ])
            self.t_bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # N-axis blocks (always bidirectional — points have no canonical order)
        self.n_fwd_blocks = nn.ModuleList([
            RealDeltaNetBlock(hidden_dim, num_heads, n_q, n_v, dropout)
            for _ in range(num_layers)
        ])
        self.n_fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.n_bwd_blocks = nn.ModuleList([
            RealDeltaNetBlock(hidden_dim, num_heads, n_q, n_v, dropout)
            for _ in range(num_layers)
        ])
        self.n_bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.t_norm = nn.LayerNorm(hidden_dim)
        self.n_norm = nn.LayerNorm(hidden_dim)

        if fuse == 'concat':
            self.output_proj = nn.Linear(2 * hidden_dim, self.output_dim)
        else:  # sum
            self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def _stack(self, x, layers, norms):
        for blk, norm in zip(layers, norms):
            r = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x)
            x = x + r
        return x

    def _t_pass(self, x):
        # x: (Bz, C, T, N) -> (Bz*N, T, C) -> RD over T -> (Bz, D, T, N)
        Bz, C, T, N = x.shape
        x_t = x.permute(0, 3, 2, 1).reshape(Bz * N, T, C)
        x_t = self.input_proj_t(x_t)
        fwd = self._stack(x_t, self.t_fwd_blocks, self.t_fwd_norms)
        out = fwd
        if self.t_bidirectional:
            bwd = self._stack(x_t.flip(1), self.t_bwd_blocks, self.t_bwd_norms).flip(1)
            out = out + bwd
        out = self.t_norm(out)
        # back to (Bz, D, T, N)
        out = out.reshape(Bz, N, T, self.hidden_dim).permute(0, 3, 2, 1)
        return out

    def _n_pass(self, x):
        # x: (Bz, C, T, N) -> (Bz*T, N, C) -> RD over N (bidir) -> (Bz, D, T, N)
        Bz, C, T, N = x.shape
        x_n = x.permute(0, 2, 3, 1).reshape(Bz * T, N, C)
        x_n = self.input_proj_n(x_n)
        fwd = self._stack(x_n, self.n_fwd_blocks, self.n_fwd_norms)
        bwd = self._stack(x_n.flip(1), self.n_bwd_blocks, self.n_bwd_norms).flip(1)
        out = fwd + bwd
        out = self.n_norm(out)
        # back to (Bz, D, T, N)
        out = out.reshape(Bz, T, N, self.hidden_dim).permute(0, 3, 1, 2)
        return out

    def forward(self, x):
        Bz, C, T, N = x.shape
        t_out = self._t_pass(x)
        n_out = self._n_pass(x)
        if self.fuse == 'concat':
            fused = torch.cat([t_out, n_out], dim=1)        # (Bz, 2*D, T, N)
        else:
            fused = t_out + n_out                           # (Bz, D, T, N)
        # output_proj over channel dim
        out = self.output_proj(fused.permute(0, 2, 3, 1))   # (Bz, T, N, out_d)
        return out.permute(0, 3, 1, 2)                      # (Bz, out_d, T, N)


class MotionBilateralRD(Motion):
    """PMamba spatial trunk + Bilateral RD (T-RD + N-RD) temporal head."""
    def __init__(self, *args, brd_hidden_dim=128, brd_num_layers=2, brd_num_heads=4,
                 brd_n_q=4, brd_n_v=8, brd_dropout=0.3, brd_t_bidirectional=True,
                 brd_fuse='sum', **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = BilateralRDTemporalEncoder(
            in_channels=in_c, hidden_dim=brd_hidden_dim, output_dim=out_d,
            num_layers=brd_num_layers, num_heads=brd_num_heads, n_q=brd_n_q,
            n_v=brd_n_v, dropout=brd_dropout, t_bidirectional=brd_t_bidirectional,
            fuse=brd_fuse,
        )
