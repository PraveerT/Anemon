"""Quaternion Hamilton DeltaNet (QHDelta) with PARALLEL SCAN.

Recurrence: S_t = A_t · S_{t-1} + B_t (linear in S → composable → Hillis-Steele scan).

A_t = diag(α_t) · (I - diag(β_t) · M_kk_t)        ∈ H^(n_q × n_q)
M_kk_t[i,j] = qmul(k_t[i], qconj(k_t[j]))         (quaternion outer product matrix)
B_t[i,j] = β_t[i] · qmul(k_t[i], qconj(v_t[j]))   (rank-1 quaternion outer)

Composition (associative): (A_a, B_a) ⊕ (A_b, B_b) = (A_b @ A_a, A_b @ B_a + B_b)
where @ is quaternion matrix-Hamilton product.

Hillis-Steele inclusive scan in O(log T) levels = 5 levels for T=32.
Replaces the sequential Python loop (1s/forward) with batched parallel ops.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


def qmul(p, q):
    """Hamilton product. Last dim is 4-component."""
    pw, px, py, pz = p[..., 0:1], p[..., 1:2], p[..., 2:3], p[..., 3:4]
    qw, qx, qy, qz = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    return torch.cat([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], dim=-1)


def qconj(q):
    """Quaternion conjugate."""
    out = q.clone()
    out[..., 1:] *= -1
    return out


def hmatmul(A, B):
    """Quaternion matrix-Hamilton product. A: (..., m, k, 4); B: (..., k, n, 4) → (..., m, n, 4)."""
    A_e = A.unsqueeze(-2)                    # (..., m, k, 1, 4)
    B_e = B.unsqueeze(-4)                    # (..., 1, k, n, 4)
    H = qmul(A_e, B_e)                       # (..., m, k, n, 4) broadcast Hamilton
    return H.sum(dim=-3)                     # sum over k


class QHDeltaNetBlock(nn.Module):
    """Hamilton-product DeltaNet with Hillis-Steele parallel scan over time."""
    def __init__(self, d_model, num_heads=4, n_q=4, n_v=8, dropout=0.1, use_short_conv=True, conv_size=4):
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

        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, n_q)        # (B, T, H, n_q)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H, n_q)

        # Build A_t and B_t per timestep (vectorized over T)
        # M_kk[..., t, i, j] = qmul(k[..., t, i], qconj(k[..., t, j]))
        k_c = qconj(k)                                                     # (B, T, H, n_q, 4)
        M_kk = qmul(k.unsqueeze(-2), k_c.unsqueeze(-3))                    # (B, T, H, n_q, n_q, 4)

        # A_t[i, j] = α_t[i] (δ_{ij} - β_t[i] M_kk_t[i, j])
        # Identity quaternion matrix: diagonal entries = (1,0,0,0), off = 0
        eye_q = torch.zeros(n_q, n_q, 4, device=x.device, dtype=x.dtype)
        eye_q[torch.arange(n_q), torch.arange(n_q), 0] = 1.0
        eye_q = eye_q.expand(B, T, H, n_q, n_q, 4)
        beta_i = beta.unsqueeze(-1).unsqueeze(-1)                          # (B, T, H, n_q, 1, 1)
        alpha_i = alpha.unsqueeze(-1).unsqueeze(-1)
        A = alpha_i * (eye_q - beta_i * M_kk)                              # (B, T, H, n_q, n_q, 4)

        # B_t[i, j] = β_t[i] · qmul(k_t[i], qconj(v_t[j]))
        v_c = qconj(v)                                                      # (B, T, H, n_v, 4)
        kv = qmul(k.unsqueeze(-2), v_c.unsqueeze(-3))                       # (B, T, H, n_q, n_v, 4)
        B_acc = beta.unsqueeze(-1).unsqueeze(-1) * kv                      # (B, T, H, n_q, n_v, 4)
        A_acc = A                                                           # (B, T, H, n_q, n_q, 4)

        # Hillis-Steele inclusive scan over T axis (dim=1)
        # Identity matrix (4D quaternion) for padding
        ident_A = eye_q[:, :1]                                              # (B, 1, H, n_q, n_q, 4)
        zero_B = torch.zeros_like(B_acc[:, :1])

        log_T = max(1, math.ceil(math.log2(T)))
        for level in range(log_T):
            step = 1 << level
            if step >= T:
                break
            # earlier[t] = acc[t-step] for t >= step, else identity
            earlier_A = torch.cat([ident_A.expand(-1, step, -1, -1, -1, -1),
                                    A_acc[:, :T-step]], dim=1)
            earlier_B = torch.cat([zero_B.expand(-1, step, -1, -1, -1, -1),
                                    B_acc[:, :T-step]], dim=1)
            # Compose: (A_acc) @ (earlier_A), (A_acc) @ (earlier_B) + B_acc
            A_new = hmatmul(A_acc, earlier_A)
            B_new = hmatmul(A_acc, earlier_B) + B_acc
            A_acc, B_acc = A_new, B_new

        # B_acc[t] = S_t (since S_0 = 0)
        # Output: y_t = q_t @ S_t (Hamilton-multiply over n_q axis)
        # q: (B, T, H, n_q, 4); S: (B, T, H, n_q, n_v, 4)
        # y[..., j] = sum_i qmul(q[..., i], S[..., i, j])
        Y = qmul(q.unsqueeze(-2), B_acc).sum(dim=-3)                       # (B, T, H, n_v, 4)

        # Wait — current readout uses full n_q × n_v matrix output. Original used flatten H*n_q*n_v*4
        # New: contracted to H*n_v*4. Changing readout dim affects o_proj.
        # Stay consistent with the simpler readout (sum over n_q):
        y = Y.reshape(B, T, H * n_v * 4)
        # Adjust o_proj input dim accordingly (caller must rebuild o_proj for this layer size)
        y = self.dropout(y)
        return self.o_proj(y)


class QHDeltaNetTemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=192, output_dim=None, num_layers=2,
                 num_heads=4, n_q=4, n_v=8,
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
        # Override o_proj to accept H * n_v * 4 (after sum over n_q)
        for blk in self.fwd_layers:
            blk.o_proj = nn.Linear(num_heads * n_v * 4, hidden_dim, bias=False)
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                QHDeltaNetBlock(hidden_dim, num_heads=num_heads, n_q=n_q, n_v=n_v,
                                use_short_conv=use_short_conv, conv_size=conv_size, dropout=dropout)
                for _ in range(num_layers)
            ])
            for blk in self.bwd_layers:
                blk.o_proj = nn.Linear(num_heads * n_v * 4, hidden_dim, bias=False)
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
                 qh_n_q=4, qh_n_v=8, qh_dropout=0.3, qh_bidirectional=True, **kwargs):
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
