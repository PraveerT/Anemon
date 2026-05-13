"""Real-DeltaProduct: RD architecture + K Householder reflections per timestep.

RD ablation kept the QHDeltaNet skeleton (4-fold channel split, n_q/n_v outer
products, Hillis-Steele parallel scan, bidirectional) but dropped Hamilton
products. RD scored 90.46 solo on NVGesture N1.

DeltaProduct (Schlag et al., arXiv:2502.10297) generalizes the single delta
step per token to K Householder reflections per token. K=1 reduces to RD.
K>1 is strictly more expressive (state-tracking + parity tasks).

Per-token recurrence (K Householders):
  S_t = a_t * (I - b_K k_K k_K^T) ... (I - b_1 k_1 k_1^T) * S_{t-1}
        + sum_{j=1..K} (I - b_K k_K k_K^T) ... (I - b_{j+1} k_{j+1} k_{j+1}^T)
                       * b_j k_j v_j^T

Implementation: compose the K factors inside each token to produce a single
per-token (A_t, B_t) pair, then plug into the existing log-T Hillis-Steele
scan over T. K is small (2 or 3), so the inner loop is cheap.
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
    """(..., m, k, 4) x (..., k, n, 4) -> (..., m, n, 4). Sum over k; elementwise over 4."""
    A_e = A.unsqueeze(-2)
    B_e = B.unsqueeze(-4)
    H = emul(A_e, B_e)
    return H.sum(dim=-3)


class RealDeltaProductBlock(nn.Module):
    """RD block but with K Householder reflections per timestep.

    Each token emits K independent (k_j, v_j, beta_j) triples plus a single
    (q, alpha). The K factors compose inside the token to produce one
    per-token A and B for the standard parallel scan.
    """
    def __init__(self, d_model, num_heads=4, n_q=4, n_v=8, num_householder=2,
                 dropout=0.1, use_short_conv=True, conv_size=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_q = n_q
        self.n_v = n_v
        self.K = num_householder
        H, K = num_heads, num_householder

        # q and alpha are single per timestep
        self.q_proj = nn.Linear(d_model, H * n_q * 4, bias=False)
        self.alpha_proj = nn.Linear(d_model, H * n_q)
        # k, v, beta have K copies (one per Householder sub-step)
        self.k_proj = nn.Linear(d_model, K * H * n_q * 4, bias=False)
        self.v_proj = nn.Linear(d_model, K * H * n_v * 4, bias=False)
        self.beta_proj = nn.Linear(d_model, K * H * n_q)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            ch = H * n_q * 4 + K * (H * n_q * 4 + H * n_v * 4)
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(H * n_v * 4, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, n_q, n_v, K = self.num_heads, self.n_q, self.n_v, self.K

        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            s1 = H * n_q * 4
            s2 = s1 + K * H * n_q * 4
            q_proj, k_proj, v_proj = qkv[..., :s1], qkv[..., s1:s2], qkv[..., s2:]

        q = q_proj.view(B, T, H, n_q, 4)
        k = k_proj.view(B, T, K, H, n_q, 4)
        v = v_proj.view(B, T, K, H, n_v, 4)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-9)
        q = F.silu(q)

        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, K, H, n_q)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H, n_q)

        # Build per-token A_t (erase) and B_t (write) by composing K Householders.
        # Convention: apply j=1 first, j=K last (so left-multiplications stack).
        eye_q = torch.zeros(n_q, n_q, 4, device=x.device, dtype=x.dtype)
        eye_q[torch.arange(n_q), torch.arange(n_q), 0] = 1.0
        eye_q_b = eye_q.expand(B, T, H, n_q, n_q, 4)

        A_t = eye_q_b
        B_t = torch.zeros(B, T, H, n_q, n_v, 4, device=x.device, dtype=x.dtype)

        for j in range(K):
            k_j = k[:, :, j]                                        # (B,T,H,n_q,4)
            v_j = v[:, :, j]                                        # (B,T,H,n_v,4)
            beta_j = beta[:, :, j]                                  # (B,T,H,n_q)
            k_jc = econj(k_j)
            M_kk_j = emul(k_j.unsqueeze(-2), k_jc.unsqueeze(-3))    # (B,T,H,n_q,n_q,4)
            beta_ji = beta_j.unsqueeze(-1).unsqueeze(-1)
            fac_j = eye_q_b - beta_ji * M_kk_j                      # (B,T,H,n_q,n_q,4)
            # left-multiply: A_new = fac_j @ A_t; B_new = fac_j @ B_t + beta_j k_j v_j^T
            A_t = rmatmul(fac_j, A_t)
            v_jc = econj(v_j)
            kv_j = emul(k_j.unsqueeze(-2), v_jc.unsqueeze(-3))      # (B,T,H,n_q,n_v,4)
            write_j = beta_ji * kv_j
            B_t = rmatmul(fac_j, B_t) + write_j

        alpha_i = alpha.unsqueeze(-1).unsqueeze(-1)
        A_acc = alpha_i * A_t
        B_acc = B_t

        # Standard log-T Hillis-Steele parallel scan over T.
        ident_A = eye_q_b[:, :1]
        zero_B = torch.zeros_like(B_acc[:, :1])

        log_T = max(1, math.ceil(math.log2(T)))
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


class RealDeltaProductTemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, output_dim=None, num_layers=2,
                 num_heads=4, n_q=4, n_v=8, num_householder=2, dropout=0.3,
                 bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_blocks = nn.ModuleList([
            RealDeltaProductBlock(hidden_dim, num_heads, n_q, n_v, num_householder, dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_blocks = nn.ModuleList([
                RealDeltaProductBlock(hidden_dim, num_heads, n_q, n_v, num_householder, dropout)
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


class MotionRealDeltaProduct(Motion):
    """PMamba spatial trunk + RealDeltaProduct (K Householder) temporal head."""
    def __init__(self, *args, rdp_hidden_dim=128, rdp_num_layers=2, rdp_num_heads=4,
                 rdp_n_q=4, rdp_n_v=8, rdp_num_householder=2, rdp_dropout=0.3,
                 rdp_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = RealDeltaProductTemporalEncoder(
            in_channels=in_c, hidden_dim=rdp_hidden_dim, output_dim=out_d,
            num_layers=rdp_num_layers, num_heads=rdp_num_heads, n_q=rdp_n_q,
            n_v=rdp_n_v, num_householder=rdp_num_householder, dropout=rdp_dropout,
            bidirectional=rdp_bidirectional,
        )
