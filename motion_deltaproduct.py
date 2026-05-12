"""DeltaProduct (arXiv 2502.10297): H Householder reflections per timestep.

Strictly more expressive than DeltaNet (H=1). For state tracking and parity
tasks the paper shows clear gains. We test on N1 (raw depth) input where
DeltaNet v2 already best-performed.

Implementation: expand the time axis from T to T*H by generating H independent
(k_h, v_h, β_h) triples per original timestep. Apply chunkwise plain DeltaNet
solve on the expanded length, sample output at every H-th position.

q is shared across the H sub-steps within each original timestep — only the
last sub-step's output is used.

Math per timestep:
  S_{t,0} = S_{t-1}
  for h in 1..H:
    S_{t,h} = (I - β_{t,h} k_{t,h} k_{t,h}^T) S_{t,h-1} + β_{t,h} k_{t,h} v_{t,h}^T
  S_t = S_{t,H}
  y_t = q_t^T S_t
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class DeltaProductBlock(nn.Module):
    """Chunkwise DeltaProduct on (B, T, D) -> (B, T, D). H Householders per step."""
    def __init__(self, d_model, num_heads=4, head_dim=64, expand_v=2,
                 num_householder=2, use_short_conv=True, conv_size=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = head_dim * int(expand_v)
        self.H = num_householder
        H_h, dk, dv = num_heads, head_dim, self.value_dim

        # q is shared across the H sub-steps (one per original timestep)
        self.q_proj = nn.Linear(d_model, H_h * dk, bias=False)
        # k, v, β have H copies (one per sub-step / Householder)
        self.k_proj = nn.Linear(d_model, self.H * H_h * dk, bias=False)
        self.v_proj = nn.Linear(d_model, self.H * H_h * dv, bias=False)
        self.beta_proj = nn.Linear(d_model, self.H * H_h)

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        if use_short_conv:
            ch = H_h * dk + self.H * (H_h * dk + H_h * dv)
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(H_h * dv, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        H_h, dk, dv = self.num_heads, self.head_dim, self.value_dim
        H = self.H

        q = self.q_proj(x)                                 # (B, T, H_h*dk)
        k = self.k_proj(x)                                 # (B, T, H*H_h*dk)
        v = self.v_proj(x)                                 # (B, T, H*H_h*dv)
        if self.use_short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            split1 = H_h * dk
            split2 = split1 + H * H_h * dk
            q, k, v = qkv[..., :split1], qkv[..., split1:split2], qkv[..., split2:]

        # Reshape into per-sub-step view
        # q: (B, T, H_h, dk) — broadcast across H sub-steps
        q = q.view(B, T, H_h, dk)
        # k, v: (B, T, H, H_h, dk_or_dv) — H sub-steps within each step
        k = k.view(B, T, H, H_h, dk)
        v = v.view(B, T, H, H_h, dv)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, H_h)

        # Stack to expanded length T*H along time axis
        # k_seq[b, t*H + h] = k[b, t, h]
        k_seq = k.reshape(B, T * H, H_h, dk)
        v_seq = v.reshape(B, T * H, H_h, dv)
        beta_seq = beta.reshape(B, T * H, H_h)
        # q_seq: repeat each step's q across H sub-steps
        q_seq = q.unsqueeze(2).expand(B, T, H, H_h, dk).reshape(B, T * H, H_h, dk)

        # Move to per-head: (B, H_h, T*H, dk_or_dv)
        q_seq = q_seq.permute(0, 2, 1, 3).contiguous()
        k_seq = k_seq.permute(0, 2, 1, 3).contiguous()
        v_seq = v_seq.permute(0, 2, 1, 3).contiguous()
        beta_seq = beta_seq.permute(0, 2, 1).contiguous()  # (B, H_h, T*H)

        # Normalize keys
        k_seq = F.normalize(k_seq, dim=-1)
        q_seq = F.silu(q_seq)

        # Chunkwise plain DeltaNet on length T*H
        L_total = T * H
        device = x.device
        # KK[i, j] = k_i · k_j  (B, H_h, L, L)
        KK = torch.einsum('bhid,bhjd->bhij', k_seq, k_seq)
        mask_lt = torch.ones(L_total, L_total, device=device, dtype=torch.bool).tril(diagonal=-1)
        L_mat = beta_seq.unsqueeze(-2) * KK
        L_mat = L_mat * mask_lt.to(L_mat.dtype)
        eye = torch.eye(L_total, device=device, dtype=L_mat.dtype).expand(B, H_h, L_total, L_total)
        I_plus_L = eye + L_mat
        V_prime = torch.linalg.solve_triangular(I_plus_L, v_seq, upper=False, unitriangular=True)
        # A[i,j] = β_j (q_i · k_j) for j<=i, 0 elsewhere
        QK = torch.einsum('bhid,bhjd->bhij', q_seq, k_seq)
        mask_le = torch.ones(L_total, L_total, device=device, dtype=torch.bool).tril(diagonal=0)
        A = beta_seq.unsqueeze(-2) * QK
        A = A * mask_le.to(A.dtype)
        Y_expanded = torch.matmul(A, V_prime)              # (B, H_h, L_total, dv)

        # Sample output at every H-th position (last sub-step of each original step)
        Y_expanded = Y_expanded.permute(0, 2, 1, 3).contiguous()  # (B, L_total, H_h, dv)
        Y = Y_expanded.view(B, T, H, H_h, dv)[:, :, -1]    # (B, T, H_h, dv)
        y = Y.reshape(B, T, H_h * dv)
        y = self.dropout(y)
        return self.o_proj(y)


class DeltaProductTemporalEncoder(nn.Module):
    """Bidirectional DeltaProduct stack."""
    def __init__(self, in_channels, hidden_dim=256, output_dim=None, num_layers=2,
                 num_heads=4, head_dim=64, expand_v=2, num_householder=2,
                 use_short_conv=True, conv_size=4, dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_layers = nn.ModuleList([
            DeltaProductBlock(hidden_dim, num_heads=num_heads, head_dim=head_dim,
                              expand_v=expand_v, num_householder=num_householder,
                              use_short_conv=use_short_conv, conv_size=conv_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                DeltaProductBlock(hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                  expand_v=expand_v, num_householder=num_householder,
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


class MotionDeltaProduct(Motion):
    def __init__(self, *args, dp_hidden_dim=256, dp_num_layers=2, dp_num_heads=4,
                 dp_head_dim=64, dp_expand_v=2, dp_num_householder=2,
                 dp_dropout=0.3, dp_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = DeltaProductTemporalEncoder(
            in_channels=in_c,
            hidden_dim=dp_hidden_dim,
            output_dim=out_d,
            num_layers=dp_num_layers,
            num_heads=dp_num_heads,
            head_dim=dp_head_dim,
            expand_v=dp_expand_v,
            num_householder=dp_num_householder,
            dropout=dp_dropout,
            bidirectional=dp_bidirectional,
        )
