"""Quaternion DeltaNet: chunkwise delta-rule with quaternion-valued K, V, Q.

Combines DeltaNet's delta-rule selectivity with SeQuMamba's quaternion structure.
SO(3)-invariance via magnitude readout (free from quaternion algebra).

Formulation:
- Per head, n_quat channels each a 4-component quaternion
- K, V, Q ∈ H^(n_quat) per timestep — 4D each
- β ∈ R per timestep per head
- Inner product (k_i · k_j) is real (sum of 4-vector products) → chunkwise solve real
- State S ∈ H^(n_quat × n_value_quat), updated by quaternion delta rule
- Output: magnitude of state per channel (SO(3)-invariant)

Chunkwise math (per head):
    L[i,j] = β_j (k_i · k_j) for j<i, else 0     (real, lower triangular)
    V' = (I + L)^{-1} V                          (V quaternion-valued, solve per component)
    A[i,j] = β_j (q_i · k_j) for j<=i, else 0    (real)
    Y = A @ V'                                   (quaternion vectors weighted by real scalars)
    y_readout = ||Y||                            (per-channel magnitude → SO(3)-invariant)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class QuaternionDeltaNetBlock(nn.Module):
    """Chunkwise quaternion delta-rule with magnitude readout."""
    def __init__(self, d_model, num_heads=4, n_quat=16, value_quat=32,
                 use_short_conv=True, conv_size=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_quat = n_quat            # quaternion channels per head for K, Q
        self.value_quat = value_quat    # quaternion channels per head for V
        H, n_q, n_v = num_heads, n_quat, value_quat

        # Each "quaternion" = 4 reals. Project to (H * n_quat * 4) and (H * n_value * 4).
        self.q_proj = nn.Linear(d_model, H * n_q * 4, bias=False)
        self.k_proj = nn.Linear(d_model, H * n_q * 4, bias=False)
        self.v_proj = nn.Linear(d_model, H * n_v * 4, bias=False)
        self.beta_proj = nn.Linear(d_model, H)

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        if use_short_conv:
            ch = H * n_q * 4 * 2 + H * n_v * 4
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        # Output projection from per-head magnitudes (n_value real values per head)
        self.o_proj = nn.Linear(H * n_v, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        H, n_q, n_v = self.num_heads, self.n_quat, self.value_quat

        q = self.q_proj(x)                                # (B, T, H * n_q * 4)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            split1 = H * n_q * 4
            split2 = split1 + H * n_q * 4
            q, k, v = qkv[..., :split1], qkv[..., split1:split2], qkv[..., split2:]

        # Reshape per head per quaternion: (B, H, T, n_quat, 4)
        q = q.view(B, T, H, n_q, 4).permute(0, 2, 1, 3, 4).contiguous()
        k = k.view(B, T, H, n_q, 4).permute(0, 2, 1, 3, 4).contiguous()
        v = v.view(B, T, H, n_v, 4).permute(0, 2, 1, 3, 4).contiguous()

        # L2-normalize keys across ALL quaternion channels jointly
        # (per-channel norm makes inner product range [-n_quat, n_quat] which blows up the triangular solve;
        #  joint norm keeps inner product bounded by Cauchy-Schwarz at ±1)
        k = k.reshape(B, H, T, n_q * 4)
        k = F.normalize(k, dim=-1)
        k = k.reshape(B, H, T, n_q, 4)
        q = F.silu(q)

        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)  # (B, H, T)

        # Quaternion inner product → real scalar
        # KK[b, h, i, j] = sum over quat channels of (k_i · k_j) (4-vector dot)
        # = einsum: (b h i c q) (b h j c q) -> (b h i j)
        KK = torch.einsum('bhicq,bhjcq->bhij', k, k)
        device = x.device
        mask_lt = torch.ones(T, T, device=device, dtype=torch.bool).tril(diagonal=-1)
        L = beta.unsqueeze(-2) * KK
        L = L * mask_lt.to(L.dtype)
        eye = torch.eye(T, device=device, dtype=L.dtype).expand(B, H, T, T)
        I_plus_L = eye + L
        # V is (B, H, T, n_v, 4) — flatten last two dims for triangular solve
        V_flat = v.reshape(B, H, T, n_v * 4)
        V_prime_flat = torch.linalg.solve_triangular(I_plus_L, V_flat, upper=False, unitriangular=True)
        V_prime = V_prime_flat.view(B, H, T, n_v, 4)

        # A[b, h, i, j] = β_j (q_i · k_j) for j<=i
        QK = torch.einsum('bhicq,bhjcq->bhij', q, k)
        mask_le = torch.ones(T, T, device=device, dtype=torch.bool).tril(diagonal=0)
        A = beta.unsqueeze(-2) * QK
        A = A * mask_le.to(A.dtype)
        # Y[b, h, t, n_v, 4] = sum_s A[b, h, t, s] · V'[b, h, s, n_v, 4]
        Y = torch.einsum('bhts,bhscq->bhtcq', A, V_prime)        # quaternion-valued

        # Magnitude readout per channel — SO(3)-invariant
        # Use eps-stable form (norm has NaN gradient at zero)
        Y_mag = (Y.pow(2).sum(dim=-1) + 1e-9).sqrt()              # (B, H, T, n_v) real

        y = Y_mag.permute(0, 2, 1, 3).reshape(B, T, H * n_v)
        y = self.dropout(y)
        return self.o_proj(y)


class QuaternionDeltaNetTemporalEncoder(nn.Module):
    """Bidirectional quaternion DeltaNet stack."""
    def __init__(self, in_channels, hidden_dim=256, output_dim=None, num_layers=2,
                 num_heads=4, n_quat=16, value_quat=32,
                 use_short_conv=True, conv_size=4, dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_layers = nn.ModuleList([
            QuaternionDeltaNetBlock(hidden_dim, num_heads=num_heads,
                                    n_quat=n_quat, value_quat=value_quat,
                                    use_short_conv=use_short_conv,
                                    conv_size=conv_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                QuaternionDeltaNetBlock(hidden_dim, num_heads=num_heads,
                                        n_quat=n_quat, value_quat=value_quat,
                                        use_short_conv=use_short_conv,
                                        conv_size=conv_size, dropout=dropout)
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


class MotionQDeltaNet(Motion):
    def __init__(self, *args, qdn_hidden_dim=256, qdn_num_layers=2, qdn_num_heads=4,
                 qdn_n_quat=16, qdn_value_quat=32, qdn_dropout=0.3,
                 qdn_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = QuaternionDeltaNetTemporalEncoder(
            in_channels=in_c,
            hidden_dim=qdn_hidden_dim,
            output_dim=out_d,
            num_layers=qdn_num_layers,
            num_heads=qdn_num_heads,
            n_quat=qdn_n_quat,
            value_quat=qdn_value_quat,
            dropout=qdn_dropout,
            bidirectional=qdn_bidirectional,
        )
