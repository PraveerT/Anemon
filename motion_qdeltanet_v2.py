"""Quaternion DeltaNet v2: full quaternion readout (all 4 components used).

v1 used magnitude readout |Y| for SO(3)-invariance — but on fixed-camera
NVGesture there's no rotation to be invariant to, and the magnitude readout
discards 75% of the state content (only the unsigned magnitude survives).

v2 keeps all 4 quaternion components and lets a learned linear layer combine them.
Loses the SO(3) theorem but gains 4× output information bandwidth.

Same chunkwise quaternion delta-rule as v1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class QuaternionDeltaNetV2Block(nn.Module):
    def __init__(self, d_model, num_heads=4, n_quat=16, value_quat=32,
                 use_short_conv=True, conv_size=4, dropout=0.1, readout='full'):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_quat = n_quat
        self.value_quat = value_quat
        self.readout = readout  # 'full' (4 components) or 'mag' (magnitude only)
        H, n_q, n_v = num_heads, n_quat, value_quat

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
        # Output projection: full = H * n_v * 4 (4× more); mag = H * n_v
        out_features = H * n_v * (4 if readout == 'full' else 1)
        self.o_proj = nn.Linear(out_features, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, n_q, n_v = self.num_heads, self.n_quat, self.value_quat

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            split1 = H * n_q * 4
            split2 = split1 + H * n_q * 4
            q, k, v = qkv[..., :split1], qkv[..., split1:split2], qkv[..., split2:]

        q = q.view(B, T, H, n_q, 4).permute(0, 2, 1, 3, 4).contiguous()
        k = k.view(B, T, H, n_q, 4).permute(0, 2, 1, 3, 4).contiguous()
        v = v.view(B, T, H, n_v, 4).permute(0, 2, 1, 3, 4).contiguous()

        # Joint normalize K across all quaternion channels (Cauchy-Schwarz bound)
        k_flat = k.reshape(B, H, T, n_q * 4)
        k_flat = F.normalize(k_flat, dim=-1)
        k = k_flat.reshape(B, H, T, n_q, 4)
        q = F.silu(q)

        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)

        KK = torch.einsum('bhicq,bhjcq->bhij', k, k)
        device = x.device
        mask_lt = torch.ones(T, T, device=device, dtype=torch.bool).tril(diagonal=-1)
        L = beta.unsqueeze(-2) * KK
        L = L * mask_lt.to(L.dtype)
        eye = torch.eye(T, device=device, dtype=L.dtype).expand(B, H, T, T)
        I_plus_L = eye + L
        V_flat = v.reshape(B, H, T, n_v * 4)
        V_prime_flat = torch.linalg.solve_triangular(I_plus_L, V_flat, upper=False, unitriangular=True)
        V_prime = V_prime_flat.view(B, H, T, n_v, 4)

        QK = torch.einsum('bhicq,bhjcq->bhij', q, k)
        mask_le = torch.ones(T, T, device=device, dtype=torch.bool).tril(diagonal=0)
        A = beta.unsqueeze(-2) * QK
        A = A * mask_le.to(A.dtype)
        Y = torch.einsum('bhts,bhscq->bhtcq', A, V_prime)        # (B, H, T, n_v, 4)

        if self.readout == 'mag':
            Y_out = (Y.pow(2).sum(dim=-1) + 1e-9).sqrt()         # (B, H, T, n_v)
            y = Y_out.permute(0, 2, 1, 3).reshape(B, T, H * n_v)
        else:
            # Full: keep all 4 quaternion components per channel
            y = Y.permute(0, 2, 1, 3, 4).reshape(B, T, H * n_v * 4)

        y = self.dropout(y)
        return self.o_proj(y)


class QuaternionDeltaNetV2TemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, output_dim=None, num_layers=2,
                 num_heads=4, n_quat=16, value_quat=32, readout='full',
                 use_short_conv=True, conv_size=4, dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_layers = nn.ModuleList([
            QuaternionDeltaNetV2Block(hidden_dim, num_heads=num_heads,
                                      n_quat=n_quat, value_quat=value_quat,
                                      readout=readout,
                                      use_short_conv=use_short_conv,
                                      conv_size=conv_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                QuaternionDeltaNetV2Block(hidden_dim, num_heads=num_heads,
                                          n_quat=n_quat, value_quat=value_quat,
                                          readout=readout,
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


class MotionQDeltaNetV2(Motion):
    def __init__(self, *args, qdn_hidden_dim=256, qdn_num_layers=2, qdn_num_heads=4,
                 qdn_n_quat=16, qdn_value_quat=32, qdn_readout='full', qdn_dropout=0.3,
                 qdn_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = QuaternionDeltaNetV2TemporalEncoder(
            in_channels=in_c,
            hidden_dim=qdn_hidden_dim,
            output_dim=out_d,
            num_layers=qdn_num_layers,
            num_heads=qdn_num_heads,
            n_quat=qdn_n_quat,
            value_quat=qdn_value_quat,
            readout=qdn_readout,
            dropout=qdn_dropout,
            bidirectional=qdn_bidirectional,
        )
