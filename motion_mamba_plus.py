"""N2 with upgraded temporal Mamba: drop QuaternionLinear, wider hidden, deeper, bidirectional.

Motivation: QuaternionLinear was an equivariance-flavored constraint (channel-tied 4-component
weights = 4x capacity tax). NVGesture is fixed-camera with no rotation aug, so equivariance
buys nothing here. Drop it, pour capacity into the temporal model instead.

Changes vs original MambaTemporalEncoder:
  - QuaternionLinear input/output projections -> plain nn.Linear (4x more params)
  - hidden_dim 128 -> 256 (2x wider Mamba state)
  - num_layers 2 -> 4 (deeper stack)
  - d_state 16 -> 32 (richer SSM)
  - Add reverse-time Mamba branch, concat forward+reverse outputs (bidirectional)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from models.motion import Motion


class MambaTemporalEncoderPlus(nn.Module):
    """Upgraded temporal encoder. Drop-in replacement for MambaTemporalEncoder."""
    def __init__(self, in_channels, hidden_dim=256, output_dim=None, num_layers=4,
                 d_state=32, d_conv=4, expand=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional

        # Plain linear input projection (replaces QuaternionLinear)
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Forward stack
        self.fwd_layers = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(num_layers)
            ])
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        # Output projection: if bidirectional, fuse 2*hidden -> output_dim; else hidden -> output_dim
        proj_in = 2 * hidden_dim if bidirectional else hidden_dim
        self.output_proj = nn.Linear(proj_in, self.output_dim)
        self.final_norm = nn.LayerNorm(proj_in)

    def _run_stack(self, x, layers, norms):
        for mamba, norm in zip(layers, norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x) + residual
        return x

    def forward(self, x):
        # x: (B, C, T, N) -> (B*N, T, C) for per-point temporal modeling
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)

        fwd = self._run_stack(x, self.fwd_layers, self.fwd_norms)
        if self.bidirectional:
            bwd = self._run_stack(torch.flip(x, dims=[1]), self.bwd_layers, self.bwd_norms)
            bwd = torch.flip(bwd, dims=[1])
            x = torch.cat([fwd, bwd], dim=-1)
        else:
            x = fwd

        x = self.final_norm(x)
        x = self.output_proj(x)
        # Reshape back to (B, output_dim, T, N)
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        return x


class MotionMambaPlus(Motion):
    """N2 backbone with upgraded MambaTemporalEncoderPlus. Drop-in replacement."""
    def __init__(self, *args, mamba_hidden_dim=256, mamba_num_layers=4, mamba_d_state=32,
                 mamba_dropout=0.3, mamba_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = MambaTemporalEncoderPlus(
            in_channels=in_c,
            hidden_dim=mamba_hidden_dim,
            output_dim=out_d,
            num_layers=mamba_num_layers,
            d_state=mamba_d_state,
            dropout=mamba_dropout,
            bidirectional=mamba_bidirectional,
        )
