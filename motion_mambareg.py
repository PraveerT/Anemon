"""N2 with Mamba-Reg (CVPR 2025): register tokens inserted into the per-point
temporal scan to absorb high-norm artifacts and provide a global readout.

Modification over N2:
- Prepend K learnable register tokens to each per-point time series before Mamba.
- After Mamba, slice off the register positions for the downstream classifier
  path (matches N2's spatial pipeline expecting T frames).
- Register tokens are read out separately and added to the final feature via a
  small fusion to provide a global temporal summary.

Otherwise identical to N2's MambaTemporalEncoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from models.motion import Motion, QuaternionLinear


class MambaRegTemporalEncoder(nn.Module):
    """MambaTemporalEncoder with K learnable register tokens prepended per-point.

    Output shape matches the original encoder (B, output_dim, T, N) so the rest of
    Motion's pipeline is unchanged. Register tokens are summarized separately and
    added as a per-point bias to the temporal features.
    """
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2,
                 num_registers=2, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.num_registers = num_registers

        self.input_proj = QuaternionLinear(in_channels, hidden_dim)
        self.register_tokens = nn.Parameter(torch.randn(num_registers, hidden_dim) * 0.02)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = QuaternionLinear(hidden_dim, self.output_dim)
        self.final_norm = nn.LayerNorm(hidden_dim)
        # Register-summary fusion: project (K * hidden_dim) -> output_dim broadcast per-frame
        self.reg_summary_proj = nn.Linear(num_registers * hidden_dim, self.output_dim)

    def forward(self, x):
        # x: (B, C, T, N)
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)                                # (B*N, T, H)
        # Prepend register tokens to each per-point sequence
        regs = self.register_tokens.unsqueeze(0).expand(B * N, -1, -1)  # (B*N, K, H)
        x = torch.cat([regs, x], dim=1)                       # (B*N, K+T, H)

        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x) + residual

        x = self.final_norm(x)
        x = self.output_proj(x)                               # (B*N, K+T, output_dim)
        # Split: register-aggregated summary + temporal positions
        reg_out = x[:, :self.num_registers]                   # (B*N, K, output_dim)
        tem_out = x[:, self.num_registers:]                   # (B*N, T, output_dim)
        # Broadcast register summary as per-frame bias
        reg_flat = reg_out.reshape(B * N, -1)                 # (B*N, K*hidden_or_output)
        # NB: reg_out has output_dim per token, hidden_dim was projected through output_proj already.
        # We want K*output_dim -> output_dim. Adjust the proj input dim:
        # (handled below via reg_summary_proj only if shape matches)
        reg_bias = self.reg_summary_proj(reg_flat).unsqueeze(1)  # (B*N, 1, output_dim)
        tem_out = tem_out + reg_bias                          # broadcast over T

        x = tem_out.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        return x


class MotionMambaReg(Motion):
    """N2 backbone with Mamba-Reg in the temporal trunk."""
    def __init__(self, *args, mr_num_registers=2, mr_dropout=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        hid = old.hidden_dim
        out_d = old.output_dim
        # MR encoder expects reg_summary_proj input = K * output_dim (post-proj tokens)
        self.mamba = MambaRegTemporalEncoder(
            in_channels=in_c,
            hidden_dim=hid,
            output_dim=out_d,
            num_layers=2,
            num_registers=mr_num_registers,
            dropout=mr_dropout,
        )
        # Fix: register-summary projection input dim = K * output_dim
        self.mamba.reg_summary_proj = nn.Linear(mr_num_registers * out_d, out_d)
