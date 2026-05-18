"""Cleanest PMamba: replace the entire MambaTemporalEncoder with a simple
per-frame channel bottleneck. No Mamba, no temporal mixing — only the
load-bearing in/out projection identified by the full-bypass diagnostic.

Variants:
  MotionCleanestQuat: uses QuaternionLinear (original)
  MotionCleanestLin:  uses plain nn.Linear (gimmick test)

Forward: fea3 (B, 256, T, N) -> reshape -> Linear(256->128) -> LN(128)
         -> Linear(128->256) -> reshape back -> fea3_clean.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion, QuaternionLinear


class CleanestQuatEncoder(nn.Module):
    """Per-frame QuaternionLinear bottleneck. No Mamba, no recurrence."""
    def __init__(self, in_channels=256, hidden_dim=128, output_dim=256, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_proj = QuaternionLinear(in_channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = QuaternionLinear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1).contiguous()
        return x


class CleanestLinEncoder(nn.Module):
    """Per-frame nn.Linear bottleneck. Same shape as Quat version but no
    Hamilton-product structure. Gimmick test: does quaternion math matter?"""
    def __init__(self, in_channels=256, hidden_dim=128, output_dim=256, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1).contiguous()
        return x


class MotionCleanestQuat(Motion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba = CleanestQuatEncoder()


class MotionCleanestLin(Motion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba = CleanestLinEncoder()
