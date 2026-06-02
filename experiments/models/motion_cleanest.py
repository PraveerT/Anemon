"""Cleanest the model: replace the entire MambaTemporalEncoder with a simple
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

from models.motion import Motion



class CleanestLinXLEncoder(nn.Module):
    """Wider per-frame MLP encoder with bidirectional residual stack but NO
    temporal mixing. Param-matched to RealDeltaNetTemporalEncoder (~0.29M).

    Mirrors RD's structure (input_proj -> [LN -> block -> +residual]*L -> final_norm
    -> output_proj, both fwd and bwd, summed) but replaces each delta-rule block
    with a per-frame MLP (Linear -> GELU -> Linear -> Dropout). Each frame is
    processed independently — zero temporal mixing.

    Hypothesis: if this reaches RD-softres-comparable accuracy, then "the block
    is training-essential" reduces to "parameter budget in the encoder slot",
    not anything specific to the temporal mechanism.
    """
    def __init__(self, in_channels=256, hidden_dim=128, mlp_dim=256, output_dim=256,
                 num_layers=2, dropout=0.3, bidirectional=True, residual_scale=0.7):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.residual_scale = float(residual_scale)
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        def mlp_block():
            return nn.Sequential(
                nn.Linear(hidden_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, hidden_dim),
            )

        self.fwd_blocks = nn.ModuleList([mlp_block() for _ in range(num_layers)])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_blocks = nn.ModuleList([mlp_block() for _ in range(num_layers)])
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def _stack(self, x, blocks, norms):
        for blk, norm in zip(blocks, norms):
            residual = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x)
            x = x + self.residual_scale * residual
        return x

    def forward(self, x):
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().reshape(B * N, T, C)
        x = self.input_proj(x)
        fwd = self._stack(x, self.fwd_blocks, self.fwd_norms)
        out = fwd
        if self.bidirectional:
            bwd = self._stack(x.flip(1), self.bwd_blocks, self.bwd_norms).flip(1)
            out = out + bwd
        out = self.final_norm(out)
        out = self.output_proj(out)
        out = out.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1).contiguous()
        return out



class MotionCleanestLinXL(Motion):
    """the model with wider per-frame MLP encoder (no temporal mixing), param-
    matched to RD. Tests the parameter-count hypothesis: does increasing the
    encoder's param count restore the high-accuracy convergence, even without
    any temporal mechanism?"""
    def __init__(self, *args, lxl_hidden_dim=128, lxl_mlp_dim=256,
                 lxl_num_layers=2, lxl_dropout=0.3, lxl_bidirectional=True,
                 lxl_residual_scale=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottleneck = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale,
        )


