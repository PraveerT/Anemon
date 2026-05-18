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


def quat_mul(p, q):
    """Hamilton product. p, q: (..., 4) with last dim = (w, x, y, z)."""
    pw, px, py, pz = p.unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    w = pw * qw - px * qx - py * qy - pz * qz
    x = pw * qx + px * qw + py * qz - pz * qy
    y = pw * qy - px * qz + py * qw + pz * qx
    z = pw * qz + px * qy - py * qx + pz * qw
    return torch.stack([w, x, y, z], dim=-1)


def quat_inv(q):
    """Inverse of a unit quaternion = conjugate."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


class QuaternionPoseTrajectoryPool(nn.Module):
    """Replace Stage-4 max-pool with a quaternion-aware trajectory aggregator.

    Per-frame, predicts a unit quaternion q_t in S^3 from the spatially-pooled
    1024-channel features. Aggregates the trajectory {q_1, ..., q_T} via
    quaternion-aware statistics:
      - mean rotation (Frobenius-mean then re-normalize)
      - geodesic spread (S^3 variance proxy)
      - start-to-end relative rotation: q_T * q_1^{-1}
      - cumulative angular velocity proxy

    These quaternion-derived features are concatenated with standard mean/max/
    std temporal pools and projected to out_ch. On the MAIN forward path
    (no residual around it), so the network cannot bypass this aggregation.
    """
    def __init__(self, in_ch=1024, out_ch=1024):
        super().__init__()
        self.q_head = nn.Linear(in_ch, 4)
        # 3 * in_ch (mean+max+std) + 4 (q_mean) + 1 (q_var) + 4 (q_rel) + 3 (q_ang)
        self.compress = nn.Linear(3 * in_ch + 12, out_ch)

    def forward(self, x):                               # x: (B, C, T, N)
        # Spatial max over N for per-frame pooled features
        x_t = x.amax(dim=3)                              # (B, C, T)
        x_tc = x_t.transpose(1, 2)                        # (B, T, C)

        # Standard temporal stats
        s_mean = x_t.mean(dim=2)                          # (B, C)
        s_max  = x_t.amax(dim=2)                          # (B, C)
        s_std  = x_t.std(dim=2)                            # (B, C)

        # Predict per-frame quaternion (unit-norm)
        q = self.q_head(x_tc)                              # (B, T, 4)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        # Quaternion stats
        q_mean = q.mean(dim=1)                             # (B, 4)
        q_mean = q_mean / (q_mean.norm(dim=-1, keepdim=True) + 1e-8)
        q_var  = (q - q_mean.unsqueeze(1)).pow(2).sum(-1).mean(-1, keepdim=True)  # (B,1)
        q_rel  = quat_mul(q[:, -1], quat_inv(q[:, 0]))     # (B, 4)
        # cumulative angular velocity: sum of imaginary parts of q_t - q_{t-1}
        q_diff = q[:, 1:] - q[:, :-1]                       # (B, T-1, 4)
        q_ang  = q_diff[..., 1:].sum(dim=1)                # (B, 3)

        feats = torch.cat([s_mean, s_max, s_std,
                           q_mean, q_var, q_rel, q_ang], dim=-1)  # (B, 3C + 12)
        out = self.compress(feats)                          # (B, out_ch)
        return out.unsqueeze(-1).unsqueeze(-1)              # (B, out_ch, 1, 1)


class MotionCleanestLinXL(Motion):
    """PMamba with wider per-frame MLP encoder (no temporal mixing), param-
    matched to RD. Tests the parameter-count hypothesis: does increasing the
    encoder's param count restore the high-accuracy convergence, even without
    any temporal mechanism?"""
    def __init__(self, *args, lxl_hidden_dim=128, lxl_mlp_dim=256,
                 lxl_num_layers=2, lxl_dropout=0.3, lxl_bidirectional=True,
                 lxl_residual_scale=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale,
        )


class MotionCleanestLinXLQ(MotionCleanestLinXL):
    """CN-XL encoder + Quaternion Pose-Trajectory Pool replacing AdaptiveMaxPool2d."""
    def __init__(self, *args, qpool_in_ch=1024, qpool_out_ch=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool5 = QuaternionPoseTrajectoryPool(qpool_in_ch, qpool_out_ch)
