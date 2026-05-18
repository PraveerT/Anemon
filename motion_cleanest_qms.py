"""B2: Quaternion Multi-Scale Temporal Convolution (Parcollet-style Hamilton-
product convs replacing the 5 scale filters in MultiScaleFeatureProcessor).

Replaces Stage 2's MultiScaleFeatureProcessor with a version where each
scale_filter is a QuaternionConv2d using genuine Hamilton-product weight
structure. The rest of Stage 2 (scale_interaction, output_proj, residual)
is preserved.

Rationale: Stage 2's multi-scale processing is on the main forward path
(verified by NoTemporal floor = 84.85 vs ~89 with multi_scale). Adding
quaternion structure here adds genuine algebraic content where it matters,
unlike the Stage-3 encoder where any block is bypassed by residual.

Hyperparams match CN-XXL exactly elsewhere; only the multi_scale conv
filters change.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion, MultiScaleFeatureProcessor
from models.motion_cleanest import CleanestLinXLEncoder, QuaternionPoseTrajectoryPool


class QuaternionConv2d(nn.Module):
    """Parcollet-style quaternion 2D conv. Hamilton-product weight structure.

    Both in_channels and out_channels must be divisible by 4.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        assert in_channels % 4 == 0, f"in_channels {in_channels} not divisible by 4"
        assert out_channels % 4 == 0, f"out_channels {out_channels} not divisible by 4"
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.in_q = in_channels // 4
        self.out_q = out_channels // 4
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # 4 weight matrices: each (out_q, in_q, kH, kW)
        scale = (in_channels) ** -0.5
        self.W_r = nn.Parameter(torch.randn(self.out_q, self.in_q, *kernel_size) * scale)
        self.W_i = nn.Parameter(torch.randn(self.out_q, self.in_q, *kernel_size) * scale)
        self.W_j = nn.Parameter(torch.randn(self.out_q, self.in_q, *kernel_size) * scale)
        self.W_k = nn.Parameter(torch.randn(self.out_q, self.in_q, *kernel_size) * scale)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _full_weight(self):
        """Compose the 4*out_q x 4*in_q x kH x kW weight tensor via Hamilton structure."""
        return torch.cat([
            torch.cat([self.W_r, -self.W_i, -self.W_j, -self.W_k], dim=1),
            torch.cat([self.W_i,  self.W_r, -self.W_k,  self.W_j], dim=1),
            torch.cat([self.W_j,  self.W_k,  self.W_r, -self.W_i], dim=1),
            torch.cat([self.W_k, -self.W_j,  self.W_i,  self.W_r], dim=1),
        ], dim=0)

    def forward(self, x):
        return F.conv2d(x, self._full_weight(), self.bias,
                        stride=self.stride, padding=self.padding)


class MultiScaleFeatureProcessorQ(nn.Module):
    """MultiScale processor with quaternion scale filters."""
    def __init__(self, in_channels, num_scales=5, feature_dim=32):
        super().__init__()
        # Round in_channels and feature_dim to multiples of 4 if needed
        assert in_channels % 4 == 0, f"in_channels {in_channels} must be divisible by 4 for quaternion conv"
        assert feature_dim % 4 == 0, f"feature_dim {feature_dim} must be divisible by 4 for quaternion conv"
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.feature_dim = feature_dim

        # Quaternion scale filters
        self.scale_filters = nn.ModuleList([
            QuaternionConv2d(in_channels, feature_dim, kernel_size=(2**i, 1),
                             stride=(2**i, 1), padding=(2**(i-1), 0))
            for i in range(1, num_scales + 1)
        ])

        # Scale interaction (standard conv, no quaternion required)
        self.scale_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Conv2d(feature_dim, feature_dim, 1),
            ) for _ in range(num_scales - 1)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(feature_dim * num_scales + in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        B, _, T, N = x.shape
        scale_features = [f(x) for f in self.scale_filters]
        interacted = [scale_features[0]]
        for i in range(len(scale_features) - 1):
            source = F.interpolate(scale_features[i],
                                   size=(scale_features[i+1].shape[2], N),
                                   mode='bilinear', align_corners=False)
            target = scale_features[i + 1]
            combined = torch.cat([source, target], dim=1)
            interaction = self.scale_interaction[i](combined)
            interacted.append(target + interaction)
        all_features = [F.interpolate(f, size=(T, N), mode='bilinear', align_corners=False)
                        for f in interacted]
        combined_features = torch.cat(all_features, dim=1)
        combined = torch.cat([x, combined_features], dim=1)
        output = self.output_proj(combined)
        return output + x


class MotionCleanestLinXLQMS(Motion):
    """CN-XXL encoder + Quaternion Multi-Scale Processor (B2).
    Stage 2's multi_scale uses Hamilton-product quaternion convs.
    Pool stays as the standard AdaptiveMaxPool2d (we already showed quaternion
    pool doesn't help; this isolates the multi-scale change)."""
    def __init__(self, *args, lxl_hidden_dim=256, lxl_mlp_dim=512,
                 lxl_num_layers=4, lxl_dropout=0.3, lxl_bidirectional=True,
                 lxl_residual_scale=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap encoder for CN-XXL Lin
        self.mamba = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale,
        )
        # Swap multi_scale for quaternion version
        ms_in = self.multi_scale.in_channels
        ms_scales = self.multi_scale.num_scales
        ms_feat = self.multi_scale.feature_dim
        self.multi_scale = MultiScaleFeatureProcessorQ(
            in_channels=ms_in, num_scales=ms_scales, feature_dim=ms_feat,
        )
