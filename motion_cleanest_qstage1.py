"""REQNN-style Stage 1: replace stage1 MLPBlock with a quaternion-MLP.

Each point's 4 input channels (x, y, z, extra) are treated as a quaternion
(w, x, y, z). All Stage-1 convolutions use Hamilton-product weight structure
(Parcollet 2018 / Shen et al 2024 REQNN). Output features at every layer are
rotation-equivariant by construction.

Position in architecture: BEFORE Stage 1's pool1, BEFORE Stages 2/3/encoder.
No residual escape route around Stage 1, so this is on the main forward path.

Hypothesis: REQNN-style rotation-equivariance at the spatial-feature
extraction step provides genuine inductive bias for gesture motion
(hand pose is rotational), and being on the main path it cannot be bypassed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.motion_cleanest_qms import QuaternionConv2d
from models.motion_cleanest import CleanestLinXLEncoder


class QuaternionMLPBlock(nn.Module):
    """Quaternion-valued MLP for Stage 1 (k=1 Hamilton-product convs)."""
    def __init__(self, channels, with_bn=True):
        super().__init__()
        for c in channels:
            assert c % 4 == 0, f'channel count {c} must be divisible by 4'
        layers = []
        for i in range(len(channels) - 1):
            in_c, out_c = channels[i], channels[i + 1]
            layers.append(QuaternionConv2d(in_c, out_c, kernel_size=(1, 1)))
            if with_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MotionCleanestLinXLQStage1(Motion):
    """CN-XXL encoder + REQNN-style quaternion Stage 1.

    Stage 1's MLPBlock([4,32,64]) is replaced with a Hamilton-product
    quaternion MLP. Everything downstream (Stages 2/3, encoder, classifier)
    is identical to MotionCleanestLinXL.
    """
    def __init__(self, *args, lxl_hidden_dim=256, lxl_mlp_dim=512,
                 lxl_num_layers=4, lxl_dropout=0.3, lxl_bidirectional=True,
                 lxl_residual_scale=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap encoder for CN-XXL
        self.mamba = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale,
        )
        # Replace Stage 1 with quaternion MLP
        # coord_channels=4 (already a quaternion's worth); output 64 channels = 16 quaternions
        self.stage1 = QuaternionMLPBlock([self.coord_channels, 32, 64], with_bn=True)
