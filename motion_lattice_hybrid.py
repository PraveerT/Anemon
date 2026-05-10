"""PMamba on hybrid input: depth points + lattice quaternion arrow points.

Input: (B, T, P_depth + 216, 8) with channels [x, y, z, time, q_w, q_x, q_y, q_z]
  - Depth points: q = (1, 0, 0, 0) (identity)
  - Lattice points: q = quaternion encoding deflection from north

Architecture: standard PMamba but with 8-channel input. coord_channels=4 keeps
xyz+time for kNN; the 4 extra channels carry quaternion info that propagates
through stage1's MLP.

A 1×1 projection initialized identity-on-first-4 + zero-on-quat allows the
network to learn how to weight quaternion info during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.motion_qumamba import MotionQuMamba


class MotionLatticeHybrid(nn.Module):
    def __init__(self, num_classes=25, pts_size=472, knn=(32,24,48,24), topk=8,
                 multi_scale_num_scales=5, coord_channels=4, in_channels=8,
                 qumamba_layers=2, dynamic_pts_size=False):
        super().__init__()
        self.in_channels = in_channels
        self.coord_channels = coord_channels
        self.proj = nn.Conv2d(in_channels, coord_channels, kernel_size=1, bias=True)
        with torch.no_grad():
            W = self.proj.weight
            W.zero_()
            for i in range(coord_channels):
                W[i, i, 0, 0] = 1.0
            self.proj.bias.zero_()
        # Base = PMamba with SeQuMamba inside (replaces standard Mamba temporal step)
        self.base = MotionQuMamba(num_classes=num_classes, pts_size=pts_size,
                                   knn=list(knn), topk=topk,
                                   multi_scale_num_scales=multi_scale_num_scales,
                                   coord_channels=coord_channels,
                                   qumamba_layers=qumamba_layers)

    def forward(self, x):
        # x: (B, T, P, C) — typical loader format
        if x.dim() == 4 and x.shape[-1] == self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, T, P)
        x_proj = self.proj(x)
        return self.base(x_proj)
