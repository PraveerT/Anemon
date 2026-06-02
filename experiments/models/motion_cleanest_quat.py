"""CN-XXL + InertiaQuat: per-frame inertia-tensor quaternion branch.

Without point correspondence, we cannot do Hamilton products of consecutive
frames. Instead, summarize each frame's spatial distribution via the top
eigenvector of its xyz covariance, encode (axis, angle) as a unit quaternion,
and project that 4-vector to feature space. Inject additively at fea3 level.

Cheap (1 small MLP), correspondence-free, differentiable.
"""
import torch
import torch.nn as nn

from models.motion import Motion
from models.motion_cleanest import CleanestLinXLEncoder


def _inertia_quat(xyz):
    """xyz: (B, 3, T, N) → quaternion per frame: (B*T, 4).

    Centers points per frame, computes covariance, takes principal axis
    (top eigenvector) and principal magnitude (top eigenvalue) as the
    rotation axis and angle of a unit quaternion.
    """
    B, _, T, N = xyz.shape
    centered = xyz - xyz.mean(dim=-1, keepdim=True)
    # (B, 3, T, N) -> (B*T, 3, N)
    flat = centered.permute(0, 2, 1, 3).contiguous().reshape(B * T, 3, N)
    cov = torch.bmm(flat, flat.transpose(1, 2)) / max(N - 1, 1)
    # eigh returns ascending eigenvalues; last is largest
    eigvals, eigvecs = torch.linalg.eigh(cov)
    axis = eigvecs[..., -1]              # (B*T, 3)
    angle = eigvals[..., -1].clamp(min=0).sqrt()  # use std-magnitude as angle
    half = (angle * 0.5).unsqueeze(-1)
    quat = torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)
    return quat


class InertiaQuatBranch(nn.Module):
    def __init__(self, out_channels=256, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels),
        )

    def forward(self, coords):
        # coords: (B, C, T, N), C ≥ 3 (xyz are first three channels)
        B, _, T, N = coords.shape
        quat = _inertia_quat(coords[:, :3])           # (B*T, 4)
        feat = self.mlp(quat)                          # (B*T, out_channels)
        return feat.reshape(B, T, -1).permute(0, 2, 1) # (B, out_channels, T)


class MotionCleanestLinXLQuat(Motion):
    """CN-XXL with inertia-quaternion branch injected at fea3."""
    def __init__(self, *args, lxl_hidden_dim=128, lxl_mlp_dim=256,
                 lxl_num_layers=2, lxl_dropout=0.3, lxl_bidirectional=True,
                 lxl_residual_scale=0.7, quat_mlp_hidden=64,
                 quat_inject_scale=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottleneck = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale,
        )
        self.quat_branch = InertiaQuatBranch(out_channels=256, hidden=quat_mlp_hidden)
        # Learnable scale (init small so it can't break the baseline at ep0).
        self.quat_scale = nn.Parameter(torch.tensor(float(quat_inject_scale)))

    def _encode_sampled_points(self, coords):
        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[0]
        ret_group_array2 = self.group.st_group_points(
            fea1, 3, [0, 1, 2], self.knn[1], 3, coord_dim=self.coord_channels,
        )
        ret_array2, coords = self.select_ind(
            ret_group_array2, coords, batchsize, in_dims, timestep, pts_num,
        )
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[1]
        ret_group_array3 = self.group.st_group_points(
            fea2, 3, [0, 1, 2], self.knn[2], 3, coord_dim=self.coord_channels,
        )
        ret_array3, coords = self.select_ind(
            ret_group_array3, coords, batchsize, in_dims, timestep, pts_num,
        )
        fea3 = self.pool3(self.stage3(ret_array3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_enc = self.bottleneck(fea3)

        # Inject inertia-quat branch (per-frame, broadcast over N).
        quat_feat = self.quat_branch(coords)            # (B, 256, T)
        quat_feat = quat_feat.unsqueeze(-1)             # (B, 256, T, 1)
        fea3_enc = fea3_enc + self.quat_scale * quat_feat

        return torch.cat((coords, fea3_enc), dim=1)
