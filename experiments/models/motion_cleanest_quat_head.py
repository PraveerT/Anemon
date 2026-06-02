"""CN-XXL + Quaternion Trajectory Head: parallel aux classifier from per-frame
inertia quaternions, ensembled with the main classifier output.

Idea: even if the spatial pipeline already captures most of what the inertia
quat encodes, a separate weak-but-uncorrelated classifier from the quat
trajectory will average out residual error, yielding a small ensemble lift
without architectural risk.
"""
import torch
import torch.nn as nn

from models.motion import Motion
from models.motion_cleanest import CleanestLinXLEncoder
from models.motion_cleanest_quat import _inertia_quat


class MotionCleanestLinXLQuatHead(Motion):
    def __init__(self, *args, lxl_hidden_dim=128, lxl_mlp_dim=256,
                 lxl_num_layers=2, lxl_dropout=0.3, lxl_bidirectional=True,
                 lxl_residual_scale=0.7, framesize=32,
                 quat_head_hidden=128, quat_head_scale=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottleneck = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale,
        )
        self.framesize = framesize
        self.quat_head = nn.Sequential(
            nn.Linear(framesize * 4, quat_head_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(quat_head_hidden, self.num_classes),
        )
        self.quat_head_scale = nn.Parameter(torch.tensor(float(quat_head_scale)))

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)            # (B, 4, T, N)
        # Aux: inertia-quat trajectory -> small MLP -> aux logits
        B = coords.shape[0]
        T = coords.shape[2]
        quat = _inertia_quat(coords[:, :3])             # (B*T, 4)
        quat_traj = quat.reshape(B, T * 4)
        aux_logits = self.quat_head(quat_traj)          # (B, num_classes)

        # Main path
        fea3 = self._encode_sampled_points(coords)
        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        main_logits = self.classify_features(output.flatten(1))

        return main_logits + self.quat_head_scale * aux_logits
