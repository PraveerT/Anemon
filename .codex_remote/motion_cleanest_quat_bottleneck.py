"""CN-XXL frontend + QUATERNION stage5 expansion (not rotation).

Mirrors CN-XXL's real head capacity: the 260->1024 stage5 expansion becomes a
Hamilton-product (quaternion) channel map; pool5 / global_bn / classifier stay
as in CN-XXL. Goal: same capacity, different *mechanism* -> test if the error
pattern diverges from real-head CN-XXL enough to be a fusion partner.

mode='quat' : quaternion 260->1024 expansion (65q -> 256q).
mode='real' : plain conv 260->1024 (≈ CN-XXL head, structure-ablation control).
"""
import torch
import torch.nn as nn

from models.motion import Motion
from models.motion_quat_pointnet import QuaternionLinear


class _ChQuatLinear(nn.Module):
    """QuaternionLinear over the channel dim of (B,C,T,N). C must be 4*in_q."""
    def __init__(self, in_q, out_q):
        super().__init__()
        self.q = QuaternionLinear(in_q, out_q)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        return x.permute(0, 3, 1, 2)


class MotionQuatBottleneck(Motion):
    def __init__(self, *args, expand=1024, head_dropout=0.3, mode='quat',
                 lxl_hidden_dim=None, lxl_mlp_dim=None, lxl_num_layers=None,
                 lxl_dropout=None, lxl_bidirectional=None, lxl_residual_scale=None,
                 framesize=32, quat_head_hidden=None, quat_head_scale=None,
                 q_in=None, q_hidden=None, head_hidden=None, real_width=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        cin = 256 + self.coord_channels                 # 260
        assert cin % 4 == 0 and expand % 4 == 0
        self.act = nn.GELU()
        if mode == 'quat':
            self.expand = _ChQuatLinear(cin // 4, expand // 4)   # 65q -> 256q
        else:
            self.expand = nn.Conv2d(cin, expand, 1)
        self.bn5 = nn.BatchNorm2d(expand)
        self.pool5 = nn.AdaptiveMaxPool2d((1, 1))
        self.global_bn = nn.BatchNorm2d(expand)
        self.drop = nn.Dropout(head_dropout)
        self.classify = nn.Linear(expand, self.num_classes)

    def _head(self, fea3):
        x = self.act(self.bn5(self.expand(fea3)))       # (B,1024,T,N)
        x = self.global_bn(self.pool5(x)).flatten(1)    # (B,1024)
        return self.classify(self.drop(x))

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)
        fea3 = self._encode_sampled_points(coords)       # (B,260,T,N)
        return self._head(fea3)

    def extract_features(self, inputs):
        return self.forward(inputs)
