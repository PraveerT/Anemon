"""CN-XXL backbone + SECOND-ORDER (bilinear/covariance) pooling head.

Motivated by the error analysis: CN-XXL's failures are systematic confusions
between similar gestures (fine-grained), and its head is first-order
(stage5 -> AdaptiveMaxPool), which discards the feature co-occurrence structure
that separates similar classes. We add a covariance-pooling branch alongside the
original max-pool branch (so it can only ADD, never underperform the baseline):

  feat = [ maxpool(stage5)  (1024) ||  bilinear(reduce(stage5))  (d*d) ]  -> classify

Same kNN frontend + LinXL block + recipe as CN-XXL for a fair comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.motion_cleanest import CleanestLinXLEncoder


class MotionBilinear(Motion):
    def __init__(self, *args, d=96, head_dropout=0.3,
                 lxl_hidden_dim=256, lxl_mlp_dim=512, lxl_num_layers=4,
                 lxl_dropout=0.3, lxl_bidirectional=True, lxl_residual_scale=0.7,
                 framesize=32, **kwargs):
        super().__init__(*args, **kwargs)
        # match CN-XXL backbone: LinXL temporal encoder in stage-3 slot
        self.mamba = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale)
        self.mamba.in_channels = 256
        self.mamba.output_dim = 256
        self.d = d
        self.reduce = nn.Conv2d(1024, d, 1)           # stage5(1024) -> d for 2nd-order
        self.bn2 = nn.BatchNorm1d(d * d)
        self.drop = nn.Dropout(head_dropout)
        self.classify = nn.Linear(1024 + d * d, self.num_classes)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)
        fea3 = self._encode_sampled_points(coords)        # (B, 260, T, N)
        s5 = self.stage5(fea3)                            # (B, 1024, T, N)
        # first-order branch (= CN-XXL signal)
        fo = self.global_bn(self.pool5(s5)).flatten(1)    # (B, 1024)
        # second-order (covariance) branch
        r = self.reduce(s5).flatten(2)                    # (B, d, T*N)
        G = torch.bmm(r, r.transpose(1, 2)) / r.shape[-1]  # (B, d, d)
        bp = G.flatten(1)
        bp = torch.sign(bp) * torch.sqrt(bp.abs() + 1e-6)  # signed sqrt
        bp = self.bn2(bp)
        bp = F.normalize(bp, dim=1)                        # L2 norm
        feat = torch.cat([fo, bp], dim=1)
        return self.classify(self.drop(feat))

    def extract_features(self, inputs):
        return self.forward(inputs)
