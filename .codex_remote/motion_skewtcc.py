"""CN-XXL backbone + Skew-Symmetric Temporal Cross-Covariance pooling (Skew-TCC).

Error analysis: CN-XXL's errors are ANTISYMMETRIC (mirror 440->121, time-reversal
440->363), so the missing signal flips sign under reflection/time-reversal.
Symmetric covariance pooling is provably blind to this. Skew-TCC reads the
ANTISYMMETRIC part of a low-rank LAGGED feature cross-covariance, which flips sign
under both time-reversal and feature/channel swap.

Per-frame feature z_t (spatial max over N) -> two low-rank projectors U=zW_u, V=zW_v
(rank r) -> lagged cross-Gram C_d = sum_t u_t v_{t+d}^T -> take off-diagonal of:
  mode='skew'  : A_d = (C_d - C_d^T)/2     (the contribution)
  mode='sym'   : S_d = (C_d + C_d^T)/2     (matched symmetric control)
  mode='random': skew, but projectors FROZEN at init (decorative-content control)
Descriptor (tril off-diagonal, identical length for skew/sym) is concatenated onto
the existing first-order pooled feature -> existing classifier. So it can only ADD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.motion_cleanest import CleanestLinXLEncoder


class MotionSkewTCC(Motion):
    def __init__(self, *args, r=12, lags=(1, 2), head_dropout=0.3, mode='skew',
                 lxl_hidden_dim=256, lxl_mlp_dim=512, lxl_num_layers=4,
                 lxl_dropout=0.3, lxl_bidirectional=True, lxl_residual_scale=0.7,
                 framesize=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale)
        self.mamba.in_channels = 256
        self.mamba.output_dim = 256
        self.r = int(r)
        self.lags = tuple(lags)
        self.mode = mode
        self.Wu = nn.Linear(1024, self.r, bias=False)
        self.Wv = nn.Linear(1024, self.r, bias=False)
        if mode == 'random':                      # frozen-projector control
            for p in list(self.Wu.parameters()) + list(self.Wv.parameters()):
                p.requires_grad = False
        # strictly-lower-triangle indices (i>j): identical length for skew & sym
        idx = torch.tril_indices(self.r, self.r, offset=-1)
        self.register_buffer('ti', idx)
        desc_len = self.r * (self.r - 1) // 2 * len(self.lags)
        self.desc_bn = nn.BatchNorm1d(desc_len)
        self.drop = nn.Dropout(head_dropout)
        self.classify = nn.Linear(1024 + desc_len, self.num_classes)

    def _desc(self, z):                            # z: (B, T, 1024)
        U, V = self.Wu(z), self.Wv(z)              # (B, T, r)
        T = z.shape[1]
        outs = []
        for d in self.lags:
            u, v = U[:, :T - d], V[:, d:]          # (B, T-d, r)
            C = torch.einsum('bti,btj->bij', u, v) / max(1, T - d)   # (B, r, r)
            M = (C - C.transpose(1, 2)) * 0.5 if self.mode in ('skew', 'random') else (C + C.transpose(1, 2)) * 0.5
            outs.append(M[:, self.ti[0], self.ti[1]])   # (B, r(r-1)/2)
        return torch.cat(outs, dim=1)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)
        fea3 = self._encode_sampled_points(coords)        # (B, 260, T, N)
        s5 = self.stage5(fea3)                            # (B, 1024, T, N)
        fo = self.global_bn(self.pool5(s5)).flatten(1)    # (B, 1024) first-order = CN-XXL signal
        z = s5.amax(dim=3).transpose(1, 2)                # (B, T, 1024) per-frame spatial-max
        desc = self.desc_bn(self._desc(z))                # (B, desc_len)
        feat = torch.cat([fo, self.drop(desc)], dim=1)
        return self.classify(feat)

    def extract_features(self, inputs):
        return self.forward(inputs)
