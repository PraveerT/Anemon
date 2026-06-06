"""PGCNet + backbone-decoupled Skew-TCC aux head (the 91.91 result).

The aux's input is the antisymmetric lagged cross-covariance (Skew-TCC) of a raw-
coordinate per-frame summary -- the per-frame 3x3 covariance of the sampled xyz
points -- NOT backbone features. Because it's a fixed transform of the RAW points,
its gradients update only the aux head, never the backbone: a clean parallel
regularizer, logit-ensembled with the main classifier (out = main + scale*aux).

  mode='skew'   : A_d = (C_d - C_d^T)/2   (the contribution)
  mode='sym'    : S_d = (C_d + C_d^T)/2   (matched symmetric control)
  mode='random' : skew, projectors FROZEN at init (decorative-content control)
  mode='off'    : aux disabled -> out = main_logits (no-aux floor, same arch/seed)
Per-frame raw feature z_t = flatten(cov(xyz_t)) in R^9; rank-r projectors U=zW_u,
V=zW_v; lagged cross-Gram C_d = mean_t u_t v_{t+d}^T; tril off-diag -> aux MLP ->
out = main_logits + scale * aux_logits.
"""
import torch
import torch.nn as nn

from models.base import Base
from models.encoder import PGCNetEncoder


class PGCNetSkewRawHead(Base):
    def __init__(self, *args, lxl_hidden_dim=128, lxl_mlp_dim=256,
                 lxl_num_layers=2, lxl_dropout=0.3, lxl_bidirectional=True,
                 lxl_residual_scale=0.7, framesize=32,
                 r=12, lags=(1, 2), mode='skew', head_dropout=0.3,
                 skew_head_hidden=128, skew_head_scale=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottleneck = PGCNetEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale,
        )
        self.framesize = framesize
        self.r = int(r)
        self.lags = tuple(lags)
        self.mode = mode
        self.Wu = nn.Linear(9, self.r, bias=False)        # 9 = flattened 3x3 cov
        self.Wv = nn.Linear(9, self.r, bias=False)
        if mode == 'random':
            for p in list(self.Wu.parameters()) + list(self.Wv.parameters()):
                p.requires_grad = False
        idx = torch.tril_indices(self.r, self.r, offset=-1)
        self.register_buffer('ti', idx)
        desc_len = self.r * (self.r - 1) // 2 * len(self.lags)
        self.skew_head = nn.Sequential(
            nn.Linear(desc_len, skew_head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(skew_head_hidden, self.num_classes),
        )
        self.skew_head_scale = nn.Parameter(torch.tensor(float(skew_head_scale)))

    def _per_frame_cov(self, coords):                     # coords (B, C>=3, T, N)
        xyz = coords[:, :3]                                # (B, 3, T, N)
        B, _, T, N = xyz.shape
        x = xyz.permute(0, 2, 1, 3)                        # (B, T, 3, N)
        x = x - x.mean(dim=-1, keepdim=True)               # center per frame
        cov = torch.matmul(x, x.transpose(-1, -2)) / max(N - 1, 1)  # (B, T, 3, 3)
        return cov.reshape(B, T, 9)                        # (B, T, 9) raw, backbone-free

    def _desc(self, z):                                    # z: (B, T, 9)
        U, V = self.Wu(z), self.Wv(z)                      # (B, T, r)
        T = z.shape[1]
        outs = []
        for d in self.lags:
            if T - d <= 0:
                outs.append(z.new_zeros(z.shape[0], self.r * (self.r - 1) // 2)); continue
            u, v = U[:, :T - d], V[:, d:]
            C = torch.einsum('bti,btj->bij', u, v) / max(1, T - d)
            M = (C - C.transpose(1, 2)) * 0.5 if self.mode in ('skew', 'random') else (C + C.transpose(1, 2)) * 0.5
            outs.append(M[:, self.ti[0], self.ti[1]])
        return torch.cat(outs, dim=1)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)               # (B, 4, T, N) RAW
        # MAIN path -- identical to the quat head
        fea3 = self._encode_sampled_points(coords)
        s5 = self.stage5(fea3)
        main_logits = self.classify_features(self.global_bn(self.pool5(s5)).flatten(1))
        if self.mode == 'off':                             # no-aux floor: main path only
            return main_logits
        # AUX: decoupled skew/sym from RAW per-frame covariance -- NO backbone gradient
        aux_logits = self.skew_head(self._desc(self._per_frame_cov(coords)))
        return main_logits + self.skew_head_scale * aux_logits

    def extract_features(self, inputs):
        return self.forward(inputs)
