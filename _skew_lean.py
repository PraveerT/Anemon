"""PGCNetSkewLean -- lean PGCNetPruned with a LOAD-BEARING Skew-TCC descriptor.

Unlike the 91.91 raw head (a decoupled logit-ensemble aux: out = main + scale*aux,
scale -> 0 at inference => decorative), here the antisymmetric lagged
cross-covariance descriptor is CONCATENATED into the dual-pool feature that feeds
the classifier:

    feat = [ g(1024 global-max) | t(256 temporal-head) | skew_desc(132) ]
    logits = dual_fc(feat)

So the classifier weights the descriptor directly and gradients flow into the
Wu/Wv projectors from the MAIN cross-entropy loss -- it cannot be scaled/zeroed
away. Descriptor = antisymmetric part of the rank-r lagged cross-Gram of the raw
per-frame uvd 3x3 covariance (a fixed transform of the input points). desc_len =
r*(r-1)/2 * len(lags) = 12*11/2 * 2 = 132.
"""
import torch
import torch.nn as nn

from models.pgcnet_pruned import PGCNetPruned


class PGCNetSkewLean(PGCNetPruned):
    def __init__(self, *args, r=12, lags=(1, 2), mode='skew', **kwargs):
        super().__init__(*args, **kwargs)
        self.r = int(r)
        self.lags = tuple(lags)
        self.mode = mode
        self.Wu = nn.Linear(9, self.r, bias=False)        # 9 = flattened 3x3 cov
        self.Wv = nn.Linear(9, self.r, bias=False)
        if mode == 'random':                              # frozen-projector control
            for p in list(self.Wu.parameters()) + list(self.Wv.parameters()):
                p.requires_grad = False
        idx = torch.tril_indices(self.r, self.r, offset=-1)
        self.register_buffer('ti', idx)
        desc_len = self.r * (self.r - 1) // 2 * len(self.lags)
        self.desc_len = desc_len
        self.desc_bn = nn.BatchNorm1d(desc_len)
        # rebuild the classifier to ingest the descriptor (load-bearing)
        in_dim = 1024 + self.temporal_head.out_dim + desc_len
        self.dual_fc = nn.Linear(in_dim, self.num_classes)

    def _per_frame_cov(self, coords):                     # coords (B, C>=3, T, N)
        xyz = coords[:, :3]                                # (B,3,T,N) image-plane uvd
        B, _, T, N = xyz.shape
        x = xyz.permute(0, 2, 1, 3)                        # (B,T,3,N)
        x = x - x.mean(dim=-1, keepdim=True)               # center per frame
        cov = torch.matmul(x, x.transpose(-1, -2)) / max(N - 1, 1)   # (B,T,3,3)
        return cov.reshape(B, T, 9)

    def _desc(self, z):                                    # z: (B, T, 9)
        U, V = self.Wu(z), self.Wv(z)                      # (B, T, r)
        T = z.shape[1]
        outs = []
        half = self.r * (self.r - 1) // 2
        for d in self.lags:
            if T - d <= 0:
                outs.append(z.new_zeros(z.shape[0], half))
                continue
            u, v = U[:, :T - d], V[:, d:]
            C = torch.einsum('bti,btj->bij', u, v) / max(1, T - d)
            if self.mode in ('skew', 'random'):
                M = (C - C.transpose(1, 2)) * 0.5
            else:
                M = (C + C.transpose(1, 2)) * 0.5
            outs.append(M[:, self.ti[0], self.ti[1]])
        return torch.cat(outs, dim=1)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)
        desc = self.desc_bn(self._desc(self._per_frame_cov(coords)))   # (B, 132)
        fea3 = self._encode_sampled_points(coords)
        s5 = self.stage5(fea3)
        g = self.global_bn(self.pool5(s5)).flatten(1)
        t = self.temporal_head(s5.max(dim=3).values)
        feat = torch.cat([g, t, desc], dim=1)
        return self.dual_fc(self.dual_drop(feat))

    def extract_features(self, inputs):
        return self.forward(inputs)
