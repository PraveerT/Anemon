"""CN-XXL (90.46/91.08 quat-head) with the global max-pool (pool5) replaced by a
NetVLAD learnable cluster-residual pool (Arandjelovic et al., as used in
PointNetVLAD). ONLY the pooling changes: stage5 features (B,1024,T,N) are VLAD-
pooled over the T*N tokens instead of max-pooled. Everything else = 91.08 backbone
(quat aux head kept identical). Tests whether a soft-assignment distribution pool
beats the max on the saturated depth stream.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from models.motion_cleanest_quat import _inertia_quat


class NetVLAD(nn.Module):
    # lean: 1x1 reduce (in_dim->dim) keeps params ~baseline so the test isolates
    # the soft-assignment pooling, not added capacity (capacity hurts here).
    def __init__(self, in_dim=1024, dim=128, K=16, out_dim=1024):
        super().__init__()
        self.K = K
        self.dim = dim
        self.reduce = nn.Conv2d(in_dim, dim, kernel_size=1, bias=True)
        self.assign = nn.Conv2d(dim, K, kernel_size=1, bias=True)   # soft-assign logits
        self.centroids = nn.Parameter(torch.randn(K, dim) * 0.01)
        self.proj = nn.Linear(K * dim, out_dim)

    def forward(self, x):                       # x: (B, in_dim, T, N)
        x = self.reduce(x)                                  # (B,dim,T,N)
        B, C, T, N = x.shape
        M = T * N
        xf = x.reshape(B, C, M)                              # (B,C,M)
        a = self.assign(x).reshape(B, self.K, M)
        a = torch.softmax(a, dim=1)                          # (B,K,M) soft assign
        ax = torch.einsum('bkm,bcm->bkc', a, xf)            # (B,K,C)
        a_sum = a.sum(dim=-1)                                # (B,K)
        vlad = ax - a_sum.unsqueeze(-1) * self.centroids.unsqueeze(0)   # residual
        vlad = F.normalize(vlad, dim=2)                      # intra-normalize
        vlad = vlad.reshape(B, self.K * C)
        vlad = F.normalize(vlad, dim=1)                      # global L2
        return self.proj(vlad)                              # (B, out_dim)


class MotionCleanestLinXLNetVLAD(MotionCleanestLinXLQuatHead):
    def __init__(self, *args, vlad_clusters=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlad = NetVLAD(in_dim=1024, dim=128, K=int(vlad_clusters), out_dim=1024)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)            # (B,4,T,N)
        B = coords.shape[0]
        T = coords.shape[2]
        # quat aux head (identical to 91.08)
        quat = _inertia_quat(coords[:, :3]).reshape(B, T * 4)
        if self.quat_zero_input:
            quat = torch.zeros_like(quat)
        aux_logits = self.quat_head(quat)
        # main path: NetVLAD pool instead of pool5 max-pool
        fea3 = self._encode_sampled_points(coords)
        s5 = self.stage5(fea3)                           # (B,1024,T,N)
        g = self.vlad(s5)                                # (B,1024)
        g = self.global_bn(g.unsqueeze(-1).unsqueeze(-1)).flatten(1)
        main_logits = self.classify_features(g)
        if self.quat_no_aux:
            return main_logits
        return main_logits + self.quat_head_scale * aux_logits
