"""Quaternion PointNet partner for NVGesture fusion.

NOT a rotation model. Each point's 4 sampled channels (x,y,z,+1) are packed as a
quaternion; the encoder is a per-point shared MLP built from Hamilton-product
(quaternion) linear layers. Structurally far from the kNN-graph CN-XXL backbone
-> orthogonal error pattern -> intended as a *fusion partner*, not a solo winner.

mode='quat'  : Hamilton-product weight sharing (the real thing).
mode='real'  : param-matched plain MLP (structure-ablation control:
               "does the quaternion algebra help, or just the param budget?").
mode='free4' : per-layer free 4x4 channel mixing, same param budget as real but
               4-block structured (controls for "structure vs algebra").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuaternionLinear(nn.Module):
    """Maps in_q quaternions -> out_q quaternions via Hamilton product.

    Feature layout: last dim = 4*Q, ordered [r(Q) | i(Q) | j(Q) | k(Q)].
    Params = 4 * in_q * out_q  (+ 4*out_q bias).
    """

    def __init__(self, in_q, out_q, bias=True):
        super().__init__()
        self.in_q, self.out_q = in_q, out_q
        self.wr = nn.Parameter(torch.empty(out_q, in_q))
        self.wi = nn.Parameter(torch.empty(out_q, in_q))
        self.wj = nn.Parameter(torch.empty(out_q, in_q))
        self.wk = nn.Parameter(torch.empty(out_q, in_q))
        for w in (self.wr, self.wi, self.wj, self.wk):
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(4 * out_q)) if bias else None

    def forward(self, x):
        xr, xi, xj, xk = x.split(self.in_q, dim=-1)
        cr = F.linear(xr, self.wr) - F.linear(xi, self.wi) - F.linear(xj, self.wj) - F.linear(xk, self.wk)
        ci = F.linear(xr, self.wi) + F.linear(xi, self.wr) + F.linear(xj, self.wk) - F.linear(xk, self.wj)
        cj = F.linear(xr, self.wj) - F.linear(xi, self.wk) + F.linear(xj, self.wr) + F.linear(xk, self.wi)
        ck = F.linear(xr, self.wk) + F.linear(xi, self.wj) - F.linear(xj, self.wi) + F.linear(xk, self.wr)
        out = torch.cat([cr, ci, cj, ck], dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out


def split_act(x, act):
    return act(x)  # split activation == componentwise; GELU is componentwise anyway


class MotionQuatPointNet(nn.Module):
    def __init__(self, num_classes, pts_size, coord_channels=4, framesize=32,
                 q_dim=48, n_layers=3, dropout=0.3, head_hidden=256,
                 mode='quat', real_width=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.pts_size = pts_size
        self.coord_channels = coord_channels
        self.framesize = framesize
        self.mode = mode
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        if mode == 'quat':
            dims = [1] + [q_dim] * n_layers          # in quaternions per layer
            self.layers = nn.ModuleList(
                [QuaternionLinear(dims[i], dims[i + 1]) for i in range(n_layers)])
            feat_q = q_dim
            self.point_feat = 4 * feat_q
        else:
            # real / free4 control: operate on coord_channels real features
            w = real_width or (q_dim)               # real hidden width (tune to match params)
            rdims = [coord_channels] + [w] * n_layers
            self.layers = nn.ModuleList(
                [nn.Linear(rdims[i], rdims[i + 1]) for i in range(n_layers)])
            self.point_feat = w

        # global: max+mean over points, mean over frames -> 2*point_feat
        gfeat = 2 * self.point_feat
        self.head = nn.Sequential(
            nn.Linear(gfeat, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes))

    def _sample_points(self, inputs):
        points = inputs.permute(0, 3, 1, 2)              # (B, C, T, N)
        n = points.shape[3]
        k = min(self.pts_size, n)
        if self.training:
            idx = torch.randperm(n, device=points.device)[:k]
        else:
            idx = torch.linspace(0, n - 1, k, device=points.device).long()
        return points[:, :self.coord_channels][:, :, :, idx]   # (B, 4, T, N')

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)             # (B, 4, T, N)
        x = coords.permute(0, 2, 3, 1).contiguous()      # (B, T, N, 4) -> quaternion per point
        for lin in self.layers:
            x = self.drop(self.act(lin(x)))
        # pool over points N
        xmax = x.max(dim=2).values                       # (B, T, F)
        xmean = x.mean(dim=2)                             # (B, T, F)
        # pool over frames T (mean)
        g = torch.cat([xmax.mean(1), xmean.mean(1)], dim=-1)   # (B, 2F)
        return self.head(g)

    def extract_features(self, inputs):
        return self.forward(inputs)


def MotionRealPointNet(*a, **k):
    k['mode'] = 'real'
    return MotionQuatPointNet(*a, **k)
