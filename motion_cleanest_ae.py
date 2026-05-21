"""AE v2 for self-supervised correspondence emergence on hand point clouds.

Architecture:
  FrameEncoder: per-point MLP, no max-pool bottleneck. (B, T, N, 3) -> (B, T, N, F)
  FrameDecoder: K shared learnable queries, 2-block cross-attention, FFN.
                Each query outputs xyz. (B, T, N, F) -> (B, T, K, 3)

Training losses (used by pretrain_ae.py):
  chamfer (bidirectional)
  temporal smoothness across frames per query index
  density-weighted input->canonical coverage
  hinge-squared repulsion between canonical NN pairs

Correspondence emerges because:
  1. The K query embeddings are shared across frames + samples, so the k-th
     output always asks the same question of the input.
  2. Temporal smoothness penalises drift of the k-th canonical position
     across adjacent frames, forcing index k to track the same surface.
  3. Repulsion + density-weighted coverage prevent collapse and keep the
     K outputs distributed over the input surface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_diagonal(d, fill=1e9):
    M, A, B = d.shape
    eye = torch.eye(A, B, device=d.device, dtype=torch.bool)
    return d.masked_fill(eye, fill)


def chamfer_two_sided(p1, p2):
    """p1: (M, K, 3), p2: (M, N, 3) -> ch_a, ch_b."""
    d = torch.cdist(p1, p2)
    ch_a = d.min(dim=2)[0].mean()
    ch_b = d.min(dim=1)[0].mean()
    return ch_a, ch_b


def density_weighted_coverage(p_can, p_inp, knn=5, eps=1e-6):
    """Weight each input point's nearest-canonical distance by local sparsity."""
    M, N, _ = p_inp.shape
    with torch.no_grad():
        d_in = torch.cdist(p_inp, p_inp)
        d_in = _mask_diagonal(d_in)
        knn_dist = d_in.topk(min(knn, N - 1), dim=-1, largest=False)[0]
        local_radius = knn_dist.mean(-1) + eps
        weight = local_radius / local_radius.sum(-1, keepdim=True)
    d_in_can = torch.cdist(p_inp, p_can)
    nearest = d_in_can.min(-1)[0]
    return (nearest * weight).sum(-1).mean()


def repulsion_loss(p_can, radius):
    """Hinge-squared penalty for canonical NN pairs closer than radius."""
    d = torch.cdist(p_can, p_can)
    d = _mask_diagonal(d)
    nn_dist = d.min(-1)[0]
    return torch.relu(radius - nn_dist).pow(2).mean()


class FrameEncoder(nn.Module):
    """Per-point encoder, no max-pool. (B, T, N, 3) -> (B, T, N, F)."""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, feature_dim, 1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
        )

    def forward(self, xyz):
        B, T, N, _ = xyz.shape
        x = xyz.reshape(B * T, N, 3).transpose(1, 2)
        f = self.mlp(x).transpose(1, 2)
        return f.reshape(B, T, N, -1)


class CrossAttnBlock(nn.Module):
    def __init__(self, feature_dim, heads, ffn_mult):
        super().__init__()
        assert feature_dim % heads == 0
        self.heads = heads
        self.head_dim = feature_dim // heads
        self.norm_q = nn.LayerNorm(feature_dim)
        self.norm_kv = nn.LayerNorm(feature_dim)
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm_ffn = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(feature_dim * ffn_mult, feature_dim),
        )

    def forward(self, q, kv):
        M, K, F_ = q.shape
        N = kv.shape[1]
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        Q = self.q_proj(qn).reshape(M, K, self.heads, self.head_dim).transpose(1, 2)
        Kp = self.k_proj(kvn).reshape(M, N, self.heads, self.head_dim).transpose(1, 2)
        Vp = self.v_proj(kvn).reshape(M, N, self.heads, self.head_dim).transpose(1, 2)
        attn = (Q @ Kp.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ Vp).transpose(1, 2).reshape(M, K, F_)
        q = q + self.out_proj(out)
        q = q + self.ffn(self.norm_ffn(q))
        return q


class FrameDecoder(nn.Module):
    """Stacked cross-attention. K shared queries -> K xyz."""
    def __init__(self, feature_dim=128, K=1024, query_dim=64, heads=4,
                 num_attn_blocks=2, ffn_mult=4):
        super().__init__()
        self.K = K
        self.queries = nn.Parameter(torch.randn(K, query_dim) * 0.02)
        self.q_proj = nn.Linear(query_dim, feature_dim)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(feature_dim, heads, ffn_mult)
            for _ in range(num_attn_blocks)
        ])
        self.to_xyz = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 3),
        )

    def forward(self, point_feats):
        B, T, N, F_ = point_feats.shape
        BT = B * T
        kv = point_feats.reshape(BT, N, F_)
        q = self.q_proj(self.queries).unsqueeze(0).expand(BT, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, kv)
        return self.to_xyz(q).reshape(B, T, self.K, 3)
