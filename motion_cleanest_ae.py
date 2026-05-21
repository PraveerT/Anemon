"""CN-XXL + quat-head (91.08 winner) + AE for correspondence emergence.

AE design (no information bottleneck):
  encoder: per-point MLP (no max-pool), preserves per-point information
    input  (B, T, N, 3)  ->  features (B, T, N, F)
  decoder: K learnable query embeddings + 1-layer cross-attention to
           per-point features, then MLP -> xyz. The k-th query is shared
           across all frames, so it asks for "the same" content each
           frame -> output index k carries correspondence by construction
           (anatomical region selected by the learned k-th query).

Training pipeline:
  1. Pretrain encoder + decoder alone (chamfer + temporal smoothness).
  2. Bake offline canonical dataset (apply frozen AE to all NVGesture
     samples; save to disk).
  3. Train CN-XXL+quat-head on the baked dataset (correspondence is now
     ground truth in the data; classifier is decoupled from the AE).

This file additionally provides MotionCleanestLinXLAE, an end-to-end
wrapper kept for completeness (AE in-the-loop with the classifier).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from models.motion_cleanest_quat import _inertia_quat


def _chamfer_distance(p1, p2):
    """Batched bidirectional chamfer. p1: (M, A, 3), p2: (M, B, 3) -> scalar."""
    d = torch.cdist(p1, p2)
    d_ab = d.min(dim=2)[0].mean(dim=1)
    d_ba = d.min(dim=1)[0].mean(dim=1)
    return (d_ab + d_ba).mean()


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
        x = xyz.reshape(B * T, N, 3).transpose(1, 2)             # (BT, 3, N)
        f = self.mlp(x).transpose(1, 2)                            # (BT, N, F)
        return f.reshape(B, T, N, -1)


class FrameDecoder(nn.Module):
    """K learnable queries cross-attend to per-point features -> K xyz.

    Each query embedding e_k is shared across frames and samples; the k-th
    output index always asks "the same question", so over training it
    learns to localise the same anatomical region -> correspondence.
    """
    def __init__(self, feature_dim=128, K=512, query_dim=64, heads=4, ffn_mult=2):
        super().__init__()
        assert feature_dim % heads == 0
        self.K = K
        self.heads = heads
        self.head_dim = feature_dim // heads
        self.queries = nn.Parameter(torch.randn(K, query_dim) * 0.02)
        self.q_proj = nn.Linear(query_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(feature_dim * ffn_mult, feature_dim),
        )
        self.to_xyz = nn.Linear(feature_dim, 3)

    def forward(self, point_feats):
        B, T, N, F = point_feats.shape
        BT = B * T
        x = point_feats.reshape(BT, N, F)
        q = self.q_proj(self.queries).unsqueeze(0).expand(BT, -1, -1)   # (BT, K, F)

        Q = q.reshape(BT, self.K, self.heads, self.head_dim).transpose(1, 2)
        Kp = self.k_proj(x).reshape(BT, N, self.heads, self.head_dim).transpose(1, 2)
        Vp = self.v_proj(x).reshape(BT, N, self.heads, self.head_dim).transpose(1, 2)

        attn = (Q @ Kp.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ Vp).transpose(1, 2).reshape(BT, self.K, F)
        out = self.out_proj(out)
        out = self.norm1(out + q)
        out = self.norm2(out + self.ffn(out))
        xyz = self.to_xyz(out).reshape(B, T, self.K, 3)
        return xyz


class MotionCleanestLinXLAE(MotionCleanestLinXLQuatHead):
    def __init__(self, *args,
                 ae_feature_dim=128, ae_K=512, ae_query_dim=64, ae_heads=4,
                 chamfer_weight=0.05, temporal_weight=0.5,
                 ae_warmstart=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = FrameEncoder(feature_dim=ae_feature_dim)
        self.decoder = FrameDecoder(
            feature_dim=ae_feature_dim, K=ae_K,
            query_dim=ae_query_dim, heads=ae_heads,
        )
        self.ae_K = ae_K
        self.chamfer_weight = chamfer_weight
        self.temporal_weight = temporal_weight
        self.aux_loss = None

        if ae_warmstart:
            ckpt = torch.load(ae_warmstart, map_location='cpu')
            self.encoder.load_state_dict(ckpt['encoder'])
            self.decoder.load_state_dict(ckpt['decoder'])
            print(f'[AE] loaded warm-start from {ae_warmstart} '
                  f'(best_score={ckpt.get("best_score", "n/a")}, '
                  f'best_epoch={ckpt.get("best_epoch", "n/a")})')

    def _sample_points(self, inputs):
        # First pts_size canonical sequentially: preserves correspondence
        # on those indices, honours the parent's dynamic pts_size ramp.
        points = inputs.permute(0, 3, 1, 2).contiguous()
        point_count = points.shape[3]
        keep = min(self.pts_size, point_count)
        return points[:, :self.coord_channels, :, :keep]

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        B, T, N, _ = inputs.shape
        xyz_orig = inputs[..., :3]

        point_feats = self.encoder(xyz_orig)                        # (B, T, N, F)
        canonical = self.decoder(point_feats)                        # (B, T, K, 3)

        t_idx = torch.arange(T, device=xyz_orig.device, dtype=xyz_orig.dtype)
        t_idx = (t_idx - t_idx.mean()) / t_idx.std().clamp(min=1e-6)
        t_channel = t_idx.view(1, T, 1, 1).expand(B, T, self.ae_K, 1)
        classifier_input = torch.cat([canonical, t_channel], dim=-1)

        coords_orig = xyz_orig.permute(0, 3, 1, 2).contiguous()
        quat = _inertia_quat(coords_orig)
        quat_traj = quat.reshape(B, T * 4)
        aux_logits = self.quat_head(quat_traj)

        coords = self._sample_points(classifier_input)
        fea3 = self._encode_sampled_points(coords)
        out = self.stage5(fea3)
        out = self.pool5(out)
        out = self.global_bn(out)
        main_logits = self.classify_features(out.flatten(1))

        if self.training:
            BT = B * T
            chamfer = _chamfer_distance(
                canonical.reshape(BT, self.ae_K, 3),
                xyz_orig.reshape(BT, N, 3),
            )
            temporal = ((canonical[:, 1:] - canonical[:, :-1]) ** 2).mean()
            self.aux_loss = (
                self.chamfer_weight * chamfer +
                self.temporal_weight * temporal
            )
        else:
            self.aux_loss = None

        return main_logits + self.quat_head_scale * aux_logits
