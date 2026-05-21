"""AE v2: per-point encoder + stacked cross-attention decoder.

Decoder details:
  - K learnable query embeddings shared across frames and samples
  - 2 cross-attention blocks (pre-norm + FFN, ffn_mult=4)
  - Each query outputs xyz only (no opacity gating in v2)

Losses for self-supervised pretrain:
  chamfer (bidirectional)
  temporal smoothness
  density-weighted input->canonical coverage
  repulsion (hinge-squared on canonical NN pairs closer than radius)

Side-channel use (MotionCleanestLinXLQuatHeadCanonicalAux below):
  raw input -> main path (MotionCleanestLinXLQuatHead, the 91.08 setup)
  raw input -> AE encoder/decoder -> canonical -> tiny aux MLP -> aux logits
  output = main + quat_head_scale * quat_aux + canonical_aux_scale * canonical_aux
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from models.motion_cleanest_quat import _inertia_quat


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


# ---------------------------------------------------------------------------
# Side-channel model: AE produces canonical as an aux head on top of the
# 91.08 quat-head path.
# ---------------------------------------------------------------------------

class MotionCleanestLinXLQuatHeadCanonicalAux(MotionCleanestLinXLQuatHead):
    """Main path: standard 91.08 quat-head on raw input.
    Aux path: AE on the same raw input -> canonical -> trajectory MLP -> 25
    logits, ensembled into output via learnable scale.

    The AE losses (chamfer + temporal + density + repulsion) are added to
    self.aux_loss during training so the canonical stays anchored.
    """
    def __init__(self, *args,
                 ae_feature_dim=128, ae_K=1024, ae_query_dim=64, ae_heads=4,
                 ae_num_attn_blocks=2, ae_ffn_mult=4,
                 canonical_aux_hidden=128, canonical_aux_scale=0.3,
                 chamfer_weight=0.05, temporal_weight=0.5,
                 repulsion_weight=0.05, repulsion_radius=0.05,
                 density_weight=0.05, density_knn=5,
                 ae_warmstart=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = FrameEncoder(feature_dim=ae_feature_dim)
        self.decoder = FrameDecoder(
            feature_dim=ae_feature_dim, K=ae_K,
            query_dim=ae_query_dim, heads=ae_heads,
            num_attn_blocks=ae_num_attn_blocks, ffn_mult=ae_ffn_mult,
        )
        self.ae_K = ae_K

        # Aux head reads canonical -> (B, T, 6) per-frame summary
        # (mean xyz + std xyz over the K canonical points) -> 192-d traj
        # -> MLP -> num_classes
        self.canonical_aux = nn.Sequential(
            nn.Linear(self.framesize * 6, canonical_aux_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(canonical_aux_hidden, self.num_classes),
        )
        self.canonical_aux_scale = nn.Parameter(torch.tensor(float(canonical_aux_scale)))

        self.chamfer_weight = chamfer_weight
        self.temporal_weight = temporal_weight
        self.repulsion_weight = repulsion_weight
        self.repulsion_radius = repulsion_radius
        self.density_weight = density_weight
        self.density_knn = density_knn
        self.aux_loss = None

        if ae_warmstart:
            ckpt = torch.load(ae_warmstart, map_location='cpu')
            self.encoder.load_state_dict(ckpt['encoder'])
            self.decoder.load_state_dict(ckpt['decoder'])
            print(f'[AE] loaded warm-start from {ae_warmstart}')

    def forward(self, inputs):
        # Main path: vanilla quat-head on raw input.
        main_out = super().forward(inputs)

        # Aux path: AE -> canonical -> summary stats -> aux logits.
        if isinstance(inputs, dict):
            raw = inputs['points']
        else:
            raw = inputs
        B, T, N, _ = raw.shape
        xyz_orig = raw[..., :3]

        point_feats = self.encoder(xyz_orig)
        canonical = self.decoder(point_feats)                         # (B, T, K, 3)

        # Per-frame summary: mean + std over K canonical points.
        c_mean = canonical.mean(dim=2)                                 # (B, T, 3)
        c_std = canonical.std(dim=2)                                   # (B, T, 3)
        summary = torch.cat([c_mean, c_std], dim=-1).reshape(B, T * 6)
        canonical_aux_logits = self.canonical_aux(summary)

        # AE training losses keep canonical anchored.
        if self.training:
            BT = B * T
            can_flat = canonical.reshape(BT, self.ae_K, 3)
            inp_flat = xyz_orig.reshape(BT, N, 3)
            ch_a, ch_b = chamfer_two_sided(can_flat, inp_flat)
            chamfer = (ch_a + ch_b) / 2
            density = density_weighted_coverage(can_flat, inp_flat, knn=self.density_knn)
            rep = repulsion_loss(can_flat, radius=self.repulsion_radius)
            temporal = ((canonical[:, 1:] - canonical[:, :-1]) ** 2).mean()
            self.aux_loss = (
                self.chamfer_weight * chamfer +
                self.temporal_weight * temporal +
                self.density_weight * density +
                self.repulsion_weight * rep
            )
        else:
            self.aux_loss = None

        return main_out + self.canonical_aux_scale * canonical_aux_logits


# Back-compat alias (older yamls reference MotionCleanestLinXLAE).
MotionCleanestLinXLAE = MotionCleanestLinXLQuatHeadCanonicalAux
