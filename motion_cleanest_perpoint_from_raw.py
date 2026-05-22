"""Per-point quaternion aux ON TOP of the 91.08 quat-head winner.

Main path: raw NVGesture input -> MotionCleanestLinXLQuatHead (the 91.08
setup, untouched — includes decorative inertia quat-head ensemble).

New aux path: same raw input -> FROZEN AE (encoder + decoder) -> canonical
(B, T, K=1024, 3) -> per-point Hamilton-product quaternion chain Q (B, K, 4)
-> small per-K projection MLP + mean+max pool -> aux logits, ensembled into
output via learnable scale.

The AE is loaded from the v2 pretrain ckpt and frozen (requires_grad=False),
so no gradient flows through it during classifier training. This is the
cleanest test of "does per-point quaternion give real lift over 91.08."

Floor: 91.08 (main + inertia aux are unchanged). Aux can only add.
"""
import torch
import torch.nn as nn

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from models.motion_cleanest_ae import FrameEncoder, FrameDecoder
from models.motion_cleanest_perpoint_quat import per_point_quaternion_chain


class MotionCleanestLinXLQuatHeadPerPointFromRaw(MotionCleanestLinXLQuatHead):
    def __init__(self, *args,
                 ae_warmstart=None,
                 ae_feature_dim=128, ae_K=1024, ae_query_dim=64, ae_heads=4,
                 ae_num_attn_blocks=2, ae_ffn_mult=4,
                 perpoint_proj_dim=32, perpoint_quat_scale=0.3,
                 freeze_main=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_main = freeze_main

        # Frozen AE that produces canonical from raw input.
        self.ae_encoder = FrameEncoder(feature_dim=ae_feature_dim)
        self.ae_decoder = FrameDecoder(
            feature_dim=ae_feature_dim, K=ae_K,
            query_dim=ae_query_dim, heads=ae_heads,
            num_attn_blocks=ae_num_attn_blocks, ffn_mult=ae_ffn_mult,
        )
        if ae_warmstart:
            ckpt = torch.load(ae_warmstart, map_location='cpu')
            self.ae_encoder.load_state_dict(ckpt['encoder'])
            self.ae_decoder.load_state_dict(ckpt['decoder'])
            print(f'[AE] frozen warm-start from {ae_warmstart} '
                  f'(chamfer={ckpt.get("best_score", "n/a")})')
        for p in self.ae_encoder.parameters():
            p.requires_grad = False
        for p in self.ae_decoder.parameters():
            p.requires_grad = False
        self.ae_encoder.eval()
        self.ae_decoder.eval()
        self.ae_K = ae_K

        # Per-point quat aux head (operates on Hamilton-chain output Q (B,K,4)).
        self.perpoint_proj = nn.Sequential(
            nn.Linear(4, perpoint_proj_dim),
            nn.GELU(),
            nn.Linear(perpoint_proj_dim, perpoint_proj_dim),
        )
        self.perpoint_head = nn.Sequential(
            nn.LayerNorm(2 * perpoint_proj_dim),
            nn.Linear(2 * perpoint_proj_dim, self.num_classes),
        )
        self.perpoint_scale = nn.Parameter(torch.tensor(float(perpoint_quat_scale)))

        if self.freeze_main:
            # Freeze everything except the per-point aux head + scale.
            trainable = {'perpoint_proj', 'perpoint_head', 'perpoint_scale'}
            for name, p in self.named_parameters():
                top = name.split('.', 1)[0]
                if top not in trainable:
                    p.requires_grad = False
            n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.parameters())
            print(f'[freeze_main] {n_trainable}/{n_total} params trainable '
                  f'({n_trainable / n_total:.1%})')

    def train(self, mode=True):
        # Keep AE in eval mode permanently (BN running stats fixed).
        super().train(mode)
        self.ae_encoder.eval()
        self.ae_decoder.eval()
        # If main is frozen, force the parent's modules into eval so BN
        # running stats stay locked at the 91.08 ckpt values.
        if self.freeze_main:
            for name, module in self.named_children():
                if name in ('perpoint_proj', 'perpoint_head'):
                    continue
                module.eval()
        return self

    def forward(self, inputs):
        # Main path: 91.08 quat-head on raw input (unchanged from parent).
        main_out = super().forward(inputs)

        if isinstance(inputs, dict):
            raw = inputs['points']
        else:
            raw = inputs
        xyz_orig = raw[..., :3]                                     # (B, T, N, 3)

        # Frozen AE forward (no grad).
        with torch.no_grad():
            point_feats = self.ae_encoder(xyz_orig)
            canonical = self.ae_decoder(point_feats)                  # (B, T, K, 3)
        Q = per_point_quaternion_chain(canonical)                      # (B, K, 4)

        proj = self.perpoint_proj(Q)                                   # (B, K, P)
        mean_pool = proj.mean(dim=1)                                   # (B, P)
        max_pool = proj.max(dim=1)[0]                                  # (B, P)
        feat = torch.cat([mean_pool, max_pool], dim=-1)                # (B, 2P)
        perpoint_logits = self.perpoint_head(feat)                     # (B, num_classes)

        return main_out + self.perpoint_scale * perpoint_logits
