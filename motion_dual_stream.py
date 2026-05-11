"""Dual-stream model: depth (main, N2 backbone) + lattice arrow field (side, SeQuMamba).

v4: Two parallel classifiers (depth + lattice), additive fusion at logit level,
    auxiliary CE loss on each branch via framework's temporal/spatial pattern.

Why v4 vs v3:
  v3 used additive zero-init "correction head". Pathology: zero-init local
  optimum is "always output zero correction", and weight decay collapsed the
  side branch's lift weights ~5 orders of magnitude over 20 epochs. Side was
  effectively dead. Here, side has its own CE loss -> forced to learn.

Architecture:
  main_logits = main.classify_features(main.extract_features(depth))     # (B, 25)
  side_logits = side_classifier(side(lattice_q))                          # (B, 25)
  final       = main_logits + side_logits                                  # (B, 25)

Framework integration: exposes `temporal_logits = main_logits`, `spatial_logits = side_logits`,
and `aux_weight = 0.3`. main.py adds 0.3 * (CE(main) + CE(side)) to the primary CE loss.

Total loss = CE(main + side, y) + 0.3 * CE(main, y) + 0.3 * CE(side, y)

Inference: returns final = main + side.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.motion_qumamba import SeQuMambaBlock


class LatticeSeQuMambaEncoder(nn.Module):
    """Lattice arrow-field SeQuMamba encoder. SO(3)-invariant magnitude readout."""
    def __init__(self, n_lattice=216, hidden=192, n_layers=4, dropout=0.2):
        super().__init__()
        assert hidden % 4 == 0
        self.n_quat = hidden // 4
        self.lift = nn.Linear(n_lattice, self.n_quat, bias=False)
        self.blocks = nn.ModuleList([SeQuMambaBlock(self.n_quat) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.n_quat) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(self.n_quat)

    def _gnorm(self, x_q, norm_layer):
        mags = x_q.norm(dim=-1, keepdim=True) + 1e-9
        new_mags = F.silu(norm_layer(mags.squeeze(-1))).unsqueeze(-1)
        return x_q / mags * new_mags

    def forward(self, x):
        x = self.lift(x.transpose(-1, -2)).transpose(-1, -2)  # (B, T, n_quat, 4)
        for blk, norm in zip(self.blocks, self.norms):
            residual = x
            x_n = self._gnorm(x, norm)
            out = blk(x_n)
            x = self.dropout(out) + residual
        x = self._gnorm(x, self.final_norm)
        h = x.mean(dim=1)            # (B, n_quat, 4)
        mags = h.norm(dim=-1)        # (B, n_quat)
        return mags


class MotionDualStream(nn.Module):
    def __init__(self, num_classes=25, pts_size=96, knn=(32, 24, 48, 24), topk=8,
                 multi_scale_num_scales=5,
                 lattice_n=216, side_hidden=192, side_layers=4,
                 side_classifier_hidden=192, side_dropout=0.2,
                 aux_weight=1.0,
                 lattice_init_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.pts_size = pts_size
        self.lattice_n = lattice_n
        self.aux_weight = aux_weight   # framework reads this for branch CE weighting

        # Main branch: full Motion (N2 backbone + native classifier)
        self.main = Motion(num_classes=num_classes, pts_size=pts_size,
                           knn=list(knn), topk=topk,
                           multi_scale_num_scales=multi_scale_num_scales)

        # Side branch: lattice SeQuMamba
        self.side = LatticeSeQuMambaEncoder(n_lattice=lattice_n,
                                            hidden=side_hidden,
                                            n_layers=side_layers)
        side_mag_dim = side_hidden // 4

        # Side classifier: matches standalone LatticeSeQuMamba head shape
        # (Linear -> GELU -> Dropout -> Linear) for warm-start compatibility
        self.side_classifier = nn.Sequential(
            nn.Linear(side_mag_dim, side_classifier_hidden),
            nn.GELU(),
            nn.Dropout(side_dropout),
            nn.Linear(side_classifier_hidden, num_classes),
        )

        # Optional warm-start of side branch from standalone lattice training
        if lattice_init_weights is not None:
            self._load_lattice_warmstart(lattice_init_weights)

    def _load_lattice_warmstart(self, ckpt_path):
        import os, torch
        if not os.path.isfile(ckpt_path):
            print(f"[MotionDualStream] WARN: lattice_init_weights not found at {ckpt_path}, skipping warm-start")
            return
        ck = torch.load(ckpt_path, map_location='cpu')
        sd = ck.get('model_state_dict', ck)
        # Standalone keys: lift.weight, blocks.X.*, norms.X.*, final_norm.*, head.X.*
        # Map to:           side.lift.weight, side.blocks.X.*, side.norms.X.*, side.final_norm.*, side_classifier.X.*
        side_sd = {}
        cls_sd = {}
        for k, v in sd.items():
            if k.startswith('head.'):
                cls_sd[k[len('head.'):]] = v
            else:
                side_sd[k] = v
        m_side, u_side = self.side.load_state_dict(side_sd, strict=False)
        m_cls, u_cls = self.side_classifier.load_state_dict(cls_sd, strict=False)
        print(f"[MotionDualStream] warm-start side branch from {ckpt_path}")
        print(f"  side: missing={len(m_side)} unexpected={len(u_side)}")
        print(f"  side_classifier: missing={len(m_cls)} unexpected={len(u_cls)}")

    def _split_input(self, x):
        if x.dim() == 4 and x.shape[-1] == 8:
            B, T, P_total, C = x.shape
        elif x.dim() == 4 and x.shape[1] == 8:
            x = x.permute(0, 2, 3, 1).contiguous()
            B, T, P_total, C = x.shape
        else:
            raise ValueError(f"Unexpected input shape {tuple(x.shape)}")
        P_d = P_total - self.lattice_n
        depth_pts = x[:, :, :P_d, :4].contiguous()
        lattice_q = x[:, :, P_d:, 4:8].contiguous()
        return depth_pts, lattice_q

    def forward(self, x):
        depth_pts, lattice_q = self._split_input(x)
        # Main branch: standard Motion forward
        main_feat = self.main.extract_features(depth_pts)
        main_logits = self.main.classify_features(main_feat)     # (B, 25)
        # Side branch: SO(3)-invariant magnitude features -> classifier
        side_mag = self.side(lattice_q)
        side_logits = self.side_classifier(side_mag)              # (B, 25)
        # Expose branch logits for framework aux loss (main.py adds aux_weight * (CE(main) + CE(side)))
        self.temporal_logits = main_logits
        self.spatial_logits = side_logits
        # Fused prediction (additive ensemble)
        return main_logits + side_logits
