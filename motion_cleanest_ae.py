"""CN-XXL + quat-head (91.08 winner) + AE for correspondence emergence.

Pipeline:
  random_input (B, T, N, 4)
    -> per-frame PointNet encoder -> latent (B, T, latent_dim)
    -> per-frame MLP decoder -> canonical (B, T, K, 3)
    -> append normalised time channel -> (B, T, K, 4)
    -> CN-XXL spatial pipeline (no random permutation; canonical order preserved)
    -> main logits

In parallel: per-frame inertia quaternion from the ORIGINAL input xyz
(same signal as the 91.08 quat-head run) -> quat_head MLP -> aux logits
-> ensembled into output via learnable quat_head_scale.

AE training losses (summed into self.aux_loss for the trainer):
  - chamfer(canonical_t, input_t): reconstruction
  - temporal smoothness ||canonical[t+1] - canonical[t]||^2: forces output
    index k to track the same anatomical point across frames (correspondence)

Built on top of MotionCleanestLinXLQuatHead so we keep the +1.45 pp lift
from the quat-head and ONLY add the new AE mechanism.
"""
import torch
import torch.nn as nn

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from models.motion_cleanest_quat import _inertia_quat


def _chamfer_distance(p1, p2):
    """Batched bidirectional chamfer. p1: (M, A, 3), p2: (M, B, 3) -> scalar."""
    d = torch.cdist(p1, p2)
    d_ab = d.min(dim=2)[0].mean(dim=1)
    d_ba = d.min(dim=1)[0].mean(dim=1)
    return (d_ab + d_ba).mean()


class FrameEncoder(nn.Module):
    """(B, T, N, 3) -> (B, T, latent_dim) via per-frame PointNet."""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, latent_dim, 1, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz):
        B, T, N, _ = xyz.shape
        x = xyz.reshape(B * T, N, 3).transpose(1, 2)
        f = self.mlp(x)
        z = f.max(dim=-1)[0]
        return z.reshape(B, T, -1)


class FrameDecoder(nn.Module):
    """(B, T, latent_dim) -> (B, T, K, 3) ordered canonical points."""
    def __init__(self, latent_dim=128, K=128, hidden=256):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, K * 3),
        )

    def forward(self, latent):
        B, T, _ = latent.shape
        out = self.mlp(latent)
        return out.reshape(B, T, self.K, 3)


class MotionCleanestLinXLAE(MotionCleanestLinXLQuatHead):
    def __init__(self, *args,
                 ae_latent_dim=128, ae_K=128, ae_decoder_hidden=256,
                 chamfer_weight=0.05, temporal_weight=0.5,
                 ae_warmstart=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = FrameEncoder(latent_dim=ae_latent_dim)
        self.decoder = FrameDecoder(
            latent_dim=ae_latent_dim, K=ae_K, hidden=ae_decoder_hidden
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
        # Override parent: canonical points carry index correspondence; we must
        # NOT random-permute. But we DO honour the pts_size ramp by slicing
        # the first pts_size indices sequentially, preserving correspondence
        # on those indices across batches (always the same anatomical
        # regions, just fewer of them at early epochs).
        points = inputs.permute(0, 3, 1, 2).contiguous()
        point_count = points.shape[3]
        keep = min(self.pts_size, point_count)
        return points[:, :self.coord_channels, :, :keep]

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        B, T, N, _ = inputs.shape
        xyz_orig = inputs[..., :3]                              # (B, T, N, 3)

        # AE: random no-correspondence -> latent -> canonical ordered K-pts
        latent = self.encoder(xyz_orig)
        canonical = self.decoder(latent)                        # (B, T, K, 3)

        # Append normalised per-frame time channel for classifier input
        t_idx = torch.arange(T, device=xyz_orig.device, dtype=xyz_orig.dtype)
        t_idx = (t_idx - t_idx.mean()) / t_idx.std().clamp(min=1e-6)
        t_channel = t_idx.view(1, T, 1, 1).expand(B, T, self.ae_K, 1)
        classifier_input = torch.cat([canonical, t_channel], dim=-1)

        # Quat-head aux from ORIGINAL input distribution (the 91.08 mechanism).
        coords_orig = xyz_orig.permute(0, 3, 1, 2).contiguous()  # (B, 3, T, N)
        quat = _inertia_quat(coords_orig)                         # (B*T, 4)
        quat_traj = quat.reshape(B, T * 4)
        aux_logits = self.quat_head(quat_traj)                    # (B, num_classes)

        # Main path on canonical points
        coords = self._sample_points(classifier_input)
        fea3 = self._encode_sampled_points(coords)
        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        main_logits = self.classify_features(output.flatten(1))

        # AE training losses
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

        # Output = main + decorative quat-head aux (the 91.08 ensemble)
        return main_logits + self.quat_head_scale * aux_logits
