"""VelocityPolarBearingQCCFeatureMotion: velocity + polar + time as 8-ch input.

Replaces xyz (3 ch) with an 8-channel hybrid representation:
  [vx, vy, vz, tops_x, tops_y, tops_z, |p-centroid|, time]

Velocity vx = sampled_xyz[t+1, i] - sampled_xyz[t, i]  (forward diff)
Uses correspondence-guided sampling so that index i in frames t and t+1
refers to the same physical point.
Last frame gets backward-diff: v_i(F-1) = p_{F-1} - p_{F-2}.

Rigidity still computed from real xyz+time (unchanged).

First EdgeConv conv enlarged to 16→hidden1 (8-ch input doubled by graph).
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class VelocityPolarBearingQCCFeatureMotion(BearingQCCFeatureMotion):
    """BearingQCCFeatureMotion with xyz replaced by velocity + polar."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden1 = self.edgeconv[0].out_channels
        # 8-ch input -> graph doubles to 16
        self.edgeconv = nn.Sequential(
            nn.Conv2d(16, hidden1, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def extract_features(self, inputs, aux_input=None):
        if isinstance(inputs, dict):
            points = inputs["points"]
            aux_unpacked = inputs
        else:
            points = inputs
            aux_unpacked = None

        has_corr = (
            aux_unpacked is not None
            and "orig_flat_idx" in aux_unpacked
            and "corr_full_target_idx" in aux_unpacked
            and "corr_full_weight" in aux_unpacked
        )

        if has_corr and not self.decouple_sampling:
            sampled, corr_matched = self._correspondence_guided_sample(
                points[..., :4], aux_unpacked
            )
        else:
            sampled = self._sample_points(points[..., :4])
            corr_matched = None

        sampled = sampled[..., :4]                                   # (B, F, P, 4)
        batch_size, num_frames, pts_per_frame, _ = sampled.shape

        # Rigidity from real xyz+time.
        rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
            sampled, num_frames, knn_k=self.bearing_knn_k,
            corr_matched=corr_matched,
        )

        xyz = sampled[..., :3]
        time_ch = sampled[..., 3:4]

        # Polar from per-frame centroid.
        centroid = xyz.mean(dim=2, keepdim=True)
        rel = xyz - centroid
        magnitude = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)    # (B, F, P, 1)
        direction = (rel / magnitude).detach()                         # (B, F, P, 3)

        # Velocity (forward diff; last frame = backward diff).
        # xyz: (B, F, P, 3); velocity at t = xyz[t+1] - xyz[t]
        vel = torch.zeros_like(xyz)
        vel[:, :-1] = xyz[:, 1:] - xyz[:, :-1]
        vel[:, -1] = xyz[:, -1] - xyz[:, -2]

        # 8-ch input
        sampled_8 = torch.cat([vel, direction, magnitude, time_ch], dim=-1)  # (B, F, P, 8)
        point_features = sampled_8.reshape(batch_size, -1, 8).transpose(1, 2).contiguous()
        encoded = self._encode_to_pre_merge(point_features)

        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            encoded = encoded * (1.0 + modulation)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        encoded = self.merge_proj(self.merge_quaternions(encoded))
        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added VelocityPolarBearingQCCFeatureMotion")
