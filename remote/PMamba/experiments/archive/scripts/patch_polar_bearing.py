"""PolarBearingQCCFeatureMotion: xyz replaced by (tops_x, tops_y, tops_z, magnitude).

Replaces the 3 xyz channels of BearingQCCFeatureMotion input with a 4-channel
polar representation: unit direction from frame centroid (3) + magnitude (1).
Keeps time as a 5th channel. Rigidity still computed from real xyz.

Adds class PolarBearingQCCFeatureMotion to models/reqnn_motion.py.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class PolarBearingQCCFeatureMotion(BearingQCCFeatureMotion):
    """BearingQCCFeatureMotion with xyz replaced by polar (tops + magnitude).

    Input channels (5 total): [dir_x, dir_y, dir_z, |p-centroid|, time]
    - dir = (p - frame_centroid) / |p - frame_centroid|   (unit direction)
    - |p - centroid| = radial magnitude
    Rigidity is still computed from the real sampled xyz (first 3 channels of
    the pre-swap tensor), so the bearing-QCC geometric signal is preserved.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rebuild first EdgeConv conv for 5-channel input (doubled by graph = 10).
        hidden1 = self.edgeconv[0].out_channels
        self.edgeconv = nn.Sequential(
            nn.Conv2d(10, hidden1, kernel_size=1, bias=False),
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

        sampled = sampled[..., :4]                                    # (B, F, P, 4) = xyz+time
        batch_size, num_frames, pts_per_frame, _ = sampled.shape

        # Rigidity from real xyz+time (unchanged).
        rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
            sampled, num_frames, knn_k=self.bearing_knn_k,
            corr_matched=corr_matched,
        )

        # Polar reparameterization of xyz.
        xyz = sampled[..., :3]
        time_ch = sampled[..., 3:4]
        centroid = xyz.mean(dim=2, keepdim=True)
        rel = xyz - centroid
        magnitude = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)    # (B, F, P, 1)
        direction = (rel / magnitude).detach()                         # (B, F, P, 3)
        sampled_5 = torch.cat([direction, magnitude, time_ch], dim=-1)  # (B, F, P, 5)

        point_features = sampled_5.reshape(batch_size, -1, 5).transpose(1, 2).contiguous()
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
print("added PolarBearingQCCFeatureMotion")
