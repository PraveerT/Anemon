"""Fix TopsXYZInputMotion: compute tops AFTER sampling, not before."""
import re
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

# Replace the entire class body
pat = re.compile(r"class TopsXYZInputMotion\(BearingQCCFeatureMotion\):.*?(?=\nclass |\Z)", re.DOTALL)

new_class = '''class TopsXYZInputMotion(BearingQCCFeatureMotion):
    """XYZ + centroid-radial direction (7 channels) through full architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden1 = self.edgeconv[0].out_channels
        self.edgeconv = nn.Sequential(
            nn.Conv2d(14, hidden1, kernel_size=1, bias=False),
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
                points[..., :4], aux_unpacked)
        else:
            sampled = self._sample_points(points[..., :4])
            corr_matched = None

        sampled = sampled[..., :4]
        batch_size, num_frames, pts_per_frame, _ = sampled.shape

        # Compute tops AFTER sampling, on the sampled xyz.
        xyz = sampled[..., :3]
        centroid = xyz.mean(dim=2, keepdim=True)
        rel = xyz - centroid
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        tops = (rel / rel_norm).detach()                          # (B, F, P, 3)
        sampled_7 = torch.cat([sampled, tops], dim=-1)             # (B, F, P, 7)

        # Rigidity still uses xyz+time only.
        rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
            sampled, num_frames, knn_k=self.bearing_knn_k,
            corr_matched=corr_matched)

        point_features = sampled_7.reshape(batch_size, -1, 7).transpose(1, 2).contiguous()
        encoded = self._encode_to_pre_merge(point_features)

        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            encoded = encoded * (1.0 + modulation)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        encoded_t = encoded.transpose(1, 2).contiguous()
        encoded_bf = encoded_t.view(batch_size, num_frames, pts_per_frame, -1)
        quat_weighted, _ = self.stacked_merge(encoded_bf)
        collapsed = self.collapse(quat_weighted)
        pooled_max = collapsed.max(dim=1).values
        pooled_attn, _ = self.attn_readout(collapsed)
        return torch.cat((pooled_max, pooled_attn), dim=1)

'''

src = pat.sub(new_class, src)
PATH.write_text(src, encoding="utf-8")
print("rewrote TopsXYZInputMotion (tops computed after sampling)")
