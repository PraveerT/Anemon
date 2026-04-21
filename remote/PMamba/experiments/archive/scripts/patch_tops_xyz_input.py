"""XYZ + tops field together as 7-channel input through full backbone.

Computes unit direction from frame centroid, concatenates with xyz+time,
resulting in 7 per-point channels. Overrides the first EdgeConv conv to
accept 14 input channels (graph features double the input via
_get_graph_feature).

Tests whether making the centroid-radial direction explicit alongside
xyz beats the xyz-only ceiling of 75.93%.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class TopsXYZInputMotion(BearingQCCFeatureMotion):
    """XYZ + centroid-radial direction (7 channels) through full architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace first conv: input now 14 channels (2 * 7) instead of 8 (2 * 4).
        hidden1 = self.edgeconv[0].out_channels
        self.edgeconv = nn.Sequential(
            nn.Conv2d(14, hidden1, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def extract_features(self, inputs, aux_input=None):
        # Intercept the sample and augment its points tensor with tops direction.
        if isinstance(inputs, dict):
            pts = inputs["points"]
        else:
            pts = inputs
        if pts.dim() == 4:
            B, F_, P, C = pts.shape
            xyz_full = pts[..., :3].float()
        elif pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            P = self.pts_size
            xyz_full = pts[..., :3].float().view(B, F_, P, 3)
        else:
            return super().extract_features(inputs, aux_input=aux_input)

        centroid = xyz_full.mean(dim=2, keepdim=True)
        rel = xyz_full - centroid
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        tops = (rel / rel_norm).detach()                          # (B, F, P, 3)

        if pts.dim() == 4:
            tops_flat = tops                                       # already (B, F, P, 3)
            aug = torch.cat([pts[..., :4], tops_flat], dim=-1)     # (B, F, P, 7)
        else:
            aug = torch.cat([pts[..., :4], tops.view(B, N, 3)], dim=-1)

        if isinstance(inputs, dict):
            new_inputs = dict(inputs)
            new_inputs["points"] = aug
        else:
            new_inputs = aug

        # Override the reshape-to-4 hardcoding in the parent by monkey-patching
        # the channel count this class expects. The parent\'s extract_features
        # reshapes with `-1, 4`. To preserve behavior, we need to carry the 7
        # channels through. Easiest: duplicate just the sampling+reshape logic
        # from parent, but with 7 channels. Since that\'s many lines, we instead
        # take a thin approach: call parent, then inside its body the
        # _encode_to_pre_merge will fail because sampled still has 7 channels
        # but reshape uses 4. So we need to override differently.
        #
        # Clean approach: override the reshape step by hooking extract_features.
        # Here we duplicate just the necessary bits.
        return self._extract_features_7ch(new_inputs, aux_input=aux_input)

    def _extract_features_7ch(self, inputs, aux_input=None):
        # Replicate parent\'s extract_features but with 7 channels throughout
        # the sampled -> encoded path. Aux loss/modulation logic kept minimal
        # (we set qcc_weight=0 in config so aux is effectively off).
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
            points_all = points[..., :7]
            sampled, corr_matched = self._correspondence_guided_sample(
                points_all, aux_unpacked)
        elif has_corr and self.decouple_sampling:
            points_4d = points[..., :4]
            sampled_base = self._sample_points(points_4d)
            # Gather matching tops by rebuilding from xyz centroid (since points already augmented)
            # Simpler: just use guided path
            points_all = points[..., :7]
            sampled, corr_matched = self._correspondence_guided_sample(
                points_all, aux_unpacked)
        else:
            points_all = points[..., :7]
            sampled = self._sample_points(points_all)
            corr_matched = None

        batch_size, num_frames, pts_per_frame, _ = sampled.shape

        # Rigidity: computed from xyz only (first 3 channels of sampled)
        rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
            sampled[..., :4], num_frames, knn_k=self.bearing_knn_k,
            corr_matched=corr_matched)

        # Encoder sees 7 channels
        point_features = sampled.reshape(batch_size, -1, 7).transpose(1, 2).contiguous()
        encoded = self._encode_to_pre_merge(point_features)

        # Modulate with rigidity (unchanged)
        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            encoded = encoded * (1.0 + modulation)

        # Skip aux loss path entirely; just produce classifier features.
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        # Replicate the stacked-quaternion collapse + attention readout.
        encoded_t = encoded.transpose(1, 2).contiguous()
        encoded_bf = encoded_t.view(batch_size, num_frames, pts_per_frame, -1)
        quat_weighted, _ = self.stacked_merge(encoded_bf)
        collapsed = self.collapse(quat_weighted)
        pooled_max = collapsed.max(dim=1).values
        pooled_attn, _ = self.attn_readout(collapsed)
        return torch.cat((pooled_max, pooled_attn), dim=1)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added TopsXYZInputMotion")
