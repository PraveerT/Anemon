"""Add RigidityAttentionBearingQCCFeatureMotion.

Option 2 from RIGIDITY_OPTIONS.md. Keeps the 4-channel input unchanged (xyz +
time) so the backbone's representation is not perturbed. Uses the per-point
Kabsch-residual rigidity as a soft gate during readout pooling: articulated
points (high residual) get emphasized, rigid/background points suppressed.

Learnable thresholds (rig_tau, rig_alpha) let the model calibrate where the
cutoff is. Gate is applied as a multiplicative mask on the encoded features
before both max and attention-weighted pooling.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

marker = "class RigidityInputBearingQCCFeatureMotion(BearingQCCFeatureMotion):"
assert marker in src, "anchor missing — run patch_rigidity_input.py first"

new_class = '''

class RigidityAttentionBearingQCCFeatureMotion(BearingQCCFeatureMotion):
    """Weight per-point features by sigmoid-gated per-point Kabsch residual.

    Input channels unchanged (4: x, y, z, time). Rigidity is computed on-the-
    fly from correspondence-aligned xyz and used purely as a scalar gate:

        w_i = sigmoid((r_i - tau) * alpha)

    The encoded features are multiplied by w_i before both readout pools (max
    and attention-weighted). tau and alpha are learnable; init picks a median-
    like tau and moderate sharpness so roughly half the points are gated-in at
    init.
    """

    def __init__(self, *args, rig_gate_init_tau: float = 0.05,
                 rig_gate_init_alpha: float = 8.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rig_tau = nn.Parameter(torch.tensor(float(rig_gate_init_tau)))
        self.rig_alpha = nn.Parameter(torch.tensor(float(rig_gate_init_alpha)))

    def extract_features(self, inputs, aux_input=None):
        if isinstance(inputs, dict):
            points = inputs["points"]
            aux_unpacked = inputs
        elif aux_input is not None:
            points = inputs
            aux_unpacked = aux_input
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

        sampled = sampled[..., :4]
        batch_size, num_frames, pts_per_frame, _ = sampled.shape

        rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
            sampled, num_frames, knn_k=self.bearing_knn_k,
            corr_matched=corr_matched,
        )

        # Per-point Kabsch residual -> soft gate
        with torch.no_grad():
            rig_res = _kabsch_rigidity_magnitudes(sampled[..., :3])   # (B, F, P)
        gate = torch.sigmoid((rig_res - self.rig_tau) * self.rig_alpha)   # (B, F, P)

        point_features = sampled.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()
        encoded = self._encode_to_pre_merge(point_features)

        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            encoded = encoded * (1.0 + modulation)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        encoded = self.merge_proj(self.merge_quaternions(encoded))        # (B, C, N)

        # Apply rigidity gate before pooling.
        gate_flat = gate.reshape(batch_size, -1).unsqueeze(1)             # (B, 1, N)
        encoded_gated = encoded * gate_flat

        pooled_max = encoded_gated.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded_gated * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)
'''

src = src.replace(marker, new_class.strip() + "\n\n\n" + marker, 1)
PATH.write_text(src, encoding="utf-8")
print("inserted RigidityAttentionBearingQCCFeatureMotion")
