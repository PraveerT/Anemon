"""Add RigidityInputBearingQCCFeatureMotion: BearingQCCFeatureMotion with a 5th
per-point input channel = per-point rigidity residual magnitude against the
whole-cloud Kabsch rotation.

For each correspondence-sampled frame pair (t, t+1 cyclic) over B batches, P
points per frame, we compute the best rigid rotation q_best via batched SVD
Kabsch on the 3D positions, apply it to (p_t - c_t), and the residual
norm ‖p_{t+1} - (R p_{t_centered} + c_{t+1})‖ becomes a per-point scalar.
That scalar is concatenated as the 5th channel for the first EdgeConv layer.

The rest of the BearingQCCFeatureMotion pipeline runs unchanged (Bearing-QCC
rigidity modulation stays in place).
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

marker = "class PolarBearingQCCFeatureMotion(BearingQCCFeatureMotion):"
assert marker in src, "anchor for insertion missing"

new_class = '''

def _kabsch_rigidity_magnitudes(xyz: torch.Tensor) -> torch.Tensor:
    """Per-point Kabsch-residual magnitudes over cyclic (t, t+1) frame pairs.

    xyz: (B, F, P, 3). Returns (B, F, P) non-negative scalar per point per frame
    equal to ‖x_{t+1} − (R_t · (x_t − c_t) + c_{t+1})‖, where R_t is the best
    rigid rotation fitting P_t to P_{t+1} via Kabsch-SVD.
    """
    B, F, P, _ = xyz.shape
    P_src = xyz
    P_tgt = torch.roll(xyz, shifts=-1, dims=1)
    cP = P_src.mean(dim=-2, keepdim=True)
    cQ = P_tgt.mean(dim=-2, keepdim=True)
    Pc = P_src - cP
    Qc = P_tgt - cQ
    H = Pc.transpose(-2, -1) @ Qc                                   # (B, F, 3, 3)
    # Small ridge for numerical stability during SVD.
    H = H + 1e-6 * torch.eye(3, device=xyz.device, dtype=xyz.dtype)
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-2, -1)
    d = torch.linalg.det(V @ U.transpose(-2, -1))                   # (B, F)
    D = torch.eye(3, device=xyz.device, dtype=xyz.dtype).expand(B, F, 3, 3).clone()
    D[..., 2, 2] = d
    R = V @ D @ U.transpose(-2, -1)                                  # (B, F, 3, 3)
    pred = (R @ Pc.transpose(-2, -1)).transpose(-2, -1)              # (B, F, P, 3)
    resid = Qc - pred                                                 # (B, F, P, 3)
    return resid.norm(dim=-1)                                        # (B, F, P)


class RigidityInputBearingQCCFeatureMotion(BearingQCCFeatureMotion):
    """BearingQCCFeatureMotion with extra per-point rigidity channel.

    Input channels (5 total): [x, y, z, time, ‖rigidity_residual‖].
    The residual is computed in-graph from correspondence-aligned xyz via
    whole-cloud Kabsch; not backprop-through for simplicity (detached).
    """

    def __init__(self, *args, rigidity_norm_scale: float = 8.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Rebuild first EdgeConv for 5-channel input -> 10 after graph doubling.
        hidden1 = self.edgeconv[0].out_channels
        self.edgeconv = nn.Sequential(
            nn.Conv2d(10, hidden1, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        # Scale to map typical rigidity magnitudes to ~O(1).
        self.rigidity_norm_scale = rigidity_norm_scale

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

        sampled = sampled[..., :4]                                   # (B, F, P, 4)
        batch_size, num_frames, pts_per_frame, _ = sampled.shape

        # Existing Bearing-QCC rigidity (used downstream for modulation).
        rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
            sampled, num_frames, knn_k=self.bearing_knn_k,
            corr_matched=corr_matched,
        )

        # NEW: per-point Kabsch-residual rigidity -> 5th input channel.
        with torch.no_grad():
            rig_residual = _kabsch_rigidity_magnitudes(sampled[..., :3])  # (B, F, P)
            rig_feat = (rig_residual * self.rigidity_norm_scale).unsqueeze(-1)  # (B, F, P, 1)

        sampled_5 = torch.cat([sampled, rig_feat], dim=-1)            # (B, F, P, 5)
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

# Insert BEFORE PolarBearingQCCFeatureMotion (keeps all existing subclasses
# intact after our addition).
src = src.replace(marker, new_class.strip() + "\n\n\n" + marker, 1)
PATH.write_text(src, encoding="utf-8")
print("inserted RigidityInputBearingQCCFeatureMotion + _kabsch_rigidity_magnitudes")
