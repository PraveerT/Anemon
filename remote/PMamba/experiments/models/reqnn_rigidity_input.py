"""Quaternion encoder with geometric motion features as input channels.

Variants:

RigidityInputMotion (token: ``ri``)
    Single-scale rigidity as 5th channel. Input = (x,y,z,intensity,rigidity).

RigidityMultiInputMotion (token: ``rim``)
    Multi-scale rigidity + displacement magnitude as 4 extra channels.
    Input = (x,y,z,intensity, rig_k5, rig_k15, rig_k40, disp_mag) = 8 ch.

DisplacementVectorMotion (token: ``dv``)
    Per-point mean displacement vector as 3 extra channels.
    Input = (x,y,z,intensity, dx, dy, dz) = 7 ch.
    Directly encodes motion direction — the most discriminative feature for
    gesture recognition because different gestures ARE different motion
    directions. This is what PMamba exploits internally.

QfwdMotion (token: ``qf``)
    Per-point mean forward rotation quaternion as 4 extra channels.
    Input = (x,y,z,intensity, qfwd_w, qfwd_x, qfwd_y, qfwd_z) = 8 ch.
    Encodes HOW each point rotates, not just whether neighbors agree.

QfwdDisplacementMotion (token: ``qfdv``)
    Both displacement vector + q_fwd as 7 extra channels.
    Input = (x,y,z,intensity, dx, dy, dz, qfwd_w, qfwd_x, qfwd_y, qfwd_z) = 11 ch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reqnn_motion import (
    SimpleLinearMotion,
    QuaternionPointLinear,
    _knn_indices,
    _get_graph_feature,
    _compute_bearing_qcc_aligned,
    quaternion_weighted_rms_merge,
    _reshape_quaternion_groups,
)


# =========================================================================
# Shared encoder body (parameterized by input_channels)
# =========================================================================

class _RigidityEncoderBase(SimpleLinearMotion):
    """Branch-2 'kept design' encoder, parameterized by input channel count.

    Subclasses override ``_compute_extra_channels`` to produce the geometric
    features that get concatenated to (x, y, z, intensity) before the encoder.
    """

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
        bearing_knn_k=10,
        input_channels=5,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        hidden1, hidden2 = hidden_dims
        if hidden2 % 4 != 0:
            raise ValueError("hidden_dims[1] must be divisible by 4.")

        self.edgeconv_k = edgeconv_k
        self.merge_eps = merge_eps
        self.bearing_knn_k = bearing_knn_k
        self.feature_dim = hidden2 * 2

        # edgeconv input = 2 * input_channels (relative diff + center)
        self.edgeconv = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden1, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.quaternion_encoder = QuaternionPointLinear(hidden1, hidden2)
        self.encoder_norm = nn.BatchNorm1d(hidden2)
        self.encoder_activation = nn.GELU()

        self.quaternion_refine = QuaternionPointLinear(hidden2, hidden2)
        self.refine_norm = nn.BatchNorm1d(hidden2)
        self.refine_activation = nn.GELU()

        self.merge_component_logits = nn.Parameter(torch.zeros(4))
        self.merge_proj = nn.Sequential(
            nn.Conv1d(hidden2 // 2, hidden2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
        )

        self.readout_attention = nn.Conv1d(hidden2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.readout_attention.weight)
        nn.init.zeros_(self.readout_attention.bias)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def _merge_quaternions(self, encoded):
        grouped = _reshape_quaternion_groups(encoded)
        merged_rms = quaternion_weighted_rms_merge(
            encoded,
            component_weights=self.merge_component_logits,
            eps=self.merge_eps,
        )
        real_part = grouped[:, :, 0, :]
        return torch.cat((merged_rms, real_part), dim=1)

    def _compute_extra_channels(self, sampled, num_frames, pts_per_frame):
        """Override in subclasses. Returns (B, C_extra, F*P)."""
        raise NotImplementedError

    def extract_features(self, inputs, aux_input=None):
        sampled = self._sample_points(inputs)
        batch_size, num_frames, pts_per_frame, _ = sampled.shape

        # Base 4 channels
        point_features = sampled.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        # Extra geometric channels from subclass
        extra = self._compute_extra_channels(sampled, num_frames, pts_per_frame)
        point_features = torch.cat([point_features, extra], dim=1)

        # Encoder
        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)

        refined = self.quaternion_refine(encoded.transpose(1, 2).contiguous())
        refined = self.refine_norm(refined.transpose(1, 2).contiguous())
        refined = self.refine_activation(refined)
        encoded = encoded + refined

        encoded = self.merge_proj(self._merge_quaternions(encoded))

        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


# =========================================================================
# ri — single-scale rigidity (5 channels)
# =========================================================================

class RigidityInputMotion(_RigidityEncoderBase):
    """Input = (x, y, z, intensity, rigidity). Token: ``ri``."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128),
                 dropout=0.1, edgeconv_k=20, merge_eps=1e-6, bearing_knn_k=10):
        super().__init__(
            num_classes=num_classes, pts_size=pts_size, hidden_dims=hidden_dims,
            dropout=dropout, edgeconv_k=edgeconv_k, merge_eps=merge_eps,
            bearing_knn_k=bearing_knn_k, input_channels=5,
        )

    def _compute_extra_channels(self, sampled, num_frames, pts_per_frame):
        rigidity, _ = _compute_bearing_qcc_aligned(
            sampled, num_frames, knn_k=self.bearing_knn_k,
        )
        return rigidity  # (B, 1, F*P)


# =========================================================================
# rim — multi-scale rigidity + displacement magnitude (8 channels)
# =========================================================================

class RigidityMultiInputMotion(_RigidityEncoderBase):
    """Input = (x, y, z, intensity, rig_k5, rig_k15, rig_k40, disp_mag).

    Token: ``rim``.

    4 extra channels beyond the base (x,y,z,intensity):

    rig_k5   fine-scale rigidity   — captures finger-joint deformation
    rig_k15  medium-scale rigidity — captures finger-segment motion
    rig_k40  coarse-scale rigidity — captures whole-hand-region motion
    disp_mag per-point displacement magnitude between consecutive frames,
             averaged across frame transitions — captures motion speed

    Together these give the encoder a richer geometric prior than
    single-scalar rigidity: different gestures produce different
    spatial patterns of multi-scale rigidity AND different speed profiles.
    """

    RIGIDITY_SCALES = (5, 15, 40)

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128),
                 dropout=0.1, edgeconv_k=20, merge_eps=1e-6, bearing_knn_k=10):
        super().__init__(
            num_classes=num_classes, pts_size=pts_size, hidden_dims=hidden_dims,
            dropout=dropout, edgeconv_k=edgeconv_k, merge_eps=merge_eps,
            bearing_knn_k=bearing_knn_k,
            input_channels=4 + len(self.RIGIDITY_SCALES) + 1,  # 8
        )

    def _compute_extra_channels(self, sampled, num_frames, pts_per_frame):
        batch_size = sampled.shape[0]
        device = sampled.device

        # Multi-scale rigidity
        rigidities = []
        for k in self.RIGIDITY_SCALES:
            rig, _ = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=k,
            )  # (B, 1, F*P)
            rigidities.append(rig)

        # Per-point displacement magnitude between consecutive frames
        xyz = sampled[..., :3]  # (B, F, P, 3)
        displacements = xyz[:, 1:] - xyz[:, :-1]  # (B, F-1, P, 3)
        disp_mag = displacements.norm(dim=-1)  # (B, F-1, P)

        # Average displacement across frame transitions → (B, P)
        mean_disp = disp_mag.mean(dim=1)

        # Normalize to [0, 1] range per batch for stability
        disp_max = mean_disp.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        mean_disp = mean_disp / disp_max

        # Expand to all frames: (B, P) → (B, 1, F*P)
        disp_expanded = mean_disp.unsqueeze(1).expand(
            -1, num_frames, -1,
        ).reshape(batch_size, 1, -1)

        # Concat: 3 rigidity channels + 1 displacement channel = 4 extra
        return torch.cat(rigidities + [disp_expanded], dim=1)  # (B, 4, F*P)


# =========================================================================
# Helper: per-point mean displacement vector (3D)
# =========================================================================

def _compute_displacement_vectors(sampled, num_frames):
    """Mean per-point displacement vector across frame transitions.

    Returns (B, 3, F*P): dx, dy, dz per point, expanded to all frames.
    """
    batch_size = sampled.shape[0]
    pts_per_frame = sampled.shape[2]
    xyz = sampled[..., :3]  # (B, F, P, 3)
    displacements = xyz[:, 1:] - xyz[:, :-1]  # (B, F-1, P, 3)
    mean_disp = displacements.mean(dim=1)  # (B, P, 3)

    # Expand to all frames: (B, P, 3) → (B, F, P, 3) → (B, 3, F*P)
    expanded = mean_disp.unsqueeze(1).expand(
        -1, num_frames, -1, -1,
    ).reshape(batch_size, -1, 3).transpose(1, 2).contiguous()
    return expanded


# =========================================================================
# Helper: per-point mean forward rotation quaternion (4D)
# =========================================================================

def _compute_mean_qfwd(sampled, num_frames, knn_k=10):
    """Mean per-point forward rotation quaternion across frame transitions.

    Computes bearing quaternion per point per frame, then q_fwd = q_{t+1} *
    conj(q_t).  Averages q_fwd across F-1 transitions and normalizes.

    Returns (B, 4, F*P): qfwd_w, qfwd_x, qfwd_y, qfwd_z per point,
    expanded to all frames.
    """
    batch_size = sampled.shape[0]
    pts_per_frame = sampled.shape[2]
    device = sampled.device

    xyz = sampled[..., :3]
    bbox_min = xyz.min(dim=2).values
    bbox_max = xyz.max(dim=2).values
    centroids = (bbox_min + bbox_max) / 2
    directions = xyz - centroids.unsqueeze(2)
    dir_norm = directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    directions = directions / dir_norm

    dot = directions[..., 1].clamp(-1 + 1e-7, 1 - 1e-7)
    half_angle = torch.acos(dot) / 2
    axis = torch.stack([directions[..., 2], torch.zeros_like(directions[..., 0]),
                        -directions[..., 0]], dim=-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w = torch.cos(half_angle)
    sin_ha = torch.sin(half_angle)
    bearing_q = torch.stack([w, axis[..., 0] * sin_ha, axis[..., 1] * sin_ha,
                             axis[..., 2] * sin_ha], dim=-1)

    # q_fwd = q_{t+1} * conj(q_t)
    q_curr = bearing_q[:, :-1]
    q_next = bearing_q[:, 1:]
    conj = q_curr * torch.tensor([1, -1, -1, -1], device=device, dtype=q_curr.dtype)

    aw, ax, ay, az = q_next[..., 0], q_next[..., 1], q_next[..., 2], q_next[..., 3]
    bw, bx, by, bz = conj[..., 0], conj[..., 1], conj[..., 2], conj[..., 3]
    q_fwd = torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)  # (B, F-1, P, 4)

    # Mean across transitions, normalize to unit quaternion
    mean_qfwd = q_fwd.mean(dim=1)  # (B, P, 4)
    mean_qfwd = torch.nn.functional.normalize(mean_qfwd, dim=-1)

    # Expand to all frames: (B, P, 4) → (B, 4, F*P)
    expanded = mean_qfwd.unsqueeze(1).expand(
        -1, num_frames, -1, -1,
    ).reshape(batch_size, -1, 4).transpose(1, 2).contiguous()
    return expanded


# =========================================================================
# dv — displacement vector (7 channels)
# =========================================================================

class DisplacementVectorMotion(_RigidityEncoderBase):
    """Input = (x, y, z, intensity, dx, dy, dz). Token: ``dv``.

    Per-point mean displacement vector across frame transitions.
    Directly encodes motion direction — the most discriminative feature
    for gesture recognition because different gestures ARE different
    motion directions.
    """

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128),
                 dropout=0.1, edgeconv_k=20, merge_eps=1e-6, bearing_knn_k=10):
        super().__init__(
            num_classes=num_classes, pts_size=pts_size, hidden_dims=hidden_dims,
            dropout=dropout, edgeconv_k=edgeconv_k, merge_eps=merge_eps,
            bearing_knn_k=bearing_knn_k, input_channels=7,
        )

    def _compute_extra_channels(self, sampled, num_frames, pts_per_frame):
        return _compute_displacement_vectors(sampled, num_frames)  # (B, 3, F*P)


# =========================================================================
# qf — forward rotation quaternion (8 channels)
# =========================================================================

class QfwdMotion(_RigidityEncoderBase):
    """Input = (x, y, z, intensity, qfwd_w, qfwd_x, qfwd_y, qfwd_z). Token: ``qf``.

    Per-point mean forward rotation quaternion. Encodes HOW each point
    rotates between frames, not just whether neighbors agree (rigidity).
    """

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128),
                 dropout=0.1, edgeconv_k=20, merge_eps=1e-6, bearing_knn_k=10):
        super().__init__(
            num_classes=num_classes, pts_size=pts_size, hidden_dims=hidden_dims,
            dropout=dropout, edgeconv_k=edgeconv_k, merge_eps=merge_eps,
            bearing_knn_k=bearing_knn_k, input_channels=8,
        )

    def _compute_extra_channels(self, sampled, num_frames, pts_per_frame):
        return _compute_mean_qfwd(sampled, num_frames)  # (B, 4, F*P)


# =========================================================================
# qfdv — both q_fwd + displacement vector (11 channels)
# =========================================================================

class QfwdDisplacementMotion(_RigidityEncoderBase):
    """Input = (x,y,z,intensity, dx,dy,dz, qfwd_w,qfwd_x,qfwd_y,qfwd_z). Token: ``qfdv``.

    Full per-point motion descriptor: translational velocity (displacement)
    + rotational velocity (q_fwd). 11 input channels.
    """

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128),
                 dropout=0.1, edgeconv_k=20, merge_eps=1e-6, bearing_knn_k=10):
        super().__init__(
            num_classes=num_classes, pts_size=pts_size, hidden_dims=hidden_dims,
            dropout=dropout, edgeconv_k=edgeconv_k, merge_eps=merge_eps,
            bearing_knn_k=bearing_knn_k, input_channels=11,
        )

    def _compute_extra_channels(self, sampled, num_frames, pts_per_frame):
        dv = _compute_displacement_vectors(sampled, num_frames)  # (B, 3, F*P)
        qf = _compute_mean_qfwd(sampled, num_frames)  # (B, 4, F*P)
        return torch.cat([dv, qf], dim=1)  # (B, 7, F*P)
