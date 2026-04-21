import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinearMotion(nn.Module):
    """Reset branch-2 baseline using the same first four channels as branch 1."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1):
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two values.")

        hidden1, hidden2 = hidden_dims
        self.num_classes = num_classes
        self.pts_size = pts_size
        self.feature_dim = hidden2 * 2

        self.encoder = nn.Sequential(
            nn.Linear(4, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def _sample_point_indices(self, point_count, device):
        sample_size = min(self.pts_size, point_count)
        if sample_size == point_count:
            return None

        if self.training:
            return torch.randperm(point_count, device=device)[:sample_size]
        return torch.linspace(0, point_count - 1, sample_size, device=device).long()

    def _sample_points_with_aux(self, inputs, aux_input=None):
        points = inputs[..., :4]
        _, _, point_count, _ = points.shape
        indices = self._sample_point_indices(point_count, points.device)
        if indices is None:
            sampled_points = points
        else:
            sampled_points = points[:, :, indices, :]

        sampled_aux = aux_input
        if aux_input is not None:
            sampled_aux = dict(aux_input)
            orig_flat_idx = sampled_aux.get('orig_flat_idx')
            if orig_flat_idx is not None and indices is not None:
                sampled_aux['orig_flat_idx'] = orig_flat_idx[:, :, indices]

        return sampled_points, sampled_aux

    def _sample_points(self, inputs):
        sampled_points, _ = self._sample_points_with_aux(inputs)
        return sampled_points

    def _unpack_inputs(self, inputs):
        if isinstance(inputs, dict):
            return inputs['points'], inputs
        return inputs, None

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        encoded = self.encoder(points.reshape(batch_size, -1, 4))
        pooled_max = encoded.max(dim=1).values
        pooled_mean = encoded.mean(dim=1)
        return torch.cat((pooled_max, pooled_mean), dim=1)

    def classify_features(self, features):
        return self.classifier(features)

    def forward(self, inputs):
        points, aux_input = self._unpack_inputs(inputs)
        features = self.extract_features(points, aux_input=aux_input)
        return self.classify_features(features)


def _knn_indices(x, k):
    if k <= 0:
        raise ValueError("edgeconv_k must be positive.")
    k = min(k, x.size(-1))
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def _get_graph_feature(x, k, idx=None):
    batch_size, num_dims, num_points = x.shape
    if idx is None:
        idx = _knn_indices(x, k)

    k_eff = idx.size(-1)
    idx_base = torch.arange(batch_size, device=x.device).view(-1, 1, 1) * num_points
    flat_idx = (idx + idx_base).reshape(-1)

    points = x.transpose(2, 1).contiguous()
    feature = points.reshape(batch_size * num_points, num_dims)[flat_idx, :]
    feature = feature.view(batch_size, num_points, k_eff, num_dims)
    center = points.view(batch_size, num_points, 1, num_dims).expand(-1, -1, k_eff, -1)

    return torch.cat((feature - center, center), dim=3).permute(0, 3, 1, 2).contiguous()


class QuaternionPointLinear(nn.Module):
    """Pointwise quaternion linear transform over channel groups of four."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.quat_in = (in_features + 3) // 4
        self.quat_out = (out_features + 3) // 4

        scale = 0.02
        self.weight_r = nn.Parameter(torch.randn(self.quat_out, self.quat_in) * scale)
        self.weight_i = nn.Parameter(torch.randn(self.quat_out, self.quat_in) * scale)
        self.weight_j = nn.Parameter(torch.randn(self.quat_out, self.quat_in) * scale)
        self.weight_k = nn.Parameter(torch.randn(self.quat_out, self.quat_in) * scale)
        self.bias = nn.Parameter(torch.zeros(self.quat_out * 4))

    def forward(self, x):
        batch_size, num_points, channels = x.shape
        if channels % 4 != 0:
            x = F.pad(x, (0, 4 - (channels % 4)))
            channels = x.shape[-1]

        x = x.view(batch_size, num_points, 4, channels // 4)
        x_r, x_i, x_j, x_k = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]

        out_r = torch.matmul(x_r, self.weight_r.t()) - torch.matmul(x_i, self.weight_i.t()) - \
                torch.matmul(x_j, self.weight_j.t()) - torch.matmul(x_k, self.weight_k.t())
        out_i = torch.matmul(x_r, self.weight_i.t()) + torch.matmul(x_i, self.weight_r.t()) + \
                torch.matmul(x_j, self.weight_k.t()) - torch.matmul(x_k, self.weight_j.t())
        out_j = torch.matmul(x_r, self.weight_j.t()) - torch.matmul(x_i, self.weight_k.t()) + \
                torch.matmul(x_j, self.weight_r.t()) + torch.matmul(x_k, self.weight_i.t())
        out_k = torch.matmul(x_r, self.weight_k.t()) + torch.matmul(x_i, self.weight_j.t()) - \
                torch.matmul(x_j, self.weight_i.t()) + torch.matmul(x_k, self.weight_r.t())

        out = torch.stack((out_r, out_i, out_j, out_k), dim=2).reshape(batch_size, num_points, -1)

        if out.shape[-1] > self.out_features:
            out = out[:, :, :self.out_features]
        elif out.shape[-1] < self.out_features:
            out = F.pad(out, (0, self.out_features - out.shape[-1]))

        return out + self.bias[:self.out_features]


def quaternion_merge(x):
    grouped = _reshape_quaternion_groups(x)
    return torch.sum(grouped * grouped, dim=2)


def quaternion_rms_merge(x, eps=1e-6):
    grouped = _reshape_quaternion_groups(x)
    return torch.sqrt(torch.mean(grouped * grouped, dim=2) + eps)


def quaternion_weighted_rms_merge(x, component_weights, eps=1e-6):
    grouped = _reshape_quaternion_groups(x)
    if component_weights.dim() == 1:
        normalized_weights = torch.softmax(component_weights, dim=-1).view(1, 1, 4, 1)
    elif component_weights.dim() == 2:
        normalized_weights = torch.softmax(component_weights, dim=-1).unsqueeze(1).unsqueeze(-1)
    elif component_weights.dim() == 3:
        normalized_weights = torch.softmax(component_weights, dim=1).unsqueeze(1)
    else:
        raise ValueError("component_weights must have 1, 2, or 3 dimensions.")
    return torch.sqrt(torch.sum(grouped * grouped * normalized_weights, dim=2) + eps)


def quaternion_normalize(x, eps=1e-6):
    grouped = _reshape_quaternion_groups(x)
    norms = torch.sqrt(torch.sum(grouped * grouped, dim=2, keepdim=True) + eps)
    return (grouped / norms).view_as(x)


def _reshape_quaternion_groups(x):
    if x.size(1) % 4 != 0:
        raise ValueError("Quaternion merge expects channel count divisible by 4.")
    batch_size, channels, num_points = x.shape
    return x.view(batch_size, channels // 4, 4, num_points)


class EdgeConvLinearMotion(SimpleLinearMotion):
    """Stage-1 additive branch: add a single DGCNN-style local-neighborhood block."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20):
        super().__init__(num_classes=num_classes, pts_size=pts_size, hidden_dims=hidden_dims, dropout=dropout)
        hidden1, hidden2 = hidden_dims
        self.edgeconv_k = edgeconv_k
        self.feature_dim = hidden2 * 2

        self.edgeconv = nn.Sequential(
            nn.Conv2d(8, hidden1, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden1, hidden2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values
        encoded = self.encoder(edge_features)

        pooled_max = encoded.max(dim=-1).values
        pooled_mean = encoded.mean(dim=-1)
        return torch.cat((pooled_max, pooled_mean), dim=1)


class EdgeConvQuaternionMergeMotion(EdgeConvLinearMotion):
    """Stage-1 additive branch: quaternion point mixer followed by quaternion-aware merge before pooling."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20):
        super().__init__(num_classes=num_classes, pts_size=pts_size, hidden_dims=hidden_dims, dropout=dropout, edgeconv_k=edgeconv_k)
        hidden1, hidden2 = hidden_dims
        if hidden2 % 4 != 0:
            raise ValueError("hidden_dims[1] must be divisible by 4 for quaternion merge.")

        self.quaternion_encoder = QuaternionPointLinear(hidden1, hidden2)
        self.encoder_norm = nn.BatchNorm1d(hidden2)
        self.encoder_activation = nn.GELU()
        self.merge_proj = nn.Sequential(
            nn.Conv1d(hidden2 // 4, hidden2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
        )

    def merge_quaternions(self, encoded):
        return quaternion_merge(encoded)

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)
        encoded = self.merge_proj(self.merge_quaternions(encoded))

        pooled_max = encoded.max(dim=-1).values
        pooled_mean = encoded.mean(dim=-1)
        return torch.cat((pooled_max, pooled_mean), dim=1)


class EdgeConvQuaternionRMSMergeMotion(EdgeConvQuaternionMergeMotion):
    """Winner path with RMS quaternion collapse instead of raw squared-energy collapse."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20, merge_eps=1e-6):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
        )
        self.merge_eps = merge_eps

    def merge_quaternions(self, encoded):
        return quaternion_rms_merge(encoded, eps=self.merge_eps)


class EdgeConvQuaternionWeightedRMSMergeMotion(EdgeConvQuaternionRMSMergeMotion):
    """RMS winner with learnable per-component weights in the quaternion collapse."""

    def __init__(self, num_classes, pts_size, hidden_dims=(64, 128), dropout=0.1, edgeconv_k=20, merge_eps=1e-6):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        self.merge_component_logits = nn.Parameter(torch.zeros(4))

    def merge_quaternions(self, encoded):
        return quaternion_weighted_rms_merge(
            encoded,
            component_weights=self.merge_component_logits,
            eps=self.merge_eps,
        )


class EdgeConvQuaternionStackedWeightedRMSMergeMotion(EdgeConvQuaternionWeightedRMSMergeMotion):
    """Weighted RMS winner with one extra quaternion refinement stage before collapse."""

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        _, hidden2 = hidden_dims
        self.quaternion_refine = QuaternionPointLinear(hidden2, hidden2)
        self.refine_norm = nn.BatchNorm1d(hidden2)
        self.refine_activation = nn.GELU()

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)

        refined = self.quaternion_refine(encoded.transpose(1, 2).contiguous())
        refined = self.refine_norm(refined.transpose(1, 2).contiguous())
        refined = self.refine_activation(refined)
        encoded = encoded + refined
        encoded = self.merge_proj(self.merge_quaternions(encoded))

        pooled_max = encoded.max(dim=-1).values
        pooled_mean = encoded.mean(dim=-1)
        return torch.cat((pooled_max, pooled_mean), dim=1)


class EdgeConvQuaternionStackedWeightedRMSAttentionReadoutMotion(EdgeConvQuaternionStackedWeightedRMSMergeMotion):
    """Stacked winner with an attention-pooled readout instead of plain mean pooling."""

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        _, hidden2 = hidden_dims
        self.readout_attention = nn.Conv1d(hidden2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.readout_attention.weight)
        nn.init.zeros_(self.readout_attention.bias)

    def extract_features(self, inputs, aux_input=None):
        points, _ = self._sample_points_with_aux(inputs, aux_input=aux_input)
        batch_size = points.shape[0]
        point_features = points.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()

        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values

        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)

        refined = self.quaternion_refine(encoded.transpose(1, 2).contiguous())
        refined = self.refine_norm(refined.transpose(1, 2).contiguous())
        refined = self.refine_activation(refined)
        encoded = encoded + refined

        encoded = self.merge_proj(self.merge_quaternions(encoded))

        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


class EdgeConvQuaternionStackedDualMergeWeightedRMSAttentionReadoutMotion(
    EdgeConvQuaternionStackedWeightedRMSAttentionReadoutMotion
):
    """Winner path with a dual quaternion collapse: weighted RMS plus real-part summary."""

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        _, hidden2 = hidden_dims
        self.merge_proj = nn.Sequential(
            nn.Conv1d(hidden2 // 2, hidden2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
        )

    def merge_quaternions(self, encoded):
        grouped = _reshape_quaternion_groups(encoded)
        merged_rms = quaternion_weighted_rms_merge(
            encoded,
            component_weights=self.merge_component_logits,
            eps=self.merge_eps,
        )
        real_part = grouped[:, :, 0, :]
        return torch.cat((merged_rms, real_part), dim=1)


# ---------------------------------------------------------------------------
# Bearing-quaternion QCC: geometric rigidity feature
# ---------------------------------------------------------------------------


def _compute_bearing_qcc_with_correspondence(points_4d, num_frames, knn_k=10,
                                               orig_flat_idx=None,
                                               corr_full_target_idx=None,
                                               corr_full_weight=None,
                                               min_valid_ratio=0.3):
    """Bearing QCC with proper point correspondence and fallback.

    When correspondence data is provided and enough valid matches exist
    (>= min_valid_ratio of points), uses true correspondences to pair
    points across frames.  Otherwise falls back to positional matching
    (same index in next frame), which is what the non-correspondence
    path always did.

    Returns:
        rigidity: (batch, 1, num_frames * pts_per_frame)
        valid_ratio: float, fraction of points with resolved correspondence
    """
    batch_size = points_4d.shape[0]
    pts_per_frame = points_4d.shape[2]
    device = points_4d.device
    use_corr = (orig_flat_idx is not None and corr_full_target_idx is not None
                and corr_full_weight is not None)

    xyz = points_4d[..., :3]
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
    bearing_q = torch.stack([w, axis[...,0]*sin_ha, axis[...,1]*sin_ha,
                             axis[...,2]*sin_ha], dim=-1)

    n_transitions = num_frames - 1
    inconsistency = torch.zeros(batch_size, n_transitions, pts_per_frame, device=device)
    valid_mask_all = torch.ones(batch_size, n_transitions, pts_per_frame, device=device)
    total_corr_valid = 0
    total_corr_possible = 0

    for t in range(n_transitions):
        q_src = bearing_q[:, t]  # (B, P, 4)

        if use_corr:
            q_tgt = bearing_q[:, t+1].clone()  # start with positional fallback
            frame_valid = torch.ones(batch_size, pts_per_frame, device=device)
            raw_ppf = corr_full_target_idx.shape[-1] // num_frames
            transition_valid = 0
            transition_total = batch_size * pts_per_frame

            for b in range(batch_size):
                src_orig = orig_flat_idx[b, t].long()  # (P,)
                tgt_flat = corr_full_target_idx[b, src_orig]  # (P,)
                tgt_w = corr_full_weight[b, src_orig]  # (P,)
                tgt_frame = tgt_flat // raw_ppf

                valid = (tgt_flat >= 0) & (tgt_w > 0) & (tgt_frame == t + 1)

                if valid.any():
                    tgt_orig_next = orig_flat_idx[b, t+1].long()  # (P,)
                    valid_tgt_flat = tgt_flat[valid]  # (V,)
                    match_matrix = (valid_tgt_flat.unsqueeze(1) == tgt_orig_next.unsqueeze(0))  # (V, P)
                    has_match = match_matrix.any(dim=1)  # (V,)
                    matched_j = match_matrix.float().argmax(dim=1)  # (V,)

                    valid_indices = valid.nonzero(as_tuple=True)[0]
                    for_write = valid_indices[has_match]
                    matched_sampled = matched_j[has_match]
                    q_tgt[b, for_write] = bearing_q[b, t+1, matched_sampled]
                    transition_valid += for_write.shape[0]

            total_corr_valid += transition_valid
            total_corr_possible += transition_total

            # If too few correspondences resolved, this transition's valid_mask
            # stays all-ones (positional fallback already in q_tgt for unmatched)
        else:
            q_tgt = bearing_q[:, t+1]

        q_src_conj = q_src * torch.tensor([1,-1,-1,-1], device=device, dtype=q_src.dtype)
        aw,ax,ay,az = q_tgt[...,0],q_tgt[...,1],q_tgt[...,2],q_tgt[...,3]
        bw,bx,by,bz = q_src_conj[...,0],q_src_conj[...,1],q_src_conj[...,2],q_src_conj[...,3]
        q_fwd = torch.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                             aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], dim=-1)
        q_fwd = F.normalize(q_fwd, dim=-1)

        pts_t = xyz[:, t].transpose(1, 2).contiguous()
        knn_idx = _knn_indices(pts_t, knn_k)
        idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 4)
        q_fwd_flat = q_fwd.unsqueeze(1).expand(-1, pts_per_frame, -1, -1)
        nbr_q_fwd = torch.gather(q_fwd_flat, 2, idx_exp)
        q_center = q_fwd.unsqueeze(2)
        dot_prod = (q_center * nbr_q_fwd).sum(dim=-1).abs().clamp(0, 1-1e-7)
        geo_dist = 2 * torch.acos(dot_prod)
        inconsistency[:, t] = geo_dist.mean(dim=-1)

    # Always use all points (correspondence improves q_tgt where available,
    # positional fallback elsewhere)
    mean_inconsistency = inconsistency.mean(dim=1)

    if mean_inconsistency.max() > 0:
        scale = mean_inconsistency.median().clamp(min=1e-6)
        rigidity = torch.exp(-mean_inconsistency / scale)
    else:
        rigidity = torch.ones_like(mean_inconsistency)

    valid_ratio = (total_corr_valid / max(total_corr_possible, 1)) if use_corr else 1.0

    rigidity_expanded = rigidity.unsqueeze(1).expand(-1, num_frames, -1)
    return rigidity_expanded.reshape(batch_size, 1, -1), valid_ratio



def _compute_bearing_qcc(points_4d, num_frames, knn_k=10):
    """Compute per-point bearing QCC score from raw XYZ across frames.

    For each point, compute the bearing quaternion (rotation from NORTH to
    the direction from bbox centroid to that point).  Then q_fwd = q_{f+1} *
    conj(q_f) captures per-point angular change between frames.  For rigid
    motion all nearby points share the same q_fwd; pairwise geodesic distance
    among k-NN q_fwd values measures deviation from rigidity.

    Args:
        points_4d: (batch, num_frames, pts_per_frame, 4) raw input
        num_frames: int
        knn_k: number of neighbors for pairwise comparison

    Returns:
        bearing_qcc: (batch, 1, num_frames * pts_per_frame) in [0, 1]
            0 = high inconsistency (deforming), 1 = consistent (rigid)
    """
    batch_size = points_4d.shape[0]
    pts_per_frame = points_4d.shape[2]
    device = points_4d.device

    xyz = points_4d[..., :3]  # (batch, num_frames, pts_per_frame, 3)

    # Centroid per frame: bbox center
    bbox_min = xyz.min(dim=2).values  # (batch, num_frames, 3)
    bbox_max = xyz.max(dim=2).values
    centroids = (bbox_min + bbox_max) / 2  # (batch, num_frames, 3)

    # Direction from centroid to each point
    directions = xyz - centroids.unsqueeze(2)  # (batch, nf, pts, 3)
    dir_norm = directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    directions = directions / dir_norm  # unit vectors

    # Convert directions to bearing quaternions
    # q = rotation from NORTH=[0,1,0] to direction d
    # axis = cross(north, d), angle = acos(dot(north, d))
    # w = cos(angle/2), xyz = axis * sin(angle/2)
    dot = directions[..., 1].clamp(-1 + 1e-7, 1 - 1e-7)  # dot with [0,1,0] = y component
    angle = torch.acos(dot)  # (batch, nf, pts)
    half_angle = angle / 2

    # cross([0,1,0], d) = [d_z, 0, -d_x]  (for unit north)
    axis = torch.stack([
        directions[..., 2],
        torch.zeros_like(directions[..., 0]),
        -directions[..., 0],
    ], dim=-1)  # (batch, nf, pts, 3)
    axis_norm = axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    axis = axis / axis_norm

    w = torch.cos(half_angle)  # (batch, nf, pts)
    sin_ha = torch.sin(half_angle)
    qx = axis[..., 0] * sin_ha
    qy = axis[..., 1] * sin_ha  # always 0
    qz = axis[..., 2] * sin_ha

    # bearing_q: (batch, nf, pts, 4) as [w, x, y, z]
    bearing_q = torch.stack([w, qx, qy, qz], dim=-1)

    # q_fwd = q_{f+1} * conj(q_f) for each point, each frame transition
    q_curr = bearing_q[:, :-1]  # (batch, nf-1, pts, 4)
    q_next = bearing_q[:, 1:]   # (batch, nf-1, pts, 4)
    # conj(q) = [w, -x, -y, -z]
    q_curr_conj = q_curr * torch.tensor([1, -1, -1, -1], device=device, dtype=q_curr.dtype)

    # Hamilton product: q_next * conj(q_curr)
    aw, ax, ay, az = q_next[..., 0], q_next[..., 1], q_next[..., 2], q_next[..., 3]
    bw, bx, by, bz = q_curr_conj[..., 0], q_curr_conj[..., 1], q_curr_conj[..., 2], q_curr_conj[..., 3]

    q_fwd_w = aw*bw - ax*bx - ay*by - az*bz
    q_fwd_x = aw*bx + ax*bw + ay*bz - az*by
    q_fwd_y = aw*by - ax*bz + ay*bw + az*bx
    q_fwd_z = aw*bz + ax*by - ay*bx + az*bw

    # q_fwd: (batch, nf-1, pts, 4)
    q_fwd = torch.stack([q_fwd_w, q_fwd_x, q_fwd_y, q_fwd_z], dim=-1)
    q_fwd = F.normalize(q_fwd, dim=-1)

    # For each point, compare its q_fwd with k-NN neighbors' q_fwd
    n_transitions = num_frames - 1

    # Per-point inconsistency score (mean geodesic distance to neighbors' q_fwd)
    inconsistency = torch.zeros(batch_size, n_transitions, pts_per_frame, device=device)

    for t in range(n_transitions):
        # Spatial k-NN from frame t
        pts_t = xyz[:, t].transpose(1, 2).contiguous()  # (batch, 3, pts)
        knn_idx = _knn_indices(pts_t, knn_k)  # (batch, pts, k)

        # Gather neighbor q_fwd
        q_fwd_t = q_fwd[:, t]  # (batch, pts, 4)
        idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 4)  # (batch, pts, k, 4)
        q_fwd_flat = q_fwd_t.unsqueeze(1).expand(-1, pts_per_frame, -1, -1)
        nbr_q_fwd = torch.gather(q_fwd_flat, 2, idx_exp)
        # nbr_q_fwd: (batch, pts, k, 4)

        # Geodesic distance: 2*arccos(|q1?q2|)
        q_center = q_fwd_t.unsqueeze(2)  # (batch, pts, 1, 4)
        dot_prod = (q_center * nbr_q_fwd).sum(dim=-1).abs()  # (batch, pts, k)
        dot_prod = dot_prod.clamp(0, 1 - 1e-7)
        geo_dist = 2 * torch.acos(dot_prod)  # (batch, pts, k)

        # Mean geodesic distance to neighbors
        inconsistency[:, t] = geo_dist.mean(dim=-1)  # (batch, pts)

    # Average inconsistency across frame transitions per point
    mean_inconsistency = inconsistency.mean(dim=1)  # (batch, pts)

    # Convert to rigidity score: high inconsistency = low rigidity
    if mean_inconsistency.max() > 0:
        scale = mean_inconsistency.median().clamp(min=1e-6)
        rigidity = torch.exp(-mean_inconsistency / scale)
    else:
        rigidity = torch.ones_like(mean_inconsistency)

    # Expand to all frames (each point gets the same score across frames)
    rigidity_expanded = rigidity.unsqueeze(1).expand(-1, num_frames, -1)
    rigidity_flat = rigidity_expanded.reshape(batch_size, 1, -1)

    return rigidity_flat


def _hamilton_product(a, b):
    """Hamilton product of two quaternion tensors (..., 4) in [w,x,y,z] format."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _quaternion_rotate_vector(q, v):
    """Rotate 3D vectors v by unit quaternions q.

    Args:
        q: (..., 4) unit quaternions [w,x,y,z]
        v: (..., 3) vectors

    Returns:
        rotated: (..., 3) rotated vectors
    """
    # v as pure quaternion [0, vx, vy, vz]
    v_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_conj = q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)
    rotated = _hamilton_product(_hamilton_product(q, v_quat), q_conj)
    return rotated[..., 1:]  # drop w component


class _GroundedCycleConsistency(nn.Module):
    """Grounded quaternion cycle consistency module.

    Splits encoded features into N temporal segments, estimates quaternion
    rotations between consecutive cyclic pairs.  Each quaternion is grounded
    by a reconstruction loss: rotating the source segment's pooled features
    should match the target segment.  The cycle constraint
    (q_{0,1} * q_{1,2} * ... * q_{N-1,0} = identity) adds mutual consistency
    on top, which is the novel signal.

    Default num_segments=3 preserves the original shallow-MLP behaviour.
    """

    def __init__(self, feat_dim, num_segments=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_segments = num_segments
        # Quaternion estimator: from concatenated segment summaries to [w,x,y,z]
        self.quat_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4),
        )

    def _estimate_quaternion(self, src_pooled, tgt_pooled):
        """Estimate unit quaternion from pooled segment features."""
        combined = torch.cat([src_pooled, tgt_pooled], dim=-1)
        q = self.quat_head(combined)  # (batch, 4)
        return F.normalize(q, dim=-1)  # project to unit sphere

    def forward(self, encoded, num_frames, pts_per_frame, points_xyz):
        """Compute grounded cycle consistency loss.

        Args:
            encoded: (batch, feat_dim, num_points) encoder features
            num_frames: int
            pts_per_frame: int
            points_xyz: (batch, num_frames, pts_per_frame, 3) raw XYZ coords

        Returns:
            loss: scalar (reconstruction + cycle)
            metrics: dict
        """
        batch = encoded.shape[0]
        device = encoded.device
        N = self.num_segments
        seg_size = max(num_frames // N, 1)

        # Pool features per segment.  Last segment absorbs the remainder.
        feat = encoded.permute(0, 2, 1).reshape(
            batch, num_frames, pts_per_frame, self.feat_dim,
        )
        seg_feats = []
        seg_xyzs = []
        for k in range(N):
            start = k * seg_size
            end = (k + 1) * seg_size if k < N - 1 else num_frames
            if start >= num_frames:
                start = max(num_frames - 1, 0)
                end = num_frames
            seg_feat = feat[:, start:end].reshape(batch, -1, self.feat_dim).mean(dim=1)
            seg_xyz = points_xyz[:, start:end].reshape(batch, -1, 3)
            seg_feats.append(seg_feat)
            seg_xyzs.append(seg_xyz)

        # Predict quaternion for each consecutive cyclic pair (0->1, ..., N-1->0)
        # Same detach pattern as the original 3-segment version: target is
        # frozen, gradient flows only through the source segment.
        quats = []
        recon_loss = torch.tensor(0.0, device=device)
        for i in range(N):
            j = (i + 1) % N
            q = self._estimate_quaternion(seg_feats[i], seg_feats[j].detach())
            quats.append(q)

            src = seg_xyzs[i]
            tgt = seg_xyzs[j]
            src_c = src - src.mean(dim=1, keepdim=True)
            tgt_c = tgt - tgt.mean(dim=1, keepdim=True)
            min_pts = min(src_c.shape[1], tgt_c.shape[1])
            src_c = src_c[:, :min_pts]
            tgt_c = tgt_c[:, :min_pts]
            rotated = _quaternion_rotate_vector(
                q.unsqueeze(1).expand(-1, src_c.shape[1], -1), src_c,
            )
            recon_loss = recon_loss + F.mse_loss(rotated, tgt_c.detach())
        recon_loss = recon_loss / N

        # Cycle: composition of all N quats should be identity
        q_cycle = quats[0]
        for q in quats[1:]:
            q_cycle = _hamilton_product(q_cycle, q)

        q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        loss_pos = ((q_cycle - q_id) ** 2).sum(dim=-1)
        loss_neg = ((q_cycle + q_id) ** 2).sum(dim=-1)
        cycle_loss = torch.min(loss_pos, loss_neg).mean()

        total = recon_loss + cycle_loss

        metrics = {
            'cycle_raw': cycle_loss.detach(),
            'recon_raw': recon_loss.detach(),
            'q_cycle_w': q_cycle[:, 0].abs().mean().detach(),
        }
        return total, metrics


class _GroundedCycleConsistencyDeep(nn.Module):
    """Deeper-MLP grounded cycle consistency, generalized to N segments.

    Same forward semantics as _GroundedCycleConsistency:
      - Pool features per segment, predict pairwise quaternions from
        [src, target.detach()] using an MLP head
      - Reconstruction loss against raw centered XYZ
      - Cycle composition constraint q_{1,2}*q_{2,3}*...*q_{N,1} = identity

    Differences:
      - Configurable num_segments (3, 6, 9, ...) instead of hardcoded 3
      - Deeper MLP head (configurable n_hidden_layers) for more capacity

    Why not transformer: attention layers leak gradient through the
    .detach() pattern (token-level detach is bypassed by the attention
    weights), causing degenerate self-referential cycle solutions that
    destabilize the encoder.  The MLP keeps the gradient flow clean.
    """

    def __init__(self, feat_dim, num_segments=3, n_hidden_layers=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_segments = num_segments

        # Deep MLP head: takes [src, target.detach()] -> quaternion
        # n_hidden_layers controls depth (original MLP was 2 layers).
        layers = [
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
        ]
        for _ in range(max(n_hidden_layers - 1, 0)):
            layers += [
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
            ]
        layers += [nn.Linear(feat_dim, 4)]
        self.quat_head = nn.Sequential(*layers)

    def _estimate_quaternion(self, src_pooled, tgt_pooled):
        combined = torch.cat([src_pooled, tgt_pooled], dim=-1)
        q = self.quat_head(combined)
        return F.normalize(q, dim=-1)

    def forward(self, encoded, num_frames, pts_per_frame, points_xyz):
        batch = encoded.shape[0]
        device = encoded.device
        N = self.num_segments
        seg_size = max(num_frames // N, 1)

        # Pool features per segment.  Last segment absorbs the remainder.
        feat = encoded.permute(0, 2, 1).reshape(
            batch, num_frames, pts_per_frame, self.feat_dim,
        )
        seg_feats = []
        seg_xyzs = []
        for k in range(N):
            start = k * seg_size
            end = (k + 1) * seg_size if k < N - 1 else num_frames
            if start >= num_frames:
                start = max(num_frames - 1, 0)
                end = num_frames
            seg_feat = feat[:, start:end].reshape(batch, -1, self.feat_dim).mean(dim=1)
            seg_xyz = points_xyz[:, start:end].reshape(batch, -1, 3)
            seg_feats.append(seg_feat)
            seg_xyzs.append(seg_xyz)

        # Predict quaternion for each consecutive pair (with wrap-around)
        # Same detach pattern as the original 3-segment version: target is
        # frozen, gradient flows only through the source segment.
        quats = []
        recon_loss = torch.tensor(0.0, device=device)
        for i in range(N):
            j = (i + 1) % N
            q = self._estimate_quaternion(seg_feats[i], seg_feats[j].detach())
            quats.append(q)

            src = seg_xyzs[i]
            tgt = seg_xyzs[j]
            src_c = src - src.mean(dim=1, keepdim=True)
            tgt_c = tgt - tgt.mean(dim=1, keepdim=True)
            min_pts = min(src_c.shape[1], tgt_c.shape[1])
            src_c = src_c[:, :min_pts]
            tgt_c = tgt_c[:, :min_pts]
            rotated = _quaternion_rotate_vector(
                q.unsqueeze(1).expand(-1, src_c.shape[1], -1), src_c,
            )
            recon_loss = recon_loss + F.mse_loss(rotated, tgt_c.detach())
        recon_loss = recon_loss / N

        # Cycle: composition of all N quats should be identity
        q_cycle = quats[0]
        for q in quats[1:]:
            q_cycle = _hamilton_product(q_cycle, q)

        q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        loss_pos = ((q_cycle - q_id) ** 2).sum(dim=-1)
        loss_neg = ((q_cycle + q_id) ** 2).sum(dim=-1)
        cycle_loss = torch.min(loss_pos, loss_neg).mean()

        total = recon_loss + cycle_loss

        metrics = {
            'cycle_raw': cycle_loss.detach(),
            'recon_raw': recon_loss.detach(),
            'q_cycle_w': q_cycle[..., 0].abs().mean().detach(),
        }
        return total, metrics


def _compute_bearing_qcc_aligned(points_4d, num_frames, knn_k=10, corr_matched=None):
    """Bearing QCC for correspondence-aligned point sampling.

    When points are sampled with correspondence-guided alignment, point index i
    in frame t corresponds to point index i in frame t+1 (where corr_matched
    is True).  This makes the q_fwd computation trivial.
    """
    batch_size = points_4d.shape[0]
    pts_per_frame = points_4d.shape[2]
    device = points_4d.device

    xyz = points_4d[..., :3]
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

    n_transitions = num_frames - 1
    inconsistency = torch.zeros(batch_size, n_transitions, pts_per_frame, device=device)

    for t in range(n_transitions):
        q_src = bearing_q[:, t]
        q_tgt = bearing_q[:, t + 1]  # same index = corresponding point

        q_src_conj = q_src * torch.tensor([1, -1, -1, -1], device=device, dtype=q_src.dtype)
        aw, ax, ay, az = q_tgt[..., 0], q_tgt[..., 1], q_tgt[..., 2], q_tgt[..., 3]
        bw, bx, by, bz = q_src_conj[..., 0], q_src_conj[..., 1], q_src_conj[..., 2], q_src_conj[..., 3]
        q_fwd = torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                             aw * bx + ax * bw + ay * bz - az * by,
                             aw * by - ax * bz + ay * bw + az * bx,
                             aw * bz + ax * by - ay * bx + az * bw], dim=-1)
        q_fwd = F.normalize(q_fwd, dim=-1)

        pts_t = xyz[:, t].transpose(1, 2).contiguous()
        knn_idx = _knn_indices(pts_t, knn_k)
        idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 4)
        q_fwd_flat = q_fwd.unsqueeze(1).expand(-1, pts_per_frame, -1, -1)
        nbr_q_fwd = torch.gather(q_fwd_flat, 2, idx_exp)
        q_center = q_fwd.unsqueeze(2)
        dot_prod = (q_center * nbr_q_fwd).sum(dim=-1).abs().clamp(0, 1 - 1e-7)
        geo_dist = 2 * torch.acos(dot_prod)
        inconsistency[:, t] = geo_dist.mean(dim=-1)

    mean_inconsistency = inconsistency.mean(dim=1)
    if mean_inconsistency.max() > 0:
        scale = mean_inconsistency.median().clamp(min=1e-6)
        rigidity = torch.exp(-mean_inconsistency / scale)
    else:
        rigidity = torch.ones_like(mean_inconsistency)

    valid_ratio = corr_matched.float().mean().item() if corr_matched is not None else 1.0
    rigidity_expanded = rigidity.unsqueeze(1).expand(-1, num_frames, -1)
    return rigidity_expanded.reshape(batch_size, 1, -1), valid_ratio


class _CorrespondenceContrastiveLoss(nn.Module):
    """Temporal feature consistency via correspondence.

    Uses cosine similarity so the signal is scale-invariant and always
    meaningful regardless of feature magnitude.  Loss = 1 - cos_sim
    for matched point pairs across adjacent frames.
    """

    def forward(self, encoded, num_frames, pts_per_frame, corr_matched):
        B, D, _ = encoded.shape
        feat = encoded.view(B, D, num_frames, pts_per_frame)
        loss = torch.tensor(0.0, device=encoded.device)
        count = 0
        for t in range(num_frames - 1):
            mask = corr_matched[:, t].float()
            n_valid = mask.sum()
            if n_valid < 1:
                continue
            feat_t = feat[:, :, t]          # (B, D, P)
            feat_next = feat[:, :, t + 1]   # (B, D, P)
            # Cosine similarity per point: dot(f_t, f_{t+1}) / (|f_t| * |f_{t+1}|)
            cos_sim = F.cosine_similarity(feat_t, feat_next.detach(), dim=1)  # (B, P)
            per_point_loss = 1.0 - cos_sim  # in [0, 2]
            loss = loss + (per_point_loss * mask).sum() / n_valid
            count += 1
        return loss / max(count, 1)


class _InfoNCETemporalLoss(nn.Module):
    """InfoNCE contrastive loss using correspondence-based positive pairs.

    For each point i at time t, the positive is the same physical point at
    t+1 (via correspondence); negatives are all other points at t+1.
    The softmax denominator always produces gradient, even when features
    are initially uniform.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, encoded, num_frames, pts_per_frame, corr_matched):
        B, D, _ = encoded.shape
        feat = encoded.view(B, D, num_frames, pts_per_frame)
        loss = torch.tensor(0.0, device=encoded.device)
        count = 0

        for t in range(num_frames - 1):
            mask = corr_matched[:, t]  # (B, P) bool
            for b in range(B):
                valid_idx = mask[b].nonzero(as_tuple=True)[0]
                if len(valid_idx) < 2:
                    continue
                anchors = F.normalize(feat[b, :, t, valid_idx].T, dim=-1)
                targets = F.normalize(feat[b, :, t + 1].T, dim=-1)  # all P points as candidates
                sim = anchors @ targets.T / self.temperature  # (V, P)
                labels = valid_idx  # positive is at the same index
                loss = loss + F.cross_entropy(sim, labels)
                count += 1

        return loss / max(count, 1)


class _GroundedDisplacementLoss(nn.Module):
    """Predict per-point XYZ displacement from encoded features.

    For each point at frame t, a small MLP predicts the 3D displacement
    to the same-index point at frame t+1.  Grounded in real geometry:
    the predicted vector must match the actual coordinate difference.

    No correspondence needed — operates on same-index points.
    Token: ``gd``.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 3),
        )

    def forward(self, encoded, points_xyz, num_frames, pts_per_frame):
        B, D, _ = encoded.shape
        feat = encoded.view(B, D, num_frames, pts_per_frame)

        loss = torch.tensor(0.0, device=encoded.device)
        count = 0
        for t in range(num_frames - 1):
            feat_t = feat[:, :, t].permute(0, 2, 1)  # (B, P, D)
            actual_disp = points_xyz[:, t + 1] - points_xyz[:, t]  # (B, P, 3)
            predicted_disp = self.predictor(feat_t)  # (B, P, 3)
            loss = loss + F.mse_loss(predicted_disp, actual_disp.detach())
            count += 1
        return loss / max(count, 1)


class _GroundedDisplacementDirectionLoss(nn.Module):
    """Predict per-point displacement DIRECTION (unit vector) from features.

    Same as _GroundedDisplacementLoss but normalizes both prediction and
    target to unit vectors and uses cosine loss.  Focuses on WHICH WAY
    each point moves, ignoring magnitude.  Different gestures differ more
    in motion direction than speed.

    Token: ``gd_dir``.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 3),
        )

    def forward(self, encoded, points_xyz, num_frames, pts_per_frame):
        B, D, _ = encoded.shape
        feat = encoded.view(B, D, num_frames, pts_per_frame)

        loss = torch.tensor(0.0, device=encoded.device)
        count = 0
        for t in range(num_frames - 1):
            feat_t = feat[:, :, t].permute(0, 2, 1)  # (B, P, D)
            actual_disp = points_xyz[:, t + 1] - points_xyz[:, t]  # (B, P, 3)
            # Skip near-zero displacements (stationary points)
            disp_norm = actual_disp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            actual_dir = actual_disp / disp_norm
            predicted_dir = self.predictor(feat_t)  # (B, P, 3)
            # Cosine loss: 1 - cos_sim
            cos_sim = F.cosine_similarity(predicted_dir, actual_dir.detach(), dim=-1)
            loss = loss + (1.0 - cos_sim).mean()
            count += 1
        return loss / max(count, 1)


class _GroundedDisplacementBidirLoss(nn.Module):
    """Predict displacement BOTH forward (t->t+1) AND backward (t->t-1).

    Two separate heads predict forward and backward displacement from the
    same encoded features.  Doubles the grounding signal: each point must
    encode both where it came from and where it's going.

    Token: ``gd_bidir``.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.fwd_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 3),
        )
        self.bwd_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 3),
        )

    def forward(self, encoded, points_xyz, num_frames, pts_per_frame):
        B, D, _ = encoded.shape
        feat = encoded.view(B, D, num_frames, pts_per_frame)

        loss = torch.tensor(0.0, device=encoded.device)
        count = 0
        for t in range(num_frames):
            feat_t = feat[:, :, t].permute(0, 2, 1)  # (B, P, D)
            # Forward: predict t -> t+1
            if t < num_frames - 1:
                fwd_disp = points_xyz[:, t + 1] - points_xyz[:, t]
                pred_fwd = self.fwd_predictor(feat_t)
                loss = loss + F.mse_loss(pred_fwd, fwd_disp.detach())
                count += 1
            # Backward: predict t -> t-1
            if t > 0:
                bwd_disp = points_xyz[:, t - 1] - points_xyz[:, t]
                pred_bwd = self.bwd_predictor(feat_t)
                loss = loss + F.mse_loss(pred_bwd, bwd_disp.detach())
                count += 1
        return loss / max(count, 1)


class _BearingRotationQCCLoss(nn.Module):
    """True per-point bearing-rotation QCC.

    Predicts per-point quaternion that rotates bearing(t) to bearing(t+1) from
    encoded features. This is the literal quaternion cross-correlation: a
    quaternion-valued prediction supervised against the geometric quaternion
    rotating the point's bearing vector between frames.

    Token: ``qr``.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4),
        )

    @staticmethod
    def _bearing(xyz):
        # xyz: (B, F, P, 3). Center per-frame then normalize to unit vectors.
        bbox_min = xyz.min(dim=2).values
        bbox_max = xyz.max(dim=2).values
        centroids = (bbox_min + bbox_max) / 2
        d = xyz - centroids.unsqueeze(2)
        return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    @staticmethod
    def _rotation_quat(a, b):
        # Quaternion rotating unit vector a -> unit vector b.
        # q = [1 + a.b, a x b], normalized. Falls back to 180-deg handling.
        dot = (a * b).sum(dim=-1, keepdim=True)
        cross = torch.cross(a, b, dim=-1)
        w = 1.0 + dot
        q = torch.cat([w, cross], dim=-1)
        # 180-deg case: a ~= -b, use any perpendicular axis.
        near_opposite = (w.squeeze(-1) < 1e-6)
        if near_opposite.any():
            # Pick axis perpendicular to a.
            ex = torch.zeros_like(a); ex[..., 0] = 1.0
            ey = torch.zeros_like(a); ey[..., 1] = 1.0
            use_ey = (a[..., 0].abs() > 0.9).unsqueeze(-1)
            axis = torch.where(use_ey, ey, ex)
            perp = axis - (axis * a).sum(dim=-1, keepdim=True) * a
            perp = perp / perp.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            q_flip = torch.cat([torch.zeros_like(dot), perp], dim=-1)
            q = torch.where(near_opposite.unsqueeze(-1), q_flip, q)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def forward(self, encoded, points_xyz, num_frames, pts_per_frame):
        # encoded: (B, D, F*P); points_xyz: (B, F, P, 3)
        B, D, _ = encoded.shape
        feat = encoded.view(B, D, num_frames, pts_per_frame)
        bearings = self._bearing(points_xyz)  # (B, F, P, 3)

        loss = torch.tensor(0.0, device=encoded.device)
        count = 0
        for t in range(num_frames - 1):
            feat_t = feat[:, :, t].permute(0, 2, 1)  # (B, P, D)
            a = bearings[:, t]        # (B, P, 3)
            b = bearings[:, t + 1]    # (B, P, 3)
            q_gt = self._rotation_quat(a, b).detach()  # (B, P, 4)

            q_pred = self.predictor(feat_t)  # (B, P, 4)
            q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            # Geodesic loss: 1 - (q_pred . q_gt)^2 handles q ~ -q equivalence.
            dot = (q_pred * q_gt).sum(dim=-1)
            loss = loss + (1.0 - dot.pow(2)).mean()
            count += 1
        return loss / max(count, 1)



class _PartsFeatureProcrustes(nn.Module):
    """K-part Procrustes rotation + rigidity residual, used as features only.

    Returns (aux_loss, features):
      aux_loss: scalar entropy-collapse penalty (tiny) -- keeps parts distinct.
      features: (B, F-1, K, 5) with [qw, qx, qy, qz, rigidity_residual] per part.
    """

    def __init__(self, feat_dim, num_parts=6, entropy_weight=0.01):
        super().__init__()
        self.num_parts = num_parts
        self.entropy_weight = entropy_weight
        self.assign_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, num_parts),
        )

    @staticmethod
    def _rot_to_quat(R):
        m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
        m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
        m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
        tr = m00 + m11 + m22
        eps = 1e-8
        s1 = torch.sqrt(torch.clamp(1 + tr, min=eps)) * 2
        q1 = torch.stack([0.25 * s1, (m21 - m12) / s1, (m02 - m20) / s1, (m10 - m01) / s1], dim=-1)
        s2 = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=eps)) * 2
        q2 = torch.stack([(m21 - m12) / s2, 0.25 * s2, (m01 + m10) / s2, (m02 + m20) / s2], dim=-1)
        s3 = torch.sqrt(torch.clamp(1 + m11 - m00 - m22, min=eps)) * 2
        q3 = torch.stack([(m02 - m20) / s3, (m01 + m10) / s3, 0.25 * s3, (m12 + m21) / s3], dim=-1)
        s4 = torch.sqrt(torch.clamp(1 + m22 - m00 - m11, min=eps)) * 2
        q4 = torch.stack([(m10 - m01) / s4, (m02 + m20) / s4, (m12 + m21) / s4, 0.25 * s4], dim=-1)
        cond1 = tr > 0
        cond2 = (m00 >= m11) & (m00 >= m22)
        cond3 = m11 >= m22
        q_nt = torch.where(cond2.unsqueeze(-1), q2,
                           torch.where(cond3.unsqueeze(-1), q3, q4))
        q = torch.where(cond1.unsqueeze(-1), q1, q_nt)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=eps)

    def forward(self, encoded, points_xyz_flat, num_frames, pts_per_frame, corr_matched):
        B, D, _ = encoded.shape
        F_, P = num_frames, pts_per_frame
        K = self.num_parts
        device = encoded.device

        feat = encoded.transpose(1, 2).contiguous()
        logits = self.assign_head(feat)                  # (B, F*P, K)
        assign = torch.softmax(logits, dim=-1).view(B, F_, P, K)

        xyz = points_xyz_flat.view(B, F_, P, 3)

        feats_list = []
        I3 = torch.eye(3, device=device).view(1, 1, 3, 3)

        for t in range(F_ - 1):
            src = xyz[:, t]
            tgt = xyz[:, t + 1]
            mask = corr_matched[:, t].float() if corr_matched is not None \
                else torch.ones(B, P, device=device)
            sa = assign[:, t]
            ta = assign[:, t + 1]

            w_k = (sa * ta).permute(0, 2, 1) * mask.unsqueeze(1)       # (B, K, P)
            w_sum = w_k.sum(dim=-1, keepdim=True).clamp(min=1e-6)      # (B, K, 1)

            src_b = src.unsqueeze(1).expand(B, K, P, 3)
            tgt_b = tgt.unsqueeze(1).expand(B, K, P, 3)
            src_mean = (w_k.unsqueeze(-1) * src_b).sum(dim=-2) / w_sum
            tgt_mean = (w_k.unsqueeze(-1) * tgt_b).sum(dim=-2) / w_sum
            src_c = src_b - src_mean.unsqueeze(-2)
            tgt_c = tgt_b - tgt_mean.unsqueeze(-2)

            H = torch.einsum("bkp,bkpi,bkpj->bkij", w_k, src_c, tgt_c)
            H = H + 1e-6 * I3

            try:
                U, S, Vh = torch.linalg.svd(H)
            except Exception:
                R_used = I3.expand(B, K, 3, 3).contiguous()
                quats = self._rot_to_quat(R_used)
                residuals = torch.zeros(B, K, device=device)
                feats_list.append(torch.cat([quats, residuals.unsqueeze(-1)], dim=-1))
                continue

            V = Vh.transpose(-1, -2)
            det = torch.det(torch.matmul(V, U.transpose(-1, -2)))
            D_diag = torch.ones(B, K, 3, device=device)
            D_diag[..., -1] = det
            D_mat = torch.diag_embed(D_diag)
            R = torch.matmul(V, torch.matmul(D_mat, U.transpose(-1, -2)))

            # Detach R: feature-only path, no SVD gradient needed.
            R_used = R.detach()
            bad = ~torch.isfinite(R_used).all(dim=-1).all(dim=-1)
            if bad.any():
                R_used = torch.where(bad.unsqueeze(-1).unsqueeze(-1), I3.expand_as(R_used), R_used)

            pred = torch.einsum("bkij,bkpj->bkpi", R_used, src_c)
            residual_per_point = ((pred - tgt_c) ** 2).sum(dim=-1)
            residual_per_part = (w_k * residual_per_point).sum(dim=-1) / w_sum.squeeze(-1)

            quats = self._rot_to_quat(R_used)
            feats_list.append(torch.cat([quats, residual_per_part.unsqueeze(-1)], dim=-1))  # (B, K, 5)

        if feats_list:
            features = torch.stack(feats_list, dim=1)                    # (B, F-1, K, 5)
        else:
            features = torch.zeros(B, 0, K, 5, device=device)

        # Entropy collapse penalty (tiny)
        mean_assign = assign.mean(dim=(0, 1, 2))
        entropy = -(mean_assign * mean_assign.clamp(min=1e-8).log()).sum()
        max_entropy = torch.log(torch.tensor(float(K), device=device))
        collapse = (max_entropy - entropy).clamp(min=0.0)
        aux_loss = self.entropy_weight * collapse

        return aux_loss, features

class _TemporalPredictionLoss(nn.Module):
    """Predict next-frame features from current-frame features.

    A lightweight MLP predicts feat_{t+1} from feat_t for matched points.
    MSE is always non-zero from epoch 0, guaranteeing gradient flow.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, encoded, num_frames, pts_per_frame, corr_matched):
        B, D, _ = encoded.shape
        feat = encoded.view(B, D, num_frames, pts_per_frame)
        loss = torch.tensor(0.0, device=encoded.device)
        count = 0

        for t in range(num_frames - 1):
            mask = corr_matched[:, t].float()  # (B, P)
            n_valid = mask.sum()
            if n_valid < 1:
                continue
            feat_t = feat[:, :, t].permute(0, 2, 1)       # (B, P, D)
            feat_next = feat[:, :, t + 1].permute(0, 2, 1) # (B, P, D)
            predicted = self.predictor(feat_t)
            per_point = ((predicted - feat_next.detach()) ** 2).mean(dim=-1)
            loss = loss + (per_point * mask).sum() / n_valid
            count += 1

        return loss / max(count, 1)


class _LocalCycleConsistencyLoss(nn.Module):
    """Per-point cycle consistency on bearing quaternion forward rotations.

    For triplets of consecutive frames (t, t+1, t+2):
      q_composed = q_fwd(t->t+1) * q_fwd(t+1->t+2)
      q_direct   = q_fwd(t->t+2)
      loss = geodesic_distance(q_composed, q_direct)

    No global rotation assumption -- each point checked independently.
    """

    def forward(self, points_4d, num_frames, corr_matched=None):
        B, _, P, _ = points_4d.shape
        device = points_4d.device
        xyz = points_4d[..., :3]

        bbox_min = xyz.min(dim=2).values
        bbox_max = xyz.max(dim=2).values
        centroids = (bbox_min + bbox_max) / 2
        directions = xyz - centroids.unsqueeze(2)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        dot = directions[..., 1].clamp(-1 + 1e-7, 1 - 1e-7)
        half_angle = torch.acos(dot) / 2
        axis = torch.stack([directions[..., 2], torch.zeros_like(directions[..., 0]),
                            -directions[..., 0]], dim=-1)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        w = torch.cos(half_angle)
        sin_ha = torch.sin(half_angle)
        bearing_q = torch.stack([w, axis[..., 0] * sin_ha, axis[..., 1] * sin_ha,
                                 axis[..., 2] * sin_ha], dim=-1)

        conj_sign = torch.tensor([1, -1, -1, -1], device=device, dtype=bearing_q.dtype)

        # q_fwd(t->t+1) for each consecutive pair
        q_fwd = F.normalize(_hamilton_product(
            bearing_q[:, 1:], bearing_q[:, :-1] * conj_sign), dim=-1)

        # q_direct(t->t+2): skip one frame
        q_direct = F.normalize(_hamilton_product(
            bearing_q[:, 2:], bearing_q[:, :-2] * conj_sign), dim=-1)

        # q_composed = q_fwd(t+1->t+2) * q_fwd(t->t+1)
        q_composed = F.normalize(_hamilton_product(
            q_fwd[:, 1:], q_fwd[:, :-1]), dim=-1)

        dot_prod = (q_composed * q_direct).sum(dim=-1).abs().clamp(0, 1 - 1e-7)
        geo_dist = 2 * torch.acos(dot_prod)  # (B, F-2, P)

        if corr_matched is not None:
            mask = (corr_matched[:, :-1] & corr_matched[:, 1:]).float()
            n_valid = mask.sum()
            if n_valid > 0:
                return (geo_dist * mask).sum() / n_valid
            return torch.tensor(0.0, device=device)
        return geo_dist.mean()


class _DisplacementAgreementLoss(nn.Module):
    """Feature consistency loss grounded in displacement agreement.

    Points whose spatial neighbors move consistently (rigid region) should
    have similar encoded features to those neighbors.
    """

    def __init__(self, knn_k=10):
        super().__init__()
        self.knn_k = knn_k

    def forward(self, encoded, points_4d, num_frames, pts_per_frame,
                corr_matched=None):
        B, D, _ = encoded.shape
        device = encoded.device
        xyz = points_4d[..., :3]
        feat = encoded.view(B, D, num_frames, pts_per_frame)

        loss = torch.tensor(0.0, device=device)
        count = 0

        for t in range(num_frames - 1):
            disp = xyz[:, t + 1] - xyz[:, t]  # (B, P, 3)
            pts_t = xyz[:, t].transpose(1, 2).contiguous()
            k = min(self.knn_k, pts_per_frame - 1)
            knn_idx = _knn_indices(pts_t, k)  # (B, P, k)

            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(knn_idx)
            nbr_disp = disp[batch_idx, knn_idx]  # (B, P, k, 3)
            disp_diff = nbr_disp - disp.unsqueeze(2)
            disp_var = (disp_diff ** 2).sum(dim=-1).mean(dim=-1)  # (B, P)
            scale = disp_var.median().clamp(min=1e-6)
            disp_rigidity = torch.exp(-disp_var / scale).detach()

            feat_t = feat[:, :, t].permute(0, 2, 1)  # (B, P, D)
            nbr_feat = feat_t[batch_idx, knn_idx]  # (B, P, k, D)
            cos_sim = F.cosine_similarity(
                feat_t.unsqueeze(2).expand_as(nbr_feat), nbr_feat, dim=-1)
            mean_sim = cos_sim.mean(dim=-1)  # (B, P)

            per_point = disp_rigidity * (1.0 - mean_sim)

            if corr_matched is not None:
                m = corr_matched[:, t].float()
                n_valid = m.sum()
                if n_valid > 0:
                    loss = loss + (per_point * m).sum() / n_valid
                    count += 1
            else:
                loss = loss + per_point.mean()
                count += 1

        return loss / max(count, 1)


def _compute_cycle_consistency_rigidity(points_4d, num_frames):
    """Cycle-consistency score per point from bearing-quaternion triplets.

    For each interior frame t in [1, F-2], for each point i, compute
    q_prev = rotation(bearing[t-1, i] -> bearing[t, i]) and q_next = rotation
    (bearing[t, i] -> bearing[t+1, i]).  Score = (q_prev . q_next)^2 in [0, 1].
    Score is high when motion is smooth (same rotation continued), low when
    motion oscillates or reverses.  Boundary frames copy nearest interior
    score.

    Args:
        points_4d: (B, F, P, 4)
        num_frames: int
    Returns:
        (B, 1, F*P) in [0, 1]
    """
    B, F, P, _ = points_4d.shape
    device = points_4d.device
    xyz = points_4d[..., :3]

    bbox_min = xyz.min(dim=2).values
    bbox_max = xyz.max(dim=2).values
    centroids = (bbox_min + bbox_max) / 2
    d = xyz - centroids.unsqueeze(2)
    bearings = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # (B, F, P, 3)

    def rotation_quat(a, b):
        # Unit-vector rotation: q = [1 + a.b, a x b], normalized.
        dot = (a * b).sum(dim=-1, keepdim=True)
        cross = torch.cross(a, b, dim=-1)
        q = torch.cat([1.0 + dot, cross], dim=-1)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    # Need F >= 3 for triplets. If F < 3 fall back to constant 1.
    if F < 3:
        return torch.ones(B, 1, F * P, device=device), 1.0

    scores = torch.zeros(B, F, P, device=device)
    for t in range(1, F - 1):
        a = bearings[:, t - 1]
        m = bearings[:, t]
        b = bearings[:, t + 1]
        q_prev = rotation_quat(a, m)
        q_next = rotation_quat(m, b)
        dot = (q_prev * q_next).sum(dim=-1)  # (B, P)
        scores[:, t] = dot.pow(2)
    # Boundary frames: copy nearest interior
    scores[:, 0] = scores[:, 1]
    scores[:, F - 1] = scores[:, F - 2]

    return scores.reshape(B, 1, -1), 1.0


def _compute_bearing_qcc_multiscale(points_4d, num_frames, scales=(5, 15, 40),
                                     corr_matched=None):
    """Bearing QCC at multiple spatial scales.

    Returns:
        rigidity: (B, len(scales), F*P)
        valid_ratio: float
    """
    rigidities = []
    valid_ratio = 1.0
    for k in scales:
        rig, vr = _compute_bearing_qcc_aligned(
            points_4d, num_frames, knn_k=k, corr_matched=corr_matched)
        rigidities.append(rig)
        valid_ratio = vr
    return torch.cat(rigidities, dim=1), valid_ratio


class BearingQCCFeatureMotion(
    EdgeConvQuaternionStackedDualMergeWeightedRMSAttentionReadoutMotion
):
    """Bearing-quaternion QCC feature with correspondence-guided sampling.

    Samples points using correspondence chains so that the same physical
    point is at the same index across frames.  This gives the bearing QCC
    rigidity signal proper point tracking, and enables a correspondence-
    contrastive auxiliary loss on encoder features.

    qcc_variant controls which auxiliary loss / rigidity computation to use:
      - 'grounded_cycle': pooled-segment quaternion cycle + XYZ reconstruction
        grounding (the 80.29% baseline loss). No correspondence required.
        cycle_module_type='mlp'      -> 3-segment shallow MLP head (baseline)
        cycle_module_type='deep_mlp' -> N-segment deeper MLP head with
                                        LayerNorm, configurable depth
      - 'contrastive': cosine contrastive loss on features
      - 'infonce': InfoNCE temporal contrastive with negatives
      - 'prediction': temporal feature prediction via MLP
      - 'local_cycle': per-point quaternion cycle consistency
      - 'displacement': displacement-agreement feature consistency
      - 'multiscale': multi-scale rigidity features, no aux loss
    """

    def __init__(
        self,
        num_classes,
        pts_size,
        hidden_dims=(64, 128),
        dropout=0.1,
        edgeconv_k=20,
        merge_eps=1e-6,
        so3_weight=0.0,
        rotation_sigma=0.3,
        bearing_knn_k=10,
        qcc_weight=0.1,
        qcc_variant='contrastive',
        rigidity_scales=(5, 15, 40),
        disable_rigidity=False,
        decouple_sampling=False,
        cycle_module_type='mlp',
        num_cycle_segments=3,
        cycle_n_hidden_layers=3,
    ):
        super().__init__(
            num_classes=num_classes,
            pts_size=pts_size,
            hidden_dims=hidden_dims,
            dropout=dropout,
            edgeconv_k=edgeconv_k,
            merge_eps=merge_eps,
        )
        _, hidden2 = hidden_dims
        self.so3_weight = so3_weight
        self.rotation_sigma = rotation_sigma
        self.bearing_knn_k = bearing_knn_k

        # Normalize qcc_variant / qcc_weight to lists so we can stack multiple
        # auxiliary losses (e.g. ['prediction', 'grounded_cycle']). A scalar
        # input keeps the legacy single-loss path.
        if isinstance(qcc_variant, (list, tuple)):
            self.qcc_variants = list(qcc_variant)
            if isinstance(qcc_weight, (list, tuple)):
                assert len(qcc_weight) == len(self.qcc_variants), \
                    'qcc_weight list must match qcc_variant list length'
                self.qcc_weights = list(qcc_weight)
            else:
                self.qcc_weights = [qcc_weight] * len(self.qcc_variants)
        else:
            self.qcc_variants = [qcc_variant]
            self.qcc_weights = [qcc_weight]
        # Backward-compatible scalar handles (first variant)
        self.qcc_variant = self.qcc_variants[0]
        self.qcc_weight = self.qcc_weights[0]

        self.rigidity_scales = rigidity_scales
        self.disable_rigidity = disable_rigidity
        self.decouple_sampling = decouple_sampling
        self.cycle_module_type = cycle_module_type
        self.num_cycle_segments = num_cycle_segments
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        is_multiscale = 'multiscale' in self.qcc_variants
        is_cycle_rig = 'cycle_rigidity' in self.qcc_variants
        is_cycle_side = 'cycle_rigidity_side' in self.qcc_variants
        if is_multiscale:
            rig_channels = len(rigidity_scales)
        elif is_cycle_rig:
            rig_channels = 2
        else:
            rig_channels = 1
        self.rigidity_proj = nn.Sequential(
            nn.Conv1d(rig_channels, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.rigidity_proj[0].weight)
        nn.init.zeros_(self.rigidity_proj[0].bias)

        # Side-path cycle projection (separate from rigidity_proj so v8a
        # weights stay pristine). Only active when qcc_variant='cycle_rigidity_side'.
        self.cycle_proj = nn.Sequential(
            nn.Conv1d(1, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.cycle_proj[0].weight)
        nn.init.zeros_(self.cycle_proj[0].bias)

        # Deeper cycle projection for cycle_rigidity_mlp variant.
        # 1 -> hidden2//2 -> hidden2, with zero-init final layer so start
        # output is exactly zero (matches v14a starting behavior).
        _cyc_mid = max(hidden2 // 2, 16)
        self.cycle_proj_deep = nn.Sequential(
            nn.Conv1d(1, _cyc_mid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(_cyc_mid, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        # Kaiming init for hidden layer (default); zero init for output layer
        nn.init.zeros_(self.cycle_proj_deep[2].weight)
        nn.init.zeros_(self.cycle_proj_deep[2].bias)

        self.corr_contrastive = _CorrespondenceContrastiveLoss()
        self.infonce_loss = _InfoNCETemporalLoss(temperature=0.1)
        self.prediction_loss = _TemporalPredictionLoss(feat_dim=hidden2)
        # Parts-feature (v18a, no aux loss path)
        num_parts_f = 6
        self.parts_feature_module = _PartsFeatureProcrustes(feat_dim=hidden2, num_parts=num_parts_f)
        self.parts_feature_proj = nn.Sequential(
            nn.Linear(num_parts_f * 5, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, hidden2),
        )

        self.grounded_disp_loss = _GroundedDisplacementLoss(feat_dim=hidden2)
        self.grounded_disp_dir_loss = _GroundedDisplacementDirectionLoss(feat_dim=hidden2)
        self.grounded_disp_bidir_loss = _GroundedDisplacementBidirLoss(feat_dim=hidden2)
        self.bearing_rot_qcc_loss = _BearingRotationQCCLoss(feat_dim=hidden2)
        self.local_cycle = _LocalCycleConsistencyLoss()
        self.displacement_loss = _DisplacementAgreementLoss(knn_k=bearing_knn_k)
        # Grounded cycle consistency (used by qcc_variant='grounded_cycle')
        # cycle_module_type='mlp':      shallow 1-hidden MLP head, configurable
        #                               num_cycle_segments (default 3, the
        #                               original 80.29% recipe)
        # cycle_module_type='deep_mlp': deeper MLP head + configurable
        #                               num_cycle_segments (3, 6, 9, ...)
        if cycle_module_type == 'deep_mlp':
            self.cycle_module = _GroundedCycleConsistencyDeep(
                feat_dim=hidden2,
                num_segments=num_cycle_segments,
                n_hidden_layers=cycle_n_hidden_layers,
            )
        else:
            self.cycle_module = _GroundedCycleConsistency(
                feat_dim=hidden2,
                num_segments=num_cycle_segments,
            )

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    @staticmethod
    def _random_rotation_matrix(batch_size, sigma, device):
        axis = torch.randn(batch_size, 3, device=device)
        axis = F.normalize(axis, dim=-1)
        angle = torch.randn(batch_size, 1, device=device) * sigma
        K = torch.zeros(batch_size, 3, 3, device=device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        I = torch.eye(3, device=device).unsqueeze(0)
        sin_a = torch.sin(angle).unsqueeze(-1)
        cos_a = torch.cos(angle).unsqueeze(-1)
        return I + sin_a * K + (1 - cos_a) * torch.bmm(K, K)

    def _encode_to_pre_merge(self, point_features):
        graph_features = _get_graph_feature(point_features, k=self.edgeconv_k)
        edge_features = self.edgeconv(graph_features).max(dim=-1).values
        encoded = self.quaternion_encoder(edge_features.transpose(1, 2).contiguous())
        encoded = self.encoder_norm(encoded.transpose(1, 2).contiguous())
        encoded = self.encoder_activation(encoded)
        refined = self.quaternion_refine(encoded.transpose(1, 2).contiguous())
        refined = self.refine_norm(refined.transpose(1, 2).contiguous())
        refined = self.refine_activation(refined)
        return encoded + refined

    def _correspondence_guided_sample(self, points, aux_input):
        """Sample points following correspondence chains across frames.

        Frame 0 is sampled randomly (train) or uniformly (eval).  Each
        subsequent frame picks the correspondence target of each point in
        the previous frame, falling back to random when no match exists.

        Returns:
            sampled: (B, F, S, C) reindexed points
            corr_matched: (B, F-1, S) bool mask of resolved correspondences
        """
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device

        if sample_size == pts_per_frame:
            corr_matched = torch.ones(batch_size, num_frames - 1, pts_per_frame,
                                      dtype=torch.bool, device=device)
            return points, corr_matched

        orig_flat_idx = aux_input['orig_flat_idx']
        corr_target = aux_input['corr_full_target_idx']
        corr_weight = aux_input['corr_full_weight']
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames

        sampled = torch.zeros(batch_size, num_frames, sample_size, channels,
                              device=device, dtype=points.dtype)
        corr_matched = torch.zeros(batch_size, num_frames - 1, sample_size,
                                   dtype=torch.bool, device=device)

        for b in range(batch_size):
            if self.training:
                idx = torch.randperm(pts_per_frame, device=device)[:sample_size]
            else:
                idx = torch.linspace(0, pts_per_frame - 1, sample_size,
                                     device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()

            for t in range(num_frames - 1):
                # Reverse map: orig_flat_idx value -> position in frame t+1
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(pts_per_frame, device=device)

                # Follow correspondence
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]

                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))

                # Correspondence where valid, random elsewhere
                next_idx = torch.randint(0, pts_per_frame, (sample_size,), device=device)
                next_idx[valid] = tgt_pos[valid]

                sampled[b, t + 1] = points[b, t + 1, next_idx]
                corr_matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()

        return sampled, corr_matched

    def _compute_corr_mask_for_independent_sample(self, aux_sampled, aux_input):
        """Compute correspondence mask for independently-sampled points.

        Points were sampled via _sample_points (random/linspace), NOT via
        correspondence chains.  We check which sampled points happen to have
        valid correspondences landing in the next frame's sampled set.
        """
        orig_flat_idx = aux_sampled['orig_flat_idx']  # (B, F, S)
        corr_target = aux_input['corr_full_target_idx']  # (B, total_pts)
        corr_weight = aux_input['corr_full_weight']  # (B, total_pts)

        batch_size = orig_flat_idx.shape[0]
        num_frames = orig_flat_idx.shape[1]
        pts_per_frame = orig_flat_idx.shape[2]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames
        device = orig_flat_idx.device

        corr_matched = torch.zeros(batch_size, num_frames - 1, pts_per_frame,
                                   dtype=torch.bool, device=device)

        for b in range(batch_size):
            for t in range(num_frames - 1):
                src_orig = orig_flat_idx[b, t].long()
                tgt_flat = corr_target[b, src_orig]
                tgt_w = corr_weight[b, src_orig]
                tgt_frame = tgt_flat // raw_ppf

                valid_src = (tgt_flat >= 0) & (tgt_w > 0) & (tgt_frame == t + 1)

                # Check if target lands in next frame's sampled set
                next_orig = orig_flat_idx[b, t + 1].long()
                # Build lookup set
                next_set = set(next_orig.cpu().tolist())
                tgt_flat_cpu = tgt_flat.cpu()
                for s in valid_src.nonzero(as_tuple=True)[0]:
                    if tgt_flat_cpu[s].item() in next_set:
                        corr_matched[b, t, s] = True

        return corr_matched

    def extract_features(self, inputs, aux_input=None):
        # Handle both direct dict input and pre-unpacked (points, aux) from forward()
        if isinstance(inputs, dict):
            points = inputs['points']
            aux_unpacked = inputs
        elif aux_input is not None:
            points = inputs
            aux_unpacked = aux_input
        else:
            points = inputs
            aux_unpacked = None

        has_corr = (aux_unpacked is not None
                    and 'orig_flat_idx' in aux_unpacked
                    and 'corr_full_target_idx' in aux_unpacked
                    and 'corr_full_weight' in aux_unpacked)

        if has_corr and not self.decouple_sampling:
            # Correspondence-guided sampling (legacy: coupled)
            points_4d = points[..., :4]
            sampled, corr_matched = self._correspondence_guided_sample(
                points_4d, aux_unpacked)
        elif has_corr and self.decouple_sampling:
            # Standard sampling + post-hoc correspondence mask
            sampled, sampled_aux = self._sample_points_with_aux(
                points, aux_input=aux_unpacked)
            sampled = sampled[..., :4]
            corr_matched = self._compute_corr_mask_for_independent_sample(
                sampled_aux, aux_unpacked)
        else:
            sampled = self._sample_points(points)
            corr_matched = None

        batch_size, num_frames, pts_per_frame, _ = sampled.shape

        # Bearing QCC rigidity
        if self.qcc_variant == 'multiscale':
            rigidity, corr_valid_ratio = _compute_bearing_qcc_multiscale(
                sampled, num_frames, scales=self.rigidity_scales,
                corr_matched=corr_matched)
        elif self.qcc_variant == 'cycle_rigidity':
            geom_rig, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)
            cyc_rig, _ = _compute_cycle_consistency_rigidity(sampled, num_frames)
            rigidity = torch.cat([geom_rig, cyc_rig], dim=1)
        elif self.qcc_variant in ('cycle_rigidity_side', 'cycle_rigidity_mlp'):
            rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)
            side_cyc, _ = _compute_cycle_consistency_rigidity(sampled, num_frames)
            # stash for use after rigidity_proj
            self._side_cyc = side_cyc
        else:
            rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)

        point_features = sampled.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()
        encoded = self._encode_to_pre_merge(point_features)

        # Modulate with rigidity
        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            if self.qcc_variant == 'cycle_rigidity_side' and hasattr(self, '_side_cyc'):
                modulation = modulation + self.cycle_proj(self._side_cyc)
                self._side_cyc = None
            elif self.qcc_variant == 'cycle_rigidity_mlp' and hasattr(self, '_side_cyc'):
                modulation = modulation + self.cycle_proj_deep(self._side_cyc)
                self._side_cyc = None
            encoded = encoded * (1.0 + modulation)

        # Parts-feature modulation (always on train + eval). No aux loss beyond
        # a tiny entropy collapse penalty returned by the module.
        self._parts_feature_aux_loss = None
        if 'parts_feature' in self.qcc_variants and corr_matched is not None:
            xyz_flat_pf = sampled[..., :3].reshape(batch_size, num_frames * pts_per_frame, 3)
            pf_loss, pf_features = self.parts_feature_module(
                encoded, xyz_flat_pf, num_frames, pts_per_frame, corr_matched,
            )
            self._parts_feature_aux_loss = pf_loss
            if pf_features.shape[1] > 0:
                pf_pooled = pf_features.mean(dim=1).reshape(batch_size, -1)  # (B, K*5)
                pf_mod = self.parts_feature_proj(pf_pooled)                   # (B, hidden2)
                encoded = encoded * (1.0 + pf_mod.unsqueeze(-1))

        # Auxiliary losses
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training:
            total_aux = torch.tensor(0.0, device=encoded.device)
            metrics = {}

            if self.so3_weight > 0:
                R = self._random_rotation_matrix(
                    batch_size, self.rotation_sigma, point_features.device)
                rotated_xyz = torch.bmm(R, point_features[:, :3, :])
                rotated_features = torch.cat(
                    [rotated_xyz, point_features[:, 3:, :]], dim=1)
                with torch.no_grad():
                    encoded_rot = self._encode_to_pre_merge(rotated_features)
                orig_pooled = F.normalize(encoded.mean(dim=-1), dim=-1)
                rot_pooled = F.normalize(encoded_rot.mean(dim=-1), dim=-1)
                so3_loss = F.mse_loss(orig_pooled, rot_pooled.detach())
                total_aux = total_aux + self.so3_weight * so3_loss
                metrics['so3_equiv_raw'] = so3_loss.detach()

            # QCC variant dispatch � iterates over self.qcc_variants so a
            # single config can stack multiple auxiliary losses (e.g.
            # ['prediction', 'grounded_cycle']).
            qcc_total = torch.tensor(0.0, device=encoded.device)
            qcc_any_active = False
            for variant, weight in zip(self.qcc_variants, self.qcc_weights):
                if weight <= 0:
                    continue
                if variant == 'grounded_cycle':
                    qcc_loss, qcc_metrics = self.cycle_module(
                        encoded, num_frames, pts_per_frame,
                        sampled[..., :3],
                    )
                    total_aux = total_aux + weight * qcc_loss
                    qcc_total = qcc_total + weight * qcc_loss.detach()
                    qcc_any_active = True
                    metrics.update(qcc_metrics)
                    metrics[f'qcc_{variant}_raw'] = qcc_loss.detach()
                elif variant in ('grounded_disp', 'grounded_disp_dir', 'grounded_disp_bidir', 'bearing_rot'):
                    if variant == 'grounded_disp':
                        qcc_loss = self.grounded_disp_loss(
                            encoded, sampled[..., :3], num_frames, pts_per_frame)
                    elif variant == 'grounded_disp_dir':
                        qcc_loss = self.grounded_disp_dir_loss(
                            encoded, sampled[..., :3], num_frames, pts_per_frame)
                    elif variant == 'grounded_disp_bidir':
                        qcc_loss = self.grounded_disp_bidir_loss(
                            encoded, sampled[..., :3], num_frames, pts_per_frame)
                    elif variant == 'bearing_rot':
                        # Reshape sampled xyz for _BearingRotationQCCLoss: need (B, F, P, 3)
                        xyz_raw = sampled[..., :3].view(sampled.shape[0], num_frames, pts_per_frame, 3)
                        qcc_loss = self.bearing_rot_qcc_loss(
                            encoded, xyz_raw, num_frames, pts_per_frame)
                    total_aux = total_aux + weight * qcc_loss
                    qcc_total = qcc_total + weight * qcc_loss.detach()
                    qcc_any_active = True
                    metrics[f'qcc_{variant}_raw'] = qcc_loss.detach()
                elif corr_matched is not None:
                    if variant == 'infonce':
                        qcc_loss = self.infonce_loss(
                            encoded, num_frames, pts_per_frame, corr_matched)
                    elif variant == 'prediction':
                        qcc_loss = self.prediction_loss(
                            encoded, num_frames, pts_per_frame, corr_matched)
                    elif variant == 'local_cycle':
                        qcc_loss = self.local_cycle(
                            sampled, num_frames, corr_matched=corr_matched)
                    elif variant == 'displacement':
                        qcc_loss = self.displacement_loss(
                            encoded, sampled, num_frames, pts_per_frame,
                            corr_matched=corr_matched)
                    elif variant == 'parts_feature':
                        qcc_loss = self._parts_feature_aux_loss if self._parts_feature_aux_loss is not None else torch.tensor(0.0, device=encoded.device)
                    elif variant == 'contrastive':
                        qcc_loss = self.corr_contrastive(
                            encoded, num_frames, pts_per_frame, corr_matched)
                    else:
                        qcc_loss = torch.tensor(0.0, device=encoded.device)
                    total_aux = total_aux + weight * qcc_loss
                    qcc_total = qcc_total + weight * qcc_loss.detach()
                    qcc_any_active = True
                    metrics[f'qcc_{variant}_raw'] = qcc_loss.detach()
            if qcc_any_active:
                # Legacy key consumed by main.py logging � sum of all variants.
                metrics['qcc_raw'] = qcc_total

            metrics['qcc_valid_ratio'] = torch.tensor(corr_valid_ratio)
            metrics['rigidity_mean'] = rigidity.mean().detach()
            self.latest_aux_loss = total_aux
            self.latest_aux_metrics = metrics

        encoded = self.merge_proj(self.merge_quaternions(encoded))
        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


# Keep the legacy module alias so older imports still resolve.
REQNNMotion = SimpleLinearMotion

class ResidualOnlyMotion(nn.Module):
    """Pure residual-only classifier using sorted top-K + stat features."""

    def __init__(
        self,
        num_classes=25,
        pts_size=96,
        hidden=256,
        dropout=0.1,
        top_k=32,
        **kwargs,
    ):
        super().__init__()
        self.pts_size = pts_size
        self.top_k = top_k
        self.num_stats = 7                       # mean, std, p50, p75, p90, p95, max
        self.per_pair_dim = top_k + self.num_stats

        c1, c2 = 128, hidden
        self.pair_mlp = nn.Sequential(
            nn.Linear(self.per_pair_dim, c1),
            nn.GELU(),
            nn.Linear(c1, c2),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c2, num_classes)
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    @staticmethod
    def _procrustes_residuals(src, tgt, mask):
        B, P, _ = src.shape
        device = src.device
        w = mask.clamp(min=0.0)
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        src_mean = (src * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum.unsqueeze(-1)
        tgt_mean = (tgt * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum.unsqueeze(-1)
        src_c = src - src_mean
        tgt_c = tgt - tgt_mean
        H = torch.einsum("bp,bpi,bpj->bij", w, src_c, tgt_c)
        H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
        try:
            U, S, Vh = torch.linalg.svd(H)
        except Exception:
            R = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)
            pred = torch.einsum("bij,bpj->bpi", R, src_c)
            return ((pred - tgt_c) ** 2).sum(dim=-1) * mask
        V = Vh.transpose(-1, -2)
        det = torch.det(torch.matmul(V, U.transpose(-1, -2)))
        D_diag = torch.ones(B, 3, device=device)
        D_diag[..., -1] = det
        D_mat = torch.diag_embed(D_diag)
        R = torch.matmul(V, torch.matmul(D_mat, U.transpose(-1, -2)))
        R = R.detach()
        bad = ~torch.isfinite(R).all(dim=-1).all(dim=-1)
        if bad.any():
            R = torch.where(
                bad.unsqueeze(-1).unsqueeze(-1),
                torch.eye(3, device=device).unsqueeze(0).expand_as(R),
                R,
            )
        pred = torch.einsum("bij,bpj->bpi", R, src_c)
        res = ((pred - tgt_c) ** 2).sum(dim=-1)
        return res * mask

    def forward(self, sample):
        pts = sample["points"]
        if pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape
        device = pts.device

        xyz = pts[..., :3].float()
        tgt_idx_fp = sample["corr_full_target_idx"].view(B, F_, P).long()
        w_fp = sample["corr_full_weight"].view(B, F_, P).float()

        per_pair_feats = []
        K = min(self.top_k, P)
        for t in range(F_ - 1):
            src = xyz[:, t]
            tgt = xyz[:, t + 1]
            idx = tgt_idx_fp[:, t]
            valid = (idx >= 0) & (w_fp[:, t] > 0)
            idx_within = torch.where(valid, idx % P, torch.zeros_like(idx))
            tgt_paired = torch.gather(tgt, 1, idx_within.unsqueeze(-1).expand(-1, -1, 3))
            res = self._procrustes_residuals(src, tgt_paired, valid.float())  # (B, P)

            # sorted descending top-K (retains largest articulation signal)
            top_vals, _ = torch.topk(res, K, dim=-1, largest=True)

            # stats over all valid residuals (non-zero after masking)
            valid_f = valid.float().clamp(min=1e-6)
            w_sum = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mean = (res * valid_f).sum(dim=-1, keepdim=True) / w_sum
            var = ((res - mean) ** 2 * valid_f).sum(dim=-1, keepdim=True) / w_sum
            std = var.sqrt()
            # quantiles: sort ascending, take at k=int(qN) after removing invalid
            sorted_res, _ = torch.sort(res, dim=-1)
            p50 = sorted_res[:, P // 2 : P // 2 + 1]
            p75 = sorted_res[:, int(0.75 * P) : int(0.75 * P) + 1]
            p90 = sorted_res[:, int(0.90 * P) : int(0.90 * P) + 1]
            p95 = sorted_res[:, int(0.95 * P) : int(0.95 * P) + 1]
            mx = sorted_res[:, -1:]
            stats = torch.cat([mean, std, p50, p75, p90, p95, mx], dim=-1)  # (B, 7)

            pair_feat = torch.cat([top_vals, stats], dim=-1)  # (B, K+7)
            per_pair_feats.append(pair_feat)

        pair_stack = torch.stack(per_pair_feats, dim=1)                    # (B, F-1, K+7)

        # Per-sample z-normalize the full feature tensor so scale is consistent
        flat = pair_stack.reshape(B, -1)
        mean_n = flat.mean(dim=-1, keepdim=True)
        std_n = flat.std(dim=-1, keepdim=True).clamp(min=1e-6)
        pair_stack = (pair_stack - mean_n.unsqueeze(-1)) / std_n.unsqueeze(-1)

        # Per-pair MLP
        pp = self.pair_mlp(pair_stack)                                      # (B, F-1, c2)
        pp = pp.transpose(1, 2)                                             # (B, c2, F-1)
        fvec = self.temporal(pp).squeeze(-1)                                # (B, c2)
        fvec = self.dropout(fvec)
        return self.classifier(fvec)


class ShortestRotOnlyMotion(nn.Module):
    """Classify gestures using ONLY per-point shortest-rotation quaternions."""

    def __init__(
        self,
        num_classes=25,
        pts_size=96,
        hidden=256,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.pts_size = pts_size
        c1, c2, c3 = 64, 128, hidden
        # Per frame-pair: input is (B, 4, P) — 4 quaternion channels, P points.
        self.point_conv = nn.Sequential(
            nn.Conv1d(4, c1, 1),
            nn.GELU(),
            nn.Conv1d(c1, c2, 1),
            nn.GELU(),
            nn.Conv1d(c2, c3, 1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c3, num_classes)
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    @staticmethod
    def _shortest_rot_quats(src, tgt, mask):
        """src, tgt: (B, P, 3) centered; mask: (B, P) float.
        Returns: (B, P, 4) quat (w, x, y, z). Invalid -> identity (1,0,0,0)."""
        B, P, _ = src.shape
        device = src.device
        dot = (src * tgt).sum(dim=-1)
        sn = src.norm(dim=-1).clamp(min=1e-6)
        tn = tgt.norm(dim=-1).clamp(min=1e-6)
        cos_a = (dot / (sn * tn)).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        angle = torch.acos(cos_a)                                    # (B, P)
        cross = torch.cross(src, tgt, dim=-1)
        cross_norm = cross.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        axis = cross / cross_norm
        half = angle * 0.5
        w = torch.cos(half).unsqueeze(-1)
        xyz = torch.sin(half).unsqueeze(-1) * axis
        quat = torch.cat([w, xyz], dim=-1)
        identity = torch.zeros_like(quat)
        identity[..., 0] = 1.0
        mb = mask.bool().unsqueeze(-1)
        return torch.where(mb, quat, identity)

    def forward(self, sample):
        pts = sample["points"]
        if pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape
        device = pts.device

        xyz = pts[..., :3].float()
        tgt_idx_fp = sample["corr_full_target_idx"].view(B, F_, P).long()
        w_fp = sample["corr_full_weight"].view(B, F_, P).float()

        quat_list = []
        for t in range(F_ - 1):
            src = xyz[:, t]
            tgt = xyz[:, t + 1]
            idx = tgt_idx_fp[:, t]
            valid = (idx >= 0) & (w_fp[:, t] > 0)
            idx_within = torch.where(valid, idx % P, torch.zeros_like(idx))
            tgt_paired = torch.gather(tgt, 1, idx_within.unsqueeze(-1).expand(-1, -1, 3))

            # Center each frame at its weighted centroid (masked).
            mask_f = valid.float()
            w_sum = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            src_mean = (src * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum.unsqueeze(-1)
            tgt_mean = (tgt_paired * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum.unsqueeze(-1)
            src_c = src - src_mean
            tgt_c = tgt_paired - tgt_mean

            q = self._shortest_rot_quats(src_c, tgt_c, mask_f)      # (B, P, 4)
            quat_list.append(q)

        quats = torch.stack(quat_list, dim=1)                       # (B, F-1, P, 4)

        # Per-frame-pair conv over points
        qf = quats.permute(0, 1, 3, 2).reshape(B * (F_ - 1), 4, P)  # (B*(F-1), 4, P)
        f = self.point_conv(qf).squeeze(-1)                         # (B*(F-1), c3)
        f = f.view(B, F_ - 1, -1).transpose(1, 2)                    # (B, c3, F-1)
        f = self.temporal_conv(f).squeeze(-1)                        # (B, c3)
        f = self.dropout(f)
        return self.classifier(f)


class TopsOnlyMotion(nn.Module):
    """Pure "tops field" classifier: per-point per-frame orientation quats."""

    def __init__(
        self,
        num_classes=25,
        pts_size=96,
        hidden=256,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.pts_size = pts_size
        # Per frame: input is (B, 4, P). Convs along points.
        c1, c2, c3 = 64, 128, hidden
        self.point_conv = nn.Sequential(
            nn.Conv1d(4, c1, 1),
            nn.GELU(),
            nn.Conv1d(c1, c2, 1),
            nn.GELU(),
            nn.Conv1d(c2, c3, 1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c3, num_classes)
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    @staticmethod
    def _quat_from_north(direction):
        """direction: (B, P, 3) unit vector. Returns (B, P, 4) quat taking (0,1,0) to direction."""
        B, P, _ = direction.shape
        device = direction.device
        north = torch.zeros_like(direction)
        north[..., 1] = 1.0
        dot = (north * direction).sum(dim=-1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        angle = torch.acos(dot)                              # (B, P)
        cross = torch.cross(north, direction, dim=-1)        # (B, P, 3)
        cross_n = cross.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        axis = cross / cross_n
        half = angle * 0.5
        w = torch.cos(half).unsqueeze(-1)
        xyz = torch.sin(half).unsqueeze(-1) * axis
        return torch.cat([w, xyz], dim=-1)                    # (B, P, 4)

    def forward(self, sample):
        pts = sample["points"]
        if pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape
        device = pts.device

        xyz = pts[..., :3].float()                            # (B, F, P, 3)

        # Weighted centroid per frame (uniform weights since we have all points).
        centroid = xyz.mean(dim=2, keepdim=True)              # (B, F, 1, 3)
        rel = xyz - centroid                                  # (B, F, P, 3)
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = rel / rel_norm                            # (B, F, P, 3)

        # Per-point per-frame quaternion taking north -> direction
        q = self._quat_from_north(direction.view(B * F_, P, 3)).view(B, F_, P, 4)

        # Per-frame conv over points
        qf = q.permute(0, 1, 3, 2).reshape(B * F_, 4, P)      # (B*F, 4, P)
        f = self.point_conv(qf).squeeze(-1)                    # (B*F, c3)
        f = f.view(B, F_, -1).transpose(1, 2)                  # (B, c3, F)
        f = self.temporal_conv(f).squeeze(-1)                  # (B, c3)
        f = self.dropout(f)
        return self.classifier(f)


class LocalNormalOnlyMotion(nn.Module):
    """Classify from local-normal quaternion field only. No XYZ input."""

    def __init__(
        self,
        num_classes=25,
        pts_size=96,
        hidden=256,
        dropout=0.1,
        knn_k=10,
        **kwargs,
    ):
        super().__init__()
        self.pts_size = pts_size
        self.knn_k = knn_k
        c1, c2, c3 = 64, 128, hidden
        self.point_conv = nn.Sequential(
            nn.Conv1d(4, c1, 1),
            nn.GELU(),
            nn.Conv1d(c1, c2, 1),
            nn.GELU(),
            nn.Conv1d(c2, c3, 1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c3, num_classes)
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    @staticmethod
    def _compute_local_normals(xyz, k):
        """xyz: (B, F, P, 3); returns (B, F, P, 3) unit normals, sign-flipped toward centroid."""
        B, F_, P, _ = xyz.shape
        pts = xyz.reshape(B * F_, P, 3)
        dist = torch.cdist(pts, pts)                                 # (B*F, P, P)
        _, idx = torch.topk(dist, k=min(k + 1, P), largest=False, dim=-1)
        idx = idx[:, :, 1:]                                          # drop self
        k_eff = idx.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        pts_exp = pts.unsqueeze(1).expand(-1, P, -1, -1)             # (B*F, P, P, 3)
        neighbors = torch.gather(pts_exp, 2, idx_exp)                # (B*F, P, k, 3)
        nmean = neighbors.mean(dim=2, keepdim=True)
        nc = neighbors - nmean                                        # (B*F, P, k, 3)
        try:
            _, _, Vh = torch.linalg.svd(nc, full_matrices=False)     # Vh: (B*F, P, 3, 3)
        except Exception:
            return torch.zeros_like(pts).reshape(B, F_, P, 3)
        normals = Vh[..., -1, :]                                      # (B*F, P, 3)
        # Sign flip so normal . (centroid - p) > 0 (i.e. points TOWARD centroid).
        centroid = pts.mean(dim=1, keepdim=True)
        to_centroid = centroid - pts
        sign = torch.sign((normals * to_centroid).sum(dim=-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        normals = normals * sign
        # Normalize again just in case.
        nn2 = normals.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        normals = normals / nn2
        return normals.reshape(B, F_, P, 3)

    @staticmethod
    def _quat_from_north(direction):
        B, P, _ = direction.shape
        device = direction.device
        north = torch.zeros_like(direction)
        north[..., 1] = 1.0
        dot = (north * direction).sum(dim=-1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        angle = torch.acos(dot)
        cross = torch.cross(north, direction, dim=-1)
        cross_n = cross.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        axis = cross / cross_n
        half = angle * 0.5
        w = torch.cos(half).unsqueeze(-1)
        xyz = torch.sin(half).unsqueeze(-1) * axis
        return torch.cat([w, xyz], dim=-1)

    def forward(self, sample):
        pts = sample["points"]
        if pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape

        xyz = pts[..., :3].float()
        normals = self._compute_local_normals(xyz, self.knn_k)       # (B, F, P, 3)

        # Detach normals to avoid SVD gradient instability.
        normals = normals.detach()

        q = self._quat_from_north(normals.view(B * F_, P, 3)).view(B, F_, P, 4)

        qf = q.permute(0, 1, 3, 2).reshape(B * F_, 4, P)
        f = self.point_conv(qf).squeeze(-1)
        f = f.view(B, F_, -1).transpose(1, 2)
        f = self.temporal_conv(f).squeeze(-1)
        f = self.dropout(f)
        return self.classifier(f)


class LocalNormalXYZMotion(nn.Module):
    """XYZ coords + local-normal quaternion per point per frame (7 channels)."""

    def __init__(
        self,
        num_classes=25,
        pts_size=96,
        hidden=256,
        dropout=0.1,
        knn_k=10,
        **kwargs,
    ):
        super().__init__()
        self.pts_size = pts_size
        self.knn_k = knn_k
        c1, c2, c3 = 64, 128, hidden
        self.point_conv = nn.Sequential(
            nn.Conv1d(7, c1, 1),
            nn.GELU(),
            nn.Conv1d(c1, c2, 1),
            nn.GELU(),
            nn.Conv1d(c2, c3, 1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c3, num_classes)
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def forward(self, sample):
        pts = sample["points"]
        if pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape

        xyz = pts[..., :3].float()

        # Local-normal via PCA (same as v22a)
        normals = LocalNormalOnlyMotion._compute_local_normals(xyz, self.knn_k).detach()
        q = LocalNormalOnlyMotion._quat_from_north(
            normals.view(B * F_, P, 3)
        ).view(B, F_, P, 4)

        # Concat xyz + quat -> 7 channels per point per frame
        feat = torch.cat([xyz, q], dim=-1)                           # (B, F, P, 7)

        # Per-frame conv over points
        fp = feat.permute(0, 1, 3, 2).reshape(B * F_, 7, P)
        f = self.point_conv(fp).squeeze(-1)
        f = f.view(B, F_, -1).transpose(1, 2)
        f = self.temporal_conv(f).squeeze(-1)
        f = self.dropout(f)
        return self.classifier(f)


class NormalInputMotion(BearingQCCFeatureMotion):
    """BearingQCCFeatureMotion with xyz replaced by per-point local normals."""

    def __init__(self, *args, knn_k_normal=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.knn_k_normal = knn_k_normal

    def extract_features(self, inputs, aux_input=None):
        if isinstance(inputs, dict):
            pts = inputs["points"]
        else:
            pts = inputs
        orig_shape = pts.shape
        if pts.dim() == 4:
            B, F_, P, C = orig_shape
            xyz_full = pts[..., :3].float()
        elif pts.dim() == 3:
            B, N, C = orig_shape
            F_ = N // self.pts_size
            P = self.pts_size
            xyz_full = pts[..., :3].float().view(B, F_, P, 3)
        else:
            return super().extract_features(inputs, aux_input=aux_input)

        # Compute local normals per frame.
        normals = LocalNormalOnlyMotion._compute_local_normals(
            xyz_full, self.knn_k_normal
        ).detach()                                                    # (B, F, P, 3)

        pts_new = pts.clone()
        if pts.dim() == 4:
            pts_new[..., :3] = normals
        else:
            pts_new[..., :3] = normals.view(B, N, 3)

        if isinstance(inputs, dict):
            new_inputs = dict(inputs)
            new_inputs["points"] = pts_new
        else:
            new_inputs = pts_new

        return super().extract_features(new_inputs, aux_input=aux_input)


class TopsInputMotion(BearingQCCFeatureMotion):
    """BearingQCCFeatureMotion with xyz replaced by unit direction from frame centroid."""

    def extract_features(self, inputs, aux_input=None):
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

        centroid = xyz_full.mean(dim=2, keepdim=True)                 # (B, F, 1, 3)
        rel = xyz_full - centroid                                     # (B, F, P, 3)
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = (rel / rel_norm).detach()                         # (B, F, P, 3)

        pts_new = pts.clone()
        if pts.dim() == 4:
            pts_new[..., :3] = direction
        else:
            pts_new[..., :3] = direction.view(B, N, 3)

        if isinstance(inputs, dict):
            new_inputs = dict(inputs)
            new_inputs["points"] = pts_new
        else:
            new_inputs = pts_new

        return super().extract_features(new_inputs, aux_input=aux_input)


class TopsXYZInputMotion(BearingQCCFeatureMotion):
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

        # Tops direction from frame centroid, computed on sampled xyz.
        xyz = sampled[..., :3]
        centroid = xyz.mean(dim=2, keepdim=True)
        rel = xyz - centroid
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        tops = (rel / rel_norm).detach()                          # (B, F, P, 3)
        sampled_7 = torch.cat([sampled, tops], dim=-1)             # (B, F, P, 7)

        # Rigidity from xyz+time only.
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

        encoded = self.merge_proj(self.merge_quaternions(encoded))
        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)


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

