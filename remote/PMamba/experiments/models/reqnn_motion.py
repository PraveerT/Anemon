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

    Splits encoded features into 3 temporal segments, estimates quaternion
    rotations between pairs.  Each quaternion is grounded by a reconstruction
    loss: rotating the source segment's pooled features should match the
    target segment.  The cycle constraint (q_12 * q_23 * q_31 = identity)
    adds mutual consistency on top, which is the novel signal.
    """

    def __init__(self, feat_dim):
        super().__init__()
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
        feat_dim = encoded.shape[1]

        seg_size = num_frames // 3

        # Per-point XYZ per segment: (batch, seg_size * pts, 3)
        xyz1 = points_xyz[:, :seg_size].reshape(batch, -1, 3)
        xyz2 = points_xyz[:, seg_size:2*seg_size].reshape(batch, -1, 3)
        xyz3 = points_xyz[:, 2*seg_size:3*seg_size].reshape(batch, -1, 3)

        # Pool encoded features per segment for quaternion estimation
        feat = encoded.permute(0, 2, 1).reshape(
            batch, num_frames, pts_per_frame, feat_dim,
        )
        seg1 = feat[:, :seg_size].reshape(batch, -1, feat_dim).mean(dim=1)
        seg2 = feat[:, seg_size:2*seg_size].reshape(batch, -1, feat_dim).mean(dim=1)
        seg3 = feat[:, 2*seg_size:3*seg_size].reshape(batch, -1, feat_dim).mean(dim=1)

        # Estimate quaternion rotations between segment pairs
        q_12 = self._estimate_quaternion(seg1, seg2.detach())
        q_23 = self._estimate_quaternion(seg2, seg3.detach())
        q_31 = self._estimate_quaternion(seg3, seg1.detach())

        # Reconstruction loss on per-point XYZ (centered per segment)
        # The quaternion must rotate source point cloud to match target
        recon_loss = torch.tensor(0.0, device=encoded.device)
        for src_xyz, tgt_xyz, q in [
            (xyz1, xyz2, q_12), (xyz2, xyz3, q_23), (xyz3, xyz1, q_31),
        ]:
            src_c = src_xyz - src_xyz.mean(dim=1, keepdim=True)
            tgt_c = tgt_xyz - tgt_xyz.mean(dim=1, keepdim=True)
            # q is (batch, 4), broadcast over points
            rotated = _quaternion_rotate_vector(
                q.unsqueeze(1).expand(-1, src_c.shape[1], -1), src_c,
            )
            recon_loss = recon_loss + F.mse_loss(rotated, tgt_c.detach())
        recon_loss = recon_loss / 3.0

        # Cycle loss: q_12 * q_23 * q_31 should compose to identity
        q_cycle = _hamilton_product(_hamilton_product(q_12, q_23), q_31)
        q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=encoded.device)
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
        self.qcc_weight = qcc_weight
        self.qcc_variant = qcc_variant
        self.rigidity_scales = rigidity_scales
        self.disable_rigidity = disable_rigidity
        self.decouple_sampling = decouple_sampling
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

        rig_channels = len(rigidity_scales) if qcc_variant == 'multiscale' else 1
        self.rigidity_proj = nn.Sequential(
            nn.Conv1d(rig_channels, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.rigidity_proj[0].weight)
        nn.init.zeros_(self.rigidity_proj[0].bias)

        self.corr_contrastive = _CorrespondenceContrastiveLoss()
        self.infonce_loss = _InfoNCETemporalLoss(temperature=0.1)
        self.prediction_loss = _TemporalPredictionLoss(feat_dim=hidden2)
        self.local_cycle = _LocalCycleConsistencyLoss()
        self.displacement_loss = _DisplacementAgreementLoss(knn_k=bearing_knn_k)
        # Grounded cycle consistency (used by qcc_variant='grounded_cycle',
        # the original 80.29% baseline loss)
        self.cycle_module = _GroundedCycleConsistency(feat_dim=hidden2)

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
        else:
            rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)

        point_features = sampled.reshape(batch_size, -1, 4).transpose(1, 2).contiguous()
        encoded = self._encode_to_pre_merge(point_features)

        # Modulate with rigidity
        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            encoded = encoded * (1.0 + modulation)

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

            # QCC variant dispatch
            if self.qcc_weight > 0:
                if self.qcc_variant == 'grounded_cycle':
                    # Original 80.29% baseline loss: pooled-segment quaternion
                    # estimation grounded by XYZ reconstruction + cycle constraint.
                    # Does not require correspondence data.
                    qcc_loss, qcc_metrics = self.cycle_module(
                        encoded, num_frames, pts_per_frame,
                        sampled[..., :3],
                    )
                    total_aux = total_aux + self.qcc_weight * qcc_loss
                    metrics.update(qcc_metrics)
                    metrics['qcc_raw'] = qcc_loss.detach()
                elif corr_matched is not None:
                    if self.qcc_variant == 'infonce':
                        qcc_loss = self.infonce_loss(
                            encoded, num_frames, pts_per_frame, corr_matched)
                    elif self.qcc_variant == 'prediction':
                        qcc_loss = self.prediction_loss(
                            encoded, num_frames, pts_per_frame, corr_matched)
                    elif self.qcc_variant == 'local_cycle':
                        qcc_loss = self.local_cycle(
                            sampled, num_frames, corr_matched=corr_matched)
                    elif self.qcc_variant == 'displacement':
                        qcc_loss = self.displacement_loss(
                            encoded, sampled, num_frames, pts_per_frame,
                            corr_matched=corr_matched)
                    elif self.qcc_variant == 'contrastive':
                        qcc_loss = self.corr_contrastive(
                            encoded, num_frames, pts_per_frame, corr_matched)
                    else:
                        qcc_loss = torch.tensor(0.0, device=encoded.device)
                    total_aux = total_aux + self.qcc_weight * qcc_loss
                    metrics['qcc_raw'] = qcc_loss.detach()

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
