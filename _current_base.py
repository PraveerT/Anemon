import torch
import torch.nn as nn
import torch.nn.functional as F
from models.op import MLPBlock, MotionBlock, GroupOperation

# NOTE: stage-3 'bottleneck' encoder is a per-frame MLP (no Mamba/SSM).


class MultiScaleFeatureProcessor(nn.Module):
    """Multi-scale feature processing layer that creates diverse representations
    at different temporal scales and combines them effectively."""
    
    def __init__(self, in_channels, num_scales=4, feature_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # Multi-scale filters for temporal feature extraction
        self.scale_filters = nn.ModuleList([
            nn.Conv2d(in_channels, feature_dim, kernel_size=(2**i, 1), 
                     stride=(2**i, 1), padding=(2**(i-1), 0))
            for i in range(1, num_scales + 1)
        ])
        
        # Feature interaction network between scales
        self.scale_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim * 2, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Conv2d(feature_dim, feature_dim, 1)
            ) for _ in range(num_scales - 1)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(feature_dim * num_scales + in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
    
    def forward(self, x):
        # x shape: B, C, T, N
        B, _, T, N = x.shape
        
        # 1. Multi-scale feature extraction
        scale_features = [scale_filter(x) for scale_filter in self.scale_filters]
        
        # 2. Model feature interaction between scales
        interacted_features = [scale_features[0]]
        for i in range(len(scale_features) - 1):
            # Interpolate features for interaction
            source = F.interpolate(scale_features[i], size=(scale_features[i+1].shape[2], N), 
                                 mode='bilinear', align_corners=False)
            target = scale_features[i + 1]
            
            # Combine source and target
            combined = torch.cat([source, target], dim=1)
            
            # Model feature interaction
            interaction = self.scale_interaction[i](combined)
            interacted_features.append(target + interaction)
        
        # 3. Upsample each scale back to the original resolution
        all_features = [
            F.interpolate(feat, size=(T, N), mode='bilinear', align_corners=False)
            for feat in interacted_features
        ]

        # 4. Combine all feature representations
        combined_features = torch.cat(all_features, dim=1)  # (B, feature_dim * num_scales, T, N)
        
        # 5. Output projection with residual
        combined = torch.cat([x, combined_features], dim=1)
        output = self.output_proj(combined)
        
        return output + x



def _remap_legacy_bottleneck_keys(state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
    """Back-compat: old checkpoints stored the stage-3 encoder under 'mamba.*'.
    The attribute is now 'bottleneck.*'; remap on load so legacy ckpts still fit."""
    legacy = prefix + 'mamba.'
    new = prefix + 'bottleneck.'
    for k in [k for k in state_dict if k.startswith(legacy)]:
        state_dict[new + k[len(legacy):]] = state_dict.pop(k)


class _DetMaxPool(nn.Module):
    """Deterministic AdaptiveMaxPool2d replacement (out (None,1) or (1,1)).
    adaptive_max_pool2d_backward_cuda is nondeterministic; max/amax are not.
    Forward is identical (max is max); no params, checkpoints unaffected."""
    def __init__(self, out_size):
        super().__init__()
        self.out_size = tuple(out_size)
    def forward(self, x):
        if self.out_size == (None, 1):
            return x.max(dim=3, keepdim=True).values
        if self.out_size == (1, 1):
            return x.amax(dim=(2, 3), keepdim=True)
        raise ValueError(f'unsupported {self.out_size}')


class Base(nn.Module):
    def __init__(self, num_classes, pts_size, topk=16, downsample=(2, 2, 2),
                 knn=(16, 48, 48, 24), coord_channels=4, multi_scale_num_scales=4,
                 skip_stage3=False, multi_scale_mode='pyramid'):
        super(Base, self).__init__()
        self._register_load_state_dict_pre_hook(_remap_legacy_bottleneck_keys)
        self.coord_channels = coord_channels
        self.skip_stage3 = skip_stage3
        self.stage1 = MLPBlock([coord_channels, 32, 64], 2)
        self.pool1 = _DetMaxPool((None, 1))
        self.stage2 = MotionBlock([128, 128, ], 2, coord_channels)
        self.pool2 = _DetMaxPool((None, 1))
        self.stage3 = MotionBlock([256, 256, ], 2, coord_channels)
        self.pool3 = _DetMaxPool((None, 1))
        # Stage 4 removed to reduce overfitting and improve efficiency
        self.stage5 = MLPBlock([256 + coord_channels, 1024], 2)  # Updated from 512 to 260 (fea3 channels)
        self.pool5 = _DetMaxPool((1, 1))
        self.stage6 = MLPBlock([1024, num_classes], 2, with_bn=False)
        self.global_bn = nn.BatchNorm2d(1024)
        self.knn = knn
        self.pts_size = pts_size
        self.downsample = downsample
        self.num_classes = num_classes
        self.group = GroupOperation()
        # Stage-3 channel bottleneck encoder (NO temporal mixing; legacy name
        # was 'mamba' but no Mamba/SSM is used). Subclasses replace this Identity
        # with PGCNetEncoder, a per-frame bidirectional MLP.
        self.bottleneck = nn.Identity()
        self.bottleneck.in_channels = 256
        self.bottleneck.output_dim = 256
        
        # Add Multi-scale Feature Processor layer after stage2
        _ms_in = (coord_channels + 64) * 2 - coord_channels
        if multi_scale_mode == 'plain':
            _h = 1180  # param-matched (~315K) pointwise channel-mixer, NO temporal kernels
            self.multi_scale = nn.Sequential(
                nn.Conv2d(_ms_in, _h, 1), nn.BatchNorm2d(_h), nn.ReLU(inplace=True),
                nn.Conv2d(_h, _ms_in, 1))
        else:
            self.multi_scale = MultiScaleFeatureProcessor(in_channels=_ms_in, num_scales=multi_scale_num_scales, feature_dim=32)
        self.feature_dim = 1024

    def _sample_points(self, inputs):
        points = inputs.permute(0, 3, 1, 2)
        point_count = points.shape[3]
        device = points.device
        sample_size = min(self.pts_size, point_count)

        if self.training:
            # Random sampling during training for augmentation
            indices = torch.randperm(point_count, device=device)[:sample_size]
        else:
            # Deterministic sampling during testing for consistent results
            indices = torch.linspace(0, point_count - 1, sample_size, device=device).long()
        points = points[:, :, :, indices]
        return points[:, :self.coord_channels]

    def _encode_sampled_points(self, coords):
        batchsize, in_dims, timestep, pts_num = coords.shape

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2], array1=coords, array2=coords, knn=self.knn[0],
                                             dim=3)
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        # stage 2: inter-frame, early
        in_dims = fea1.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[0]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3, coord_dim=self.coord_channels)
        ret_array2, coords = self.select_ind(ret_group_array2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        # stage 3: inter-frame, middle (optional). When skipped, fea2's feature
        # channels feed the bottleneck directly -- the net goes one level shallower
        # and keeps N/2 points (no stage-3 downsample). Spatial-depth ablation.
        if self.skip_stage3:
            fea3 = fea2[:, self.coord_channels:]           # drop coords -> 128 feat ch
        else:
            in_dims = fea2.shape[1] * 2 - self.coord_channels
            pts_num //= self.downsample[1]
            ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3, coord_dim=self.coord_channels)
            ret_array3, coords = self.select_ind(ret_group_array3, coords, batchsize, in_dims, timestep, pts_num)
            fea3 = self.pool3(self.stage3(ret_array3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_enc = self.bottleneck(fea3)
        return torch.cat((coords, fea3_enc), dim=1)

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)
        fea3 = self._encode_sampled_points(coords)
        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

    def classify_features(self, features):
        logits = self.stage6(features.unsqueeze(-1).unsqueeze(-1))
        return logits.view(features.shape[0], self.num_classes)

    def forward(self, inputs):
        features = self.extract_features(inputs)
        return self.classify_features(features)

    def select_ind(self, group_array, inputs, batchsize, in_dim, timestep, pts_num):
        """
        Select indices and apply them to group_array and inputs tensors.
        
        Args:
            group_array: Tensor of shape (B, C, T*P, K) - grouped points
            inputs: Tensor of shape (B, C, T, P) - input points
            batchsize: Batch size
            in_dim: Input dimension
            timestep: Number of timesteps
            pts_num: Number of points to select
            
        Returns:
            ret_group_array: Selected grouped points
            inputs: Selected input points
        """
        # Validate inputs
        if pts_num <= 0:
            raise ValueError("pts_num must be positive")
        
        # Select indices based on point weights
        ind = self.weight_select(group_array, pts_num)
        
        # Apply indices to group_array
        # Optimize tensor operations by precomputing shapes
        ind_expanded = ind.unsqueeze(1).unsqueeze(-1).expand(
            -1, group_array.shape[1], -1, -1, group_array.shape[-1])
        ret_group_array = group_array.gather(-2, ind_expanded)
        ret_group_array = ret_group_array.reshape(batchsize, in_dim, timestep * pts_num, -1)
        
        # Apply indices to inputs
        inputs = inputs.gather(-1, ind.unsqueeze(1).expand(-1, inputs.shape[1], -1, -1))
        
        return ret_group_array, inputs

    @staticmethod
    def _normalize_scores(values):
        values_min = values.min(dim=-1, keepdim=True)[0]
        values_max = values.max(dim=-1, keepdim=True)[0]
        values_range = values_max - values_min
        values_range = torch.where(values_range == 0, torch.ones_like(values_range), values_range)
        return (values - values_min) / values_range

    @staticmethod
    def weight_select(position, topk):
        """
        Select points with larger ranges based on a hybrid metric combining distance and variance.
        
        This function computes a weighted score for each point based on:
        1. Distance from origin (encourages selecting distant points)
        2. Feature variance (encourages selecting points with high variation)
        3. Spatial coverage (encourages selecting spatially diverse points)
        
        Args:
            position: Tensor of shape (B, C, T*P, K) where first 3 channels are x,y,z coordinates
            topk: Number of points to select
            
        Returns:
            idx: Indices of selected points
        """
        # Validate inputs
        if topk <= 0:
            raise ValueError("topk must be positive")
        if position.shape[1] < 3:
            raise ValueError("position tensor must have at least 3 channels for x,y,z coordinates")
            
        # Compute squared Euclidean distances for first 3 dimensions (x,y,z)
        # position[:, :3] selects x,y,z coordinates
        # **2 computes squared distances
        # sum(dim=1) sums across x,y,z dimensions -> (B, T*P, K)
        # max(dim=-1)[0] takes maximum across K neighbors -> (B, T*P)
        distances = torch.max(torch.sum(position[:, :3] ** 2, dim=1), dim=-1)[0]
        
        # Normalize distances to [0, 1] range
        normalized_distances = Base._normalize_scores(distances)
        
        # Compute feature variance across neighbors if we have more than 3 channels
        if position.shape[1] > 3:
            # Compute variance for feature channels (channels 3 onwards)
            feature_var = torch.var(position[:, 3:], dim=-1).mean(dim=1)  # Mean variance across time
            # Normalize feature variance
            normalized_variance = Base._normalize_scores(feature_var)
        else:
            # If no feature channels, use zeros
            normalized_variance = torch.zeros_like(normalized_distances)
        
        # Compute spatial coverage metric to encourage diversity
        # Points that are spatially isolated from other selected points are preferred
        # Simplified approach: use distance to centroid of all points as diversity measure
        if position.shape[2] > 1:  # If we have more than one point
            # Extract spatial coordinates of centroids (first neighbor for each point)
            # The shape is (B, 3, T*P) - we need to be careful with dimensions
            coords = position[:, :3, :, 0]  # (B, 3, T*P)
            
            # Compute centroid of all points for each batch
            # We need to compute centroid across the T*P dimension (dim=2)
            centroid = torch.mean(coords, dim=2, keepdim=True)  # (B, 3, 1)
            
            # Compute distance of each point to the centroid
            # Points farther from centroid are more diverse
            # coords: (B, 3, T*P), centroid: (B, 3, 1)
            diversity_measure = torch.sqrt(torch.sum((coords - centroid) ** 2, dim=1))  # (B, T*P)
            
            # Normalize diversity measure
            normalized_diversity = Base._normalize_scores(diversity_measure)
        else:
            normalized_diversity = torch.zeros_like(normalized_distances)
        
        # Ensure all metrics have the same shape
        # All should be (B, T*P)
        if not (normalized_distances.shape == normalized_variance.shape == normalized_diversity.shape):
            # If there's a shape mismatch, fall back to just distance + variance
            weights = 0.7 * normalized_distances + 0.3 * normalized_variance
        else:
            # Combine metrics with weighted sum
            # Distance: 0.4 (coverage is still important)
            # Variance: 0.3 (as you added)
            # Spatial diversity: 0.3 (diverse point selection)
            weights = 0.4 * normalized_distances + 0.3 * normalized_variance + 0.3 * normalized_diversity
        
        # Select topk points with largest combined weights
        # Using sorted=False for better performance when order doesn't matter
        _, idx = torch.topk(weights, min(topk, weights.shape[-1]), -1, largest=True, sorted=False)
        return idx


if __name__ == '__main__':
    pass
