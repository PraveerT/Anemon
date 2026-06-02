import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, out_channel, dimension, with_bn=True, activation='gelu'):
        super(MLPBlock, self).__init__()
        self.layer_list = []
        
        # Select activation function
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)
        
        if dimension == 1:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                            nn.BatchNorm1d(out_channel[idx + 1]),
                            act_fn,
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        )
                    )
        elif dimension == 2:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                            nn.BatchNorm2d(out_channel[idx + 1]),
                            act_fn,
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        )
                    )
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, output):
        for layer in self.layer_list:
            output = layer(output)
        return output


class MotionBlock(nn.Module):
    def __init__(self, out_channel, dimension, embedding_dim):
        super(MotionBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.layer_list = []
        if dimension == 1:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, out_channel[-1], kernel_size=1),
                    nn.BatchNorm1d(out_channel[-1]),
                    nn.GELU(),
                )
            )
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        nn.BatchNorm1d(out_channel[idx + 1]),
                        nn.GELU(),
                    )
                )
        elif dimension == 2:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv2d(embedding_dim, out_channel[-1], kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channel[-1]),
                    nn.GELU(),
                )
            )
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        nn.BatchNorm2d(out_channel[idx + 1]),
                        nn.GELU(),
                    )
                )
        self.layer_list = nn.ModuleList(self.layer_list)
        
        # Learnable fusion gate for better position-feature interaction
        self.fusion_gate = nn.Parameter(torch.ones(1) * 0.5)
        
        # Lightweight channel attention
        if len(out_channel) > 1:
            final_channels = out_channel[-1]
            if dimension == 2:
                self.channel_attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(final_channels, final_channels // 4, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(final_channels // 4, final_channels, kernel_size=1),
                    nn.Sigmoid()
                )
            else:
                self.channel_attention = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(final_channels, final_channels // 4, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(final_channels // 4, final_channels, kernel_size=1),
                    nn.Sigmoid()
                )
        else:
            self.channel_attention = None

    def forward(self, output):
        position_embedding = self.layer_list[0](output[:, :self.embedding_dim])
        feature_embedding = output[:, self.embedding_dim:]
        for layer in self.layer_list[1:]:
            feature_embedding = layer(feature_embedding)
        
        # Apply channel attention to position embedding
        if self.channel_attention is not None:
            attention_weights = self.channel_attention(position_embedding)
            position_embedding = position_embedding * attention_weights
        
        # Enhanced fusion with learnable gating
        # α * (pos * feat) + (1-α) * (pos + feat)
        multiplicative = position_embedding * feature_embedding
        additive = position_embedding + feature_embedding
        output = self.fusion_gate * multiplicative + (1 - self.fusion_gate) * additive
        
        return output

class GroupOperation(object):
    def __init__(self, normalize_offsets=True, eps=1e-8):
        """
        GroupOperation with normalized spatial offsets for neighborhood features.
        
        Args:
            normalize_offsets: Whether to normalize spatial offsets
            eps: Small value for numerical stability
        """
        self.normalize_offsets = normalize_offsets
        self.eps = eps

    def group_points(self, distance_dim, array1, array2, knn, dim):
        """
        Group points based on k-nearest neighbors with improved memory efficiency.
        """
        # Use the array_distance method for all cases to ensure compatibility
        matrix, a2 = self.array_distance(array1, array2, distance_dim, dim)
        
        # Get k-nearest neighbors
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        
        # Gather neighbors
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        
        # Compute offsets
        offsets = array1.unsqueeze(dim + 1) - neighbor
        
        # Normalize spatial offsets with improved numerical stability
        if self.normalize_offsets:
            spatial_offsets = offsets[:, :3]
            # Use torch.norm for better stability
            norm = torch.norm(spatial_offsets, dim=1, keepdim=True) + self.eps
            # Avoid in-place operation by creating new tensor
            normalized_offsets = offsets.clone()
            normalized_offsets[:, :3] = spatial_offsets / norm
            offsets = normalized_offsets
            
        return offsets

    def st_group_points(self, array, interval, distance_dim, knn, dim, coord_dim=4):
        """
        Spatio-temporal grouping with optimized batch processing.
        """
        batchsize, channels, timestep, num_pts = array.shape
        
        # Use original padding method for compatibility
        if interval // 2 > 0:
            array_padded = torch.cat((array[:, :, 0].unsqueeze(2).expand(-1, -1, interval // 2, -1),
                                      array,
                                      array[:, :, -1].unsqueeze(2).expand(-1, -1, interval // 2, -1)
                                      ), dim=2)
        else:
            array_padded = array
        
        # Vectorize temporal window extraction instead of building it timestep-by-timestep.
        windows = array_padded.unfold(2, interval, 1)
        neighbor_points = windows.permute(0, 1, 2, 4, 3).reshape(
            batchsize, channels, timestep, num_pts * interval
        )
        
        # Use array_distance for compatibility
        matrix, a2 = self.array_distance(array, neighbor_points, distance_dim, dim)
        
        # Get k-nearest neighbors
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        
        # Gather neighbors using original method for compatibility  
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).
                             expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        
        # Compute features with edge attributes
        array_expanded = array.unsqueeze(-1).expand_as(neighbor)
        
        # Enhanced feature computation with relative positions and features
        position_diff = array_expanded[:, :coord_dim] - neighbor[:, :coord_dim]

        # Option to include distance as additional feature
        if channels > coord_dim:
            ret_features = torch.cat([
                position_diff,                  # Relative positions
                array_expanded[:, coord_dim:],  # Source features
                neighbor[:, coord_dim:]         # Neighbor features
            ], dim=1)
        else:
            ret_features = position_diff
            
        return ret_features

    def array_distance(self, array1, array2, dist, dim):
        """
        Legacy method kept for compatibility.
        """
        distance_mat = array1.unsqueeze(dim + 1)[:, dist] - array2.unsqueeze(dim)[:, dist]
        mat_shape = distance_mat.shape
        mat_shape = mat_shape[:1] + (array1.shape[1],) + mat_shape[2:]
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat ** 2).sum(1) + self.eps)
        return distance_mat, array2
