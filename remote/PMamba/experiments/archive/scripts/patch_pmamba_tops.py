"""PMamba with xyz + tops field as 7-channel input.

Subclasses models.motion.Motion:
  - rebuilds stage1 for 7 input channels
  - rebuilds multi_scale for 135 in_channels (was 132)
  - rebuilds stage5 for 263 input (was 260)
  - overrides _sample_points to compute tops and append
  - overrides _encode_sampled_points to use `-7` instead of `-4` in in_dims formulas

Adds class PMambaTopsMotion to models/motion.py.
"""
from pathlib import Path

PATH = Path("models/motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class PMambaTopsMotion(Motion):
    """PMamba with per-point centroid-radial direction concatenated (7-ch input)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rebuild input-sensitive layers for 7 spatial channels.
        self.stage1 = MLPBlock([7, 32, 64], 2)
        self.multi_scale = MultiScaleFeatureProcessor(in_channels=135, num_scales=4, feature_dim=32)
        self.stage5 = MLPBlock([263, 1024], 2)

    def _sample_points(self, inputs):
        points = inputs.permute(0, 3, 1, 2)                            # (B, C, T, N)
        point_count = points.shape[3]
        device = points.device
        sample_size = min(self.pts_size, point_count)

        if self.training:
            indices = torch.randperm(point_count, device=device)[:sample_size]
        else:
            indices = torch.linspace(0, point_count - 1, sample_size, device=device).long()
        points = points[:, :, :, indices]
        sampled = points[:, :4]                                         # (B, 4, T, P)

        # Compute tops: per-point unit direction from per-frame centroid.
        xyz = sampled[:, :3]                                            # (B, 3, T, P)
        centroid = xyz.mean(dim=-1, keepdim=True)                       # (B, 3, T, 1)
        rel = xyz - centroid
        rel_norm = rel.norm(dim=1, keepdim=True).clamp(min=1e-6)        # (B, 1, T, P)
        tops = (rel / rel_norm).detach()                                # (B, 3, T, P)

        return torch.cat([sampled, tops], dim=1)                        # (B, 7, T, P)

    def _encode_sampled_points(self, coords):
        batchsize, in_dims, timestep, pts_num = coords.shape
        assert in_dims == 7, f"PMambaTopsMotion expects 7-ch coords, got {in_dims}"

        # stage 1: intra-frame
        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        # stage 2: inter-frame, early. in_dims = fea1.shape[1] * 2 - 7 (was -4 in parent)
        in_dims_s2 = fea1.shape[1] * 2 - 7
        pts_num_s2 = pts_num // self.downsample[0]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret_array2, coords = self.select_ind(
            ret_group_array2, coords, batchsize, in_dims_s2, timestep, pts_num_s2,
        )
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(batchsize, -1, timestep, pts_num_s2)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        # stage 3: inter-frame, middle, applying mamba. in_dims = fea2.shape[1] * 2 - 7
        in_dims_s3 = fea2.shape[1] * 2 - 7
        pts_num_s3 = pts_num_s2 // self.downsample[1]
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret_array3, coords = self.select_ind(
            ret_group_array3, coords, batchsize, in_dims_s3, timestep, pts_num_s3,
        )
        fea3 = self.pool3(self.stage3(ret_array3)).reshape(batchsize, -1, timestep, pts_num_s3)
        fea3_mamba = self.mamba(fea3)
        return torch.cat((coords, fea3_mamba), dim=1)                    # coords=7, fea3_mamba=256 -> 263

'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added PMambaTopsMotion")
