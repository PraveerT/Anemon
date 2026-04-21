"""v3: tops only feeds stage1; downstream stages use xyz+time only so
MotionBlock's hardcoded [:, :4] position split remains valid."""
import re
from pathlib import Path

PATH = Path("models/motion.py")
src = PATH.read_text(encoding="utf-8")

pat = re.compile(r"class PMambaTopsMotion\(Motion\):.*?(?=\nclass |\Z)", re.DOTALL)

new_class = '''class PMambaTopsMotion(Motion):
    """PMamba with tops field fed only into stage1. Downstream coords stay xyz+time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only stage1 changes: accepts 7-channel input.
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _sample_points(self, inputs):
        points = inputs.permute(0, 3, 1, 2)
        point_count = points.shape[3]
        device = points.device
        sample_size = min(self.pts_size, point_count)
        if self.training:
            indices = torch.randperm(point_count, device=device)[:sample_size]
        else:
            indices = torch.linspace(0, point_count - 1, sample_size, device=device).long()
        points = points[:, :, :, indices]
        sampled = points[:, :4]                                          # (B, 4, T, P)

        xyz = sampled[:, :3]
        centroid = xyz.mean(dim=-1, keepdim=True)
        rel = xyz - centroid
        rel_norm = rel.norm(dim=1, keepdim=True).clamp(min=1e-6)
        tops = (rel / rel_norm).detach()                                 # (B, 3, T, P)
        return torch.cat([sampled, tops], dim=1)                         # (B, 7, T, P)

    def _encode_sampled_points(self, coords7):
        # coords7: (B, 7, T, P). Preserve xyz+time-only for downstream usage.
        batchsize, _, timestep, pts_num = coords7.shape
        coords = coords7[:, :4]                                          # (B, 4, T, P)

        # stage 1: intra-frame, uses full 7-channel coords7.
        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)                          # (B, 4 + 64, T, P) = (B, 68, T, P)

        # stage 2 onward: unchanged from parent (coords is xyz+time, 4 channels).
        in_dims = fea1.shape[1] * 2 - 4
        pts_num_s2 = pts_num // self.downsample[0]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret_array2, coords = self.select_ind(
            ret_group_array2, coords, batchsize, in_dims, timestep, pts_num_s2,
        )
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(batchsize, -1, timestep, pts_num_s2)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num_s3 = pts_num_s2 // self.downsample[1]
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret_array3, coords = self.select_ind(
            ret_group_array3, coords, batchsize, in_dims, timestep, pts_num_s3,
        )
        fea3 = self.pool3(self.stage3(ret_array3)).reshape(batchsize, -1, timestep, pts_num_s3)
        fea3_mamba = self.mamba(fea3)
        return torch.cat((coords, fea3_mamba), dim=1)
'''

src = pat.sub(new_class, src)
PATH.write_text(src, encoding="utf-8")
print("v3 patched")
