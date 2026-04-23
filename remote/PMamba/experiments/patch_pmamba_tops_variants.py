"""Add 4 tops variants to models/motion.py.

- MotionTopsMag (v3a): 8-ch [xyz, tops(3), |rel|, t]. Adds magnitude.
- MotionTopsClip (v3b): 7-ch. Tops from clip-mean centroid (not per-frame).
- MotionDTops (v3c): 7-ch. dtops = tops(t+1)-tops(t) per-point, pad last.
- MotionTopsFull (v3d): 10-ch [xyz, tops, dtops, t]. Shape + motion.

All extend Motion, replace stage1 for matching input channels, compute
custom tops inside _encode_sampled_points before stage1. Stages 2-3 unchanged.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")

additions = [
    ("MotionTopsMag", 8),
    ("MotionTopsClip", 7),
    ("MotionDTops", 7),
    ("MotionTopsFull", 10),
]

existing = [name for name, _ in additions if f"class {name}" in src]
if len(existing) == len(additions):
    print("all 4 already present")
else:
    snippet = '''

class MotionTopsMag(Motion):
    """PMamba + tops + |rel|. 8-ch [xyz(3), tops(3), |rel|, t]."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.stage1 = MLPBlock([8, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, _, timestep, pts_num = coords.shape
        xyz = coords[:, :3]
        centroid = xyz.mean(dim=-1, keepdim=True)
        rel = xyz - centroid
        mag = rel.norm(dim=1, keepdim=True).clamp(min=1e-6)
        tops = rel / mag
        coords8 = torch.cat([xyz, tops, mag, coords[:, 3:4]], dim=1)

        ret = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords8, array2=coords8,
            knn=self.knn[0], dim=3,
        )
        ret = ret.reshape(batchsize, 8, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret)).reshape(
            batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        return torch.cat((coords, self.mamba(fea3)), dim=1)


class MotionTopsClip(Motion):
    """PMamba + tops from CLIP-mean centroid (stable reference)."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, _, timestep, pts_num = coords.shape
        xyz = coords[:, :3]
        # Mean across points AND frames -> single clip-level centroid per-sample.
        clip_c = xyz.mean(dim=[2, 3], keepdim=True)       # (B,3,1,1)
        rel = xyz - clip_c
        tops = rel / rel.norm(dim=1, keepdim=True).clamp(min=1e-6)
        coords7 = torch.cat([xyz, tops, coords[:, 3:4]], dim=1)

        ret = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret = ret.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret)).reshape(
            batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        return torch.cat((coords, self.mamba(fea3)), dim=1)


class MotionDTops(Motion):
    """PMamba + Δtops (frame-to-frame tops delta). Captures angular velocity."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, _, timestep, pts_num = coords.shape
        xyz = coords[:, :3]
        centroid = xyz.mean(dim=-1, keepdim=True)
        rel = xyz - centroid
        tops = rel / rel.norm(dim=1, keepdim=True).clamp(min=1e-6)  # (B,3,T,P)
        # Δtops: forward diff along T, pad last with zero (no motion info).
        dtops = torch.zeros_like(tops)
        dtops[:, :, :-1] = tops[:, :, 1:] - tops[:, :, :-1]
        coords7 = torch.cat([xyz, dtops, coords[:, 3:4]], dim=1)

        ret = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret = ret.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret)).reshape(
            batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        return torch.cat((coords, self.mamba(fea3)), dim=1)


class MotionTopsFull(Motion):
    """PMamba + tops + Δtops. 10-ch [xyz, tops, dtops, t]."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.stage1 = MLPBlock([10, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, _, timestep, pts_num = coords.shape
        xyz = coords[:, :3]
        centroid = xyz.mean(dim=-1, keepdim=True)
        rel = xyz - centroid
        tops = rel / rel.norm(dim=1, keepdim=True).clamp(min=1e-6)
        dtops = torch.zeros_like(tops)
        dtops[:, :, :-1] = tops[:, :, 1:] - tops[:, :, :-1]
        coords10 = torch.cat([xyz, tops, dtops, coords[:, 3:4]], dim=1)

        ret = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords10, array2=coords10,
            knn=self.knn[0], dim=3,
        )
        ret = ret.reshape(batchsize, 10, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret)).reshape(
            batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        return torch.cat((coords, self.mamba(fea3)), dim=1)
'''
    src = src.rstrip() + snippet + "\n"
    MOTION.write_text(src, encoding="utf-8")
    print("appended 4 tops variants to models/motion.py")
