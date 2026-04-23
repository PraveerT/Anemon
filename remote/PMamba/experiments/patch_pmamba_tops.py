"""Add MotionTops: PMamba with centroid-radial tops added as stage1 input.

Tops = unit direction (p - centroid)/|p - centroid| per frame. Concats with
xyz+t into 7-ch input for stage1's edge-conv grouping. Subsequent stages use
original 4-ch coords, so only stage1 weights differ from vanilla PMamba.

Finetune recipe: load PMamba best (epoch110, 89.83%), strict_load=False so
stage1 reinits, run 20 epochs at lr 0.00012 (pre-first-drop PMamba schedule).

Hypothesis: tops gave +2pp on quaternion branch (v26a). If PMamba has the
capacity to exploit it, 89.83 can become 90+.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionTops" in src:
    print("MotionTops already present")
else:
    snippet = '''

class MotionTops(Motion):
    """PMamba Motion + centroid-radial tops direction as extra stage1 input.

    Only stage1 sees 7-ch [xyz, tops_xyz, t]. fea1 concat uses original
    4-ch coords, so stages 2-3 are identical to vanilla PMamba.
    """

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        # Replace stage1 for 7-channel input; reinits from scratch on load.
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _encode_sampled_points(self, coords):
        batchsize, in_dims, timestep, pts_num = coords.shape

        # Tops: per-frame centroid-radial unit direction.
        xyz = coords[:, :3]                                           # (B,3,T,P)
        centroid = xyz.mean(dim=-1, keepdim=True)                     # (B,3,T,1)
        rel = xyz - centroid
        tops = rel / rel.norm(dim=1, keepdim=True).clamp(min=1e-6)    # (B,3,T,P)
        coords7 = torch.cat([xyz, tops, coords[:, 3:4]], dim=1)       # (B,7,T,P)

        # stage 1: intra-frame with 7-ch input.
        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea1 = torch.cat((coords, fea1), dim=1)                       # use 4-ch coords

        # stages 2-3 identical to Motion._encode_sampled_points.
        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        ret_group_array2 = self.group.st_group_points(
            fea1, 3, [0, 1, 2], self.knn[1], 3,
        )
        ret_array2, coords = self.select_ind(
            ret_group_array2, coords, batchsize, in_dims, timestep, pts_num,
        )
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        ret_group_array3 = self.group.st_group_points(
            fea2, 3, [0, 1, 2], self.knn[2], 3,
        )
        ret_array3, coords = self.select_ind(
            ret_group_array3, coords, batchsize, in_dims, timestep, pts_num,
        )
        fea3 = self.pool3(self.stage3(ret_array3)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea3_mamba = self.mamba(fea3)
        return torch.cat((coords, fea3_mamba), dim=1)
'''
    src = src.rstrip() + snippet + "\n"
    MOTION.write_text(src, encoding="utf-8")
    print("appended MotionTops to models/motion.py")
