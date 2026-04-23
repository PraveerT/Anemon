"""Add MotionCFrameAux: PMamba + C_frame aux (per-frame centroid + velocity
+ acceleration regression, 9-dim target per frame).

Strongest standalone signal tested: 37.55% (vs centroid-only 31.95%, pair
DQCC 28%). Predicts per-frame (c, dc, d2c) from pooled per-frame features.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionCFrameAux" in src:
    start = src.find("\n\nclass MotionCFrameAux")
    src = src[:start]

snippet = '''

class MotionCFrameAux(Motion):
    """PMamba + per-frame (centroid, velocity, acceleration) regression aux.

    Target: 9-dim per frame = [c_xyz, dc_xyz, d2c_xyz].
    Supervision target carries 37.55% standalone class info — strongest found.
    """

    def __init__(self, num_classes, pts_size, cframe_weight=0.05, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.cframe_weight = cframe_weight
        feat_dim = 64
        self.cframe_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 9),                           # c, dc, d2c
        )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def extract_features(self, inputs):
        coords = self._sample_points(inputs)
        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )                                                # (B, 64, T, P)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.cframe_weight > 0:
            with torch.no_grad():
                c = coords[:, :3].mean(dim=-1).transpose(1, 2)     # (B, T, 3)
                dc = torch.zeros_like(c); dc[:, :-1] = c[:, 1:] - c[:, :-1]
                d2c = torch.zeros_like(dc); d2c[:, :-1] = dc[:, 1:] - dc[:, :-1]
                target = torch.cat([c, dc, d2c], dim=-1)           # (B, T, 9)
            feat_frame = fea1_raw.mean(dim=-1).transpose(1, 2)     # (B, T, 64)
            pred = self.cframe_head(feat_frame)                    # (B, T, 9)
            aux_loss = F.mse_loss(pred, target)
            self.latest_aux_loss = self.cframe_weight * aux_loss
            self.latest_aux_metrics = {
                "qcc_raw": aux_loss.detach(),
                "qcc_forward": aux_loss.detach(),
                "qcc_backward": aux_loss.detach(),
                "qcc_valid_ratio": torch.ones(1, device=aux_loss.device),
            }

        fea1 = torch.cat((coords, fea1_raw), dim=1)
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
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)
'''
SRC_NEW = src.rstrip() + snippet + "\n"
MOTION.write_text(SRC_NEW, encoding="utf-8")
print("added MotionCFrameAux to models/motion.py")
