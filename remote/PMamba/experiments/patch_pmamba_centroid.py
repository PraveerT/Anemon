"""Add MotionCentroidAux: PMamba + per-frame centroid regression aux.

Centroid = observable (31.95% standalone signal), non-trivial (no collapse),
and translation-dominant which matches gesture class discriminability.

Aux target: xyz centroid of sampled points per frame (B, T, 3).
Aux head: MLP on per-frame pooled features -> 3-dim prediction.
Loss: MSE.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionCentroidAux" in src:
    start = src.find("\n\nclass MotionCentroidAux")
    src = src[:start]

snippet = '''

class MotionCentroidAux(Motion):
    """PMamba + per-frame centroid regression aux."""

    def __init__(self, num_classes, pts_size, centroid_weight=0.05, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.centroid_weight = centroid_weight
        feat_dim = 64                                    # stage1 out channels
        self.centroid_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
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
        if self.training and self.centroid_weight > 0:
            # Per-frame pooled features -> predict centroid.
            with torch.no_grad():
                centroid_target = coords[:, :3].mean(dim=-1).transpose(1, 2)  # (B, T, 3)
            feat_pooled = fea1_raw.mean(dim=-1).transpose(1, 2)                # (B, T, 64)
            pred = self.centroid_head(feat_pooled)                             # (B, T, 3)
            aux_loss = F.mse_loss(pred, centroid_target)
            self.latest_aux_loss = self.centroid_weight * aux_loss
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
print("added MotionCentroidAux to models/motion.py")
