"""Patches:
  1) Append `NvidiaRGBDLoader` to nvidia_dataloader.py
  2) Append `MotionRGBD` to models/motion.py

Both follow the established 7-ch stage1 pattern (MotionTops template).
RGB-D layout per point: [u, v, d, t, x, y, z, t, R, G, B]  (11 channels in npy)
Model input to stage1: [u, v, d, t, R/127.5-1, G/127.5-1, B/127.5-1] = 7 channels.
"""
from pathlib import Path

DL = Path("nvidia_dataloader.py")
src = DL.read_text(encoding="utf-8")
if "class NvidiaRGBDLoader" in src:
    i = src.find("\n\nclass NvidiaRGBDLoader")
    j = src.find("\n\nclass ", i + 1)
    src = src[:i] if j == -1 else src[:i] + src[j:]
    print("stripped existing NvidiaRGBDLoader")

dl_snippet = '''


class NvidiaRGBDLoader(NvidiaLoader):
    """Loads `_pts_rgbd.npy` files (T, N, 11) — uvdt + xyz_t + RGB.
    Normalizes RGB to [-1, 1]. Other channels reuse the depth-pipeline stats.
    """

    def get_inputs_list(self):
        prefix = "../dataset/Nvidia/Processed"
        if self.phase == "train":
            inputs_path = prefix + "/train_rgbd_list.txt"
        elif self.phase in ("valid", "test"):
            inputs_path = prefix + "/test_rgbd_list.txt"
        else:
            raise AssertionError("Phase error.")
        return open(inputs_path).readlines()

    def __getitem__(self, index):
        line = self.inputs_list[index]
        parts = self.r.split(line)
        label = int(parts[-2])
        npy_path = "../dataset/" + parts[1][1:]
        input_data = np.load(npy_path).astype(float)            # (T, N, 11)
        # normalize channels 0..3 (uvdt) the same way as base loader
        input_data[..., 0] = (input_data[..., 0] - self.dataset_stats['x_mean']) / self.dataset_stats['x_std']
        input_data[..., 1] = (input_data[..., 1] - self.dataset_stats['y_mean']) / self.dataset_stats['y_std']
        input_data[..., 2] = (input_data[..., 2] - self.dataset_stats['z_mean']) / self.dataset_stats['z_std']
        input_data[..., 3] = (input_data[..., 3] - self.dataset_stats['t_mean']) / self.dataset_stats['t_std']
        # RGB: 0..255 -> [-1, 1]
        input_data[..., 8:11] = input_data[..., 8:11] / 127.5 - 1.0
        return input_data, label, line
'''

DL.write_text(src.rstrip() + dl_snippet + "\n", encoding="utf-8")
print("appended NvidiaRGBDLoader to nvidia_dataloader.py")


# ---------- MotionRGBD ----------
M = Path("models/motion.py")
src = M.read_text(encoding="utf-8")
if "class MotionRGBD" in src:
    i = src.find("\n\nclass MotionRGBD")
    j = src.find("\n\nclass ", i + 1)
    src = src[:i] if j == -1 else src[:i] + src[j:]
    print("stripped existing MotionRGBD")

m_snippet = '''


class MotionRGBD(Motion):
    """PMamba with RGB fed only into stage1. Downstream stays uvd+t (4-ch).

    Input layout per point (channel dim): [u, v, d, t, x, y, z, t, R, G, B]
    Stage1 sees: [uvdt, RGB] = 7 channels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _sample_points(self, inputs):
        points = inputs.permute(0, 3, 1, 2)                 # (B, C, T, N)
        point_count = points.shape[3]
        device = points.device
        sample_size = min(self.pts_size, point_count)
        if self.training:
            indices = torch.randperm(point_count, device=device)[:sample_size]
        else:
            indices = torch.linspace(0, point_count - 1, sample_size, device=device).long()
        points = points[:, :, :, indices]
        sampled = points[:, :4]                             # (B, 4, T, P) uvdt
        rgb = points[:, 8:11]                               # (B, 3, T, P) RGB in [-1,1]
        return torch.cat([sampled, rgb], dim=1)             # (B, 7, T, P)

    def _encode_sampled_points(self, coords7):
        batchsize, _, timestep, pts_num = coords7.shape
        coords = coords7[:, :4]                             # downstream uses uvdt only

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)

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

M.write_text(src.rstrip() + m_snippet + "\n", encoding="utf-8")
print("appended MotionRGBD to models/motion.py")
