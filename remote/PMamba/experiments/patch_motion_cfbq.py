"""Add MotionCfbq: PMamba + per-point fwd/bwd Kabsch residuals only (no tops).

Stage1 input layout per point (10 channels):
  [u, v, d, t]            (4) baseline pixel + time, normalized by loader
  [rf_x, rf_y, rf_z]      (3) per-point fwd Kabsch residual
  [rb_x, rb_y, rb_z]      (3) per-point bwd Kabsch residual
"""
from pathlib import Path

M = Path("models/motion.py")
src = M.read_text(encoding="utf-8")
if "class MotionCfbq" in src:
    i = src.find("\n\nclass MotionCfbq")
    j = src.find("\n\nclass ", i + 1)
    src = src[:i] if j == -1 else src[:i] + src[j:]
    print("stripped existing MotionCfbq")

snippet = '''


class MotionCfbq(Motion):
    """PMamba + per-point fwd/bwd Kabsch residuals only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage1 = MLPBlock([10, 32, 64], 2)

    @staticmethod
    def _kabsch_pair(src, tgt):
        src_mu = src.mean(dim=2, keepdim=True)
        tgt_mu = tgt.mean(dim=2, keepdim=True)
        src_c = src - src_mu
        tgt_c = tgt - tgt_mu
        H = torch.einsum("btni,btnj->btij", src_c, tgt_c)
        H = H + 1e-6 * torch.eye(3, device=src.device).view(1, 1, 3, 3)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-1, -2)
        det = torch.det(V @ U.transpose(-1, -2))
        D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
        R = V @ D @ U.transpose(-1, -2)
        t_vec = tgt_mu.squeeze(2) - torch.einsum("btij,btj->bti", R, src_mu.squeeze(2))
        return R, t_vec

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
        sampled = points[:, :4]
        xyz = points[:, 4:7]

        x = xyz.permute(0, 2, 3, 1)                                       # (B, T, P, 3)
        B, T, P, _ = x.shape
        R_fwd, t_fwd = self._kabsch_pair(x[:, :-1], x[:, 1:])
        R_bwd, t_bwd = self._kabsch_pair(x[:, 1:], x[:, :-1])
        rf = torch.zeros_like(x); rb = torch.zeros_like(x)
        for ti in range(1, T):
            R = R_fwd[:, ti - 1]; tt = t_fwd[:, ti - 1].unsqueeze(1)
            pred = torch.einsum("bij,bnj->bni", R, x[:, ti - 1]) + tt
            rf[:, ti] = x[:, ti] - pred
        for ti in range(0, T - 1):
            R = R_bwd[:, ti]; tt = t_bwd[:, ti].unsqueeze(1)
            pred = torch.einsum("bij,bnj->bni", R, x[:, ti + 1]) + tt
            rb[:, ti] = x[:, ti] - pred
        rf_chw = rf.permute(0, 3, 1, 2).detach()
        rb_chw = rb.permute(0, 3, 1, 2).detach()
        return torch.cat([sampled, rf_chw, rb_chw], dim=1)                # (B, 10, T, P)

    def _encode_sampled_points(self, coords10):
        batchsize, _, timestep, pts_num = coords10.shape
        coords = coords10[:, :4]
        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords10, array2=coords10,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 10, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)
        in_dims = fea1.shape[1] * 2 - 4
        pts_num_s2 = pts_num // self.downsample[0]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret_array2, coords = self.select_ind(ret_group_array2, coords, batchsize, in_dims, timestep, pts_num_s2)
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(batchsize, -1, timestep, pts_num_s2)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)
        in_dims = fea2.shape[1] * 2 - 4
        pts_num_s3 = pts_num_s2 // self.downsample[1]
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret_array3, coords = self.select_ind(ret_group_array3, coords, batchsize, in_dims, timestep, pts_num_s3)
        fea3 = self.pool3(self.stage3(ret_array3)).reshape(batchsize, -1, timestep, pts_num_s3)
        fea3_mamba = self.mamba(fea3)
        return torch.cat((coords, fea3_mamba), dim=1)
'''

M.write_text(src.rstrip() + snippet + "\n", encoding="utf-8")
print("appended MotionCfbq to models/motion.py")
