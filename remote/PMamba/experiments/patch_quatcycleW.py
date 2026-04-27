"""Add MotionQuatCycleW to motion.py.

Per-frame Kabsch fwd quaternion -> rotation axis -> bivector
B = n_hat n_hat^T - (1/3) I, vectorized to 5 independent components.
Broadcast to every point as 5 extra stage1 channels.

Stage1 input: 4 (uvdt) + 5 (bivector) = 9 channels.
"""
from pathlib import Path

M = Path("models/motion.py")
src = M.read_text(encoding="utf-8")
if "class MotionQuatCycleW" in src:
    i = src.find("\n\nclass MotionQuatCycleW")
    j = src.find("\n\nclass ", i + 1)
    src = src[:i] if j == -1 else src[:i] + src[j:]
    print("stripped existing MotionQuatCycleW")

snippet = '''


class MotionQuatCycleW(Motion):
    """PMamba + per-frame rotation-plane bivector (N3 from novel-QCC sweep).

    Input npy: 8-ch [u,v,d,t, x,y,z,t]. Use uvdt as the network's primary
    coords (channels 0..3, normalized). Use xyz (channels 4..6) to compute
    Kabsch quaternion per consecutive-frame pair, extract rotation axis n_hat,
    form symmetric traceless tensor B = n_hat n_hat^T - (1/3)I, vectorize to
    R^5: (b11, b22, b12, b13, b23). Broadcast to every point.

    Stage1 input: 4 (uvdt) + 5 (bivector) = 9 channels.
    Downstream coords stay 4-ch (uvdt).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage1 = MLPBlock([9, 32, 64], 2)

    @staticmethod
    def _rot_to_quat(R):
        m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
        m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
        m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
        tr = m00 + m11 + m22
        s = torch.sqrt((tr + 1).clamp(min=1e-6)) * 2
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
        q = torch.stack([qw, qx, qy, qz], dim=-1)
        return torch.nn.functional.normalize(q, dim=-1)

    def _kabsch_qf(self, xyz):
        # xyz: (B, 3, T, P)
        B, _, T, P = xyz.shape
        x = xyz.permute(0, 2, 3, 1)                                       # (B, T, P, 3)
        src = x[:, :-1]
        tgt = x[:, 1:]
        src_c = src - src.mean(dim=2, keepdim=True)
        tgt_c = tgt - tgt.mean(dim=2, keepdim=True)
        H = torch.einsum("btni,btnj->btij", src_c, tgt_c)
        H = H + 1e-6 * torch.eye(3, device=xyz.device).view(1, 1, 3, 3)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-1, -2)
        det = torch.det(V @ U.transpose(-1, -2))
        D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
        R = V @ D @ U.transpose(-1, -2)                                   # (B, T-1, 3, 3)
        qf = self._rot_to_quat(R)                                          # (B, T-1, 4)
        qf = torch.cat([qf, qf[:, -1:].clone()], dim=1)                   # (B, T, 4) pad last
        return qf

    @staticmethod
    def _bivector(qf):
        # qf: (B, T, 4) -> bivector vectorized (B, T, 5)
        v = qf[..., 1:]                                                   # (B, T, 3)
        nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        nh = v / nrm                                                       # rotation axis
        nx, ny, nz = nh[..., 0:1], nh[..., 1:2], nh[..., 2:3]
        b11 = nx * nx - 1.0 / 3
        b22 = ny * ny - 1.0 / 3
        b12 = nx * ny
        b13 = nx * nz
        b23 = ny * nz
        return torch.cat([b11, b22, b12, b13, b23], dim=-1)               # (B, T, 5)

    def _sample_points(self, inputs):
        # inputs: (B, T, N, 8)
        points = inputs.permute(0, 3, 1, 2)                                # (B, 8, T, N)
        point_count = points.shape[3]
        device = points.device
        sample_size = min(self.pts_size, point_count)
        if self.training:
            indices = torch.randperm(point_count, device=device)[:sample_size]
        else:
            indices = torch.linspace(0, point_count - 1, sample_size, device=device).long()
        points = points[:, :, :, indices]
        sampled = points[:, :4]                                           # (B, 4, T, P) uvdt
        xyz = points[:, 4:7]                                              # (B, 3, T, P)
        qf = self._kabsch_qf(xyz)                                          # (B, T, 4)
        beta = self._bivector(qf)                                          # (B, T, 5)
        # broadcast to every point: (B, 5, T, P)
        beta_p = beta.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, sample_size)
        return torch.cat([sampled, beta_p], dim=1)                         # (B, 9, T, P)

    def _encode_sampled_points(self, coords9):
        batchsize, _, timestep, pts_num = coords9.shape
        coords = coords9[:, :4]                                            # downstream uvdt only

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords9, array2=coords9,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 9, timestep * pts_num, -1)
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
print("appended MotionQuatCycleW to models/motion.py")
