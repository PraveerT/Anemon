"""Add MotionQuatCycleAA to motion.py — Dual Quaternion (SE(3)) as stage1 input.

Per consecutive frame pair: Kabsch -> (R, t). Compute:
  q     = quat(R)                        (4-d)
  q'    = 0.5 * [0, t] * q   (Hamilton)  (4-d)
DQ = (q, q') in R^8, broadcast across every point.

Stage1 input: 4 (uvdt) + 8 (dq) = 12 channels.
Downstream coords stay 4-ch (uvdt).
"""
from pathlib import Path

M = Path("models/motion.py")
src = M.read_text(encoding="utf-8")
if "class MotionQuatCycleAA" in src:
    i = src.find("\n\nclass MotionQuatCycleAA")
    j = src.find("\n\nclass ", i + 1)
    src = src[:i] if j == -1 else src[:i] + src[j:]
    print("stripped existing MotionQuatCycleAA")

snippet = '''


class MotionQuatCycleAA(Motion):
    """PMamba + per-frame Dual Quaternion (SE(3)) input feature.

    DQ captures both rotation and translation in unified algebra. Pair-grid
    showed +2.07pp mean over baseline across 6 confusion pairs (5/6 positive).
    Stage1 input: 4 (uvdt) + 8 (dq) = 12 channels.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage1 = MLPBlock([12, 32, 64], 2)

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

    @staticmethod
    def _hamilton(q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def _kabsch_dq(self, xyz):
        # xyz: (B, 3, T, P) -> (qf, q_dual) (B, T, 4) each
        B, _, T, P = xyz.shape
        x = xyz.permute(0, 2, 3, 1)                                       # (B, T, P, 3)
        src = x[:, :-1]
        tgt = x[:, 1:]
        src_mu = src.mean(dim=2, keepdim=True)
        tgt_mu = tgt.mean(dim=2, keepdim=True)
        src_c = src - src_mu
        tgt_c = tgt - tgt_mu
        H = torch.einsum("btni,btnj->btij", src_c, tgt_c)
        H = H + 1e-6 * torch.eye(3, device=xyz.device).view(1, 1, 3, 3)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-1, -2)
        det = torch.det(V @ U.transpose(-1, -2))
        D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
        R = V @ D @ U.transpose(-1, -2)                                   # (B, T-1, 3, 3)
        qf = self._rot_to_quat(R)                                          # (B, T-1, 4)
        # translation
        t_fwd = tgt_mu.squeeze(2) - torch.einsum("btij,btj->bti", R, src_mu.squeeze(2))   # (B, T-1, 3)
        # dual part q' = 0.5 * [0, t] * q
        qt = torch.cat([torch.zeros_like(t_fwd[..., 0:1]), t_fwd], dim=-1)               # (B, T-1, 4)
        q_dual = 0.5 * self._hamilton(qt, qf)                              # (B, T-1, 4)
        # pad last frame
        qf = torch.cat([qf, qf[:, -1:].clone()], dim=1)                   # (B, T, 4)
        q_dual = torch.cat([q_dual, q_dual[:, -1:].clone()], dim=1)       # (B, T, 4)
        return qf, q_dual

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
        qf, q_dual = self._kabsch_dq(xyz)                                  # (B, T, 4) each
        dq = torch.cat([qf, q_dual], dim=-1)                              # (B, T, 8)
        # broadcast to every point: (B, 8, T, P)
        dq_p = dq.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, sample_size)
        return torch.cat([sampled, dq_p], dim=1)                          # (B, 12, T, P)

    def _encode_sampled_points(self, coords12):
        batchsize, _, timestep, pts_num = coords12.shape
        coords = coords12[:, :4]

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords12, array2=coords12,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 12, timestep * pts_num, -1)
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
print("appended MotionQuatCycleAA to models/motion.py")
