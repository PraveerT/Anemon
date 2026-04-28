"""Add MotionDQ: PMamba + global-SE(3) dual quaternion broadcast as input feature.

Stage1 input layout per point (12 channels):
  [u, v, d, t]            (4)  baseline pixel + time, normalized by loader
  [qr_w, qr_x, qr_y, qr_z] (4)  per-frame Kabsch rotation as unit quat
  [qd_w, qd_x, qd_y, qd_z] (4)  dual part (encodes translation)

Per-frame dual quat is broadcast to all points in that frame.
"""
from pathlib import Path

M = Path("models/motion.py")
src = M.read_text(encoding="utf-8")
if "class MotionDQ" in src:
    i = src.find("\n\nclass MotionDQ")
    j = src.find("\n\nclass ", i + 1)
    src = src[:i] if j == -1 else src[:i] + src[j:]
    print("stripped existing MotionDQ")

snippet = '''


class MotionDQ(Motion):
    """PMamba + global-SE(3) dual-quaternion broadcast (8d) per frame."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage1 = MLPBlock([12, 32, 64], 2)

    @staticmethod
    def _kabsch_pair(src, tgt):
        """src,tgt: (B, T, N, 3). Returns R (B, T, 3, 3), t (B, T, 3)."""
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
        t = tgt_mu.squeeze(2) - torch.einsum("btij,btj->bti", R, src_mu.squeeze(2))
        return R, t

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
        sampled = points[:, :4]                                                # (B, 4, T, P)
        xyz = points[:, 4:7]

        x = xyz.permute(0, 2, 3, 1)                                            # (B, T, P, 3)
        B, T, P, _ = x.shape
        # frame-pair Kabsch: align frame t-1 to t, t in [1..T-1]; frame 0 -> identity
        R_pair, t_pair = self._kabsch_pair(x[:, :-1], x[:, 1:])                # (B, T-1, ...)
        qr_pair = self._rot_to_quat(R_pair)                                    # (B, T-1, 4)
        t_quat = torch.cat([torch.zeros_like(t_pair[..., :1]), t_pair], dim=-1) # (B, T-1, 4)
        qd_pair = 0.5 * self._hamilton(t_quat, qr_pair)                        # (B, T-1, 4)
        # prepend identity for frame 0
        ident_qr = torch.zeros(B, 1, 4, device=device); ident_qr[..., 0] = 1.0
        ident_qd = torch.zeros(B, 1, 4, device=device)
        qr = torch.cat([ident_qr, qr_pair], dim=1)                             # (B, T, 4)
        qd = torch.cat([ident_qd, qd_pair], dim=1)                             # (B, T, 4)
        # normalize qd by a learned-free constant scale (depth coords ~100, halve to ~50)
        qd = qd / 50.0
        dq = torch.cat([qr, qd], dim=-1)                                       # (B, T, 8)
        # broadcast to (B, 8, T, P)
        dq_chw = dq.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, P).detach()
        return torch.cat([sampled, dq_chw], dim=1)                             # (B, 12, T, P)

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
print("appended MotionDQ to models/motion.py")
