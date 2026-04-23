"""Add MotionRigidRes: PMamba with rigid-subtraction residual as extra 3-ch input.

For each correspondence-aligned pair (t, t+1):
  1. Kabsch: (R, trans) = argmin ||R p(t) + trans - p(t+1)||
  2. Rigid prediction: p_rigid(t+1) = R p(t) + trans
  3. Residual: res(t+1) = p(t+1) - p_rigid(t+1)

Input to stage1: [xyz, res_xyz, t] = 7-ch. Frame 0 residual = zero (no prior).
Stage1 expanded to 7-ch like MotionTops. Stages 2-3 unchanged.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionRigidRes" in src:
    start = src.find("\n\nclass MotionRigidRes")
    # Strip until end of file (last class)
    src = src[:start]
    print("stripped existing MotionRigidRes")

snippet = '''

def _rr_kabsch(src, tgt, weights):
    """Kabsch batch. src, tgt: (B, N, 3). weights: (B, N). Returns R (B,3,3), t (B,3)."""
    B = src.shape[0]; device = src.device
    w = weights.float().clamp(min=0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tgt * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src - sm; tc = tgt - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack(
        [torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    t = tm.squeeze(1) - torch.bmm(R, sm.transpose(-1, -2)).squeeze(-1)
    return R, t


class MotionRigidRes(Motion):
    """PMamba + rigid-subtraction residual as extra 3-ch stage1 input."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        # 7-ch input for stage1 [xyz, res_xyz, t]
        self.stage1 = MLPBlock([7, 32, 64], 2)

    def _corr_sample(self, points, aux_input):
        B, F_, P, C = points.shape
        S = min(self.pts_size, P)
        device = points.device
        if S == P:
            return points, torch.ones(B, F_ - 1, P, dtype=torch.bool, device=device)
        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // F_
        sampled = torch.zeros(B, F_, S, C, device=device, dtype=points.dtype)
        matched = torch.zeros(B, F_ - 1, S, dtype=torch.bool, device=device)
        for b in range(B):
            if self.training:
                idx = torch.randperm(P, device=device)[:S]
            else:
                idx = torch.linspace(0, P - 1, S, device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()
            for t in range(F_ - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(P, device=device)
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]
                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))
                next_idx = torch.randint(0, P, (S,), device=device)
                next_idx[valid] = tgt_pos[valid]
                sampled[b, t + 1] = points[b, t + 1, next_idx]
                matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()
        return sampled, matched

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
            aux = inputs
            has_corr = ("orig_flat_idx" in aux and "corr_full_target_idx" in aux
                        and "corr_full_weight" in aux)
        else:
            points_raw = inputs
            aux = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._corr_sample(points_raw[..., :4], aux)
            coords = sampled.permute(0, 3, 1, 2).contiguous()
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape
        xyz = coords[:, :3]                                  # (B, 3, T, P)
        time_ch = coords[:, 3:4]

        # Compute rigid-subtraction residuals per-frame (needs correspondence)
        with torch.no_grad():
            res = torch.zeros_like(xyz)                       # (B, 3, T, P)
            if has_corr and corr_matched is not None:
                xyz_p = xyz.permute(0, 2, 3, 1)               # (B, T, P, 3)
                for t in range(timestep - 1):
                    src = xyz_p[:, t]                         # (B, P, 3)
                    tgt = xyz_p[:, t + 1]
                    w = corr_matched[:, t].float()
                    R, tr = _rr_kabsch(src, tgt, w)           # (B,3,3), (B,3)
                    rigid_pred = torch.bmm(R, src.transpose(-1, -2)).transpose(-1, -2) + tr.unsqueeze(1)
                    r = tgt - rigid_pred                      # (B, P, 3)
                    res[:, :, t + 1] = r.permute(0, 2, 1)     # back to (B,3,P)
                # frame 0 residual stays zero

        coords7 = torch.cat([xyz, res, time_ch], dim=1)       # (B, 7, T, P)

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords7, array2=coords7,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 7, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )
        fea1 = torch.cat((coords, fea1), dim=1)               # use 4-ch coords thereafter

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

MOTION.write_text(src.rstrip() + snippet + "\n", encoding="utf-8")
print("added MotionRigidRes to models/motion.py")
