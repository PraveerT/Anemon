"""Add MotionRigidResFBQ: PMamba with forward+backward quat Kabsch residual.

Winner of 5-seed tiny-model A/B/C/D test: Cfbq = +1.50pp over baseline.
Literal quaternion rotation (Hamilton sandwich product), not R·p.

Per pair (t, t+1):
  fwd: (q_f, tr_f) = kabsch_quat(p[t], p[t+1], matched[t])
       res_fwd[t+1] = p[t+1] - (quat_rotate(q_f, p[t]) + tr_f)
  bwd: (q_b, tr_b) = kabsch_quat(p[t+1], p[t], matched[t])
       res_bwd[t]   = p[t]   - (quat_rotate(q_b, p[t+1]) + tr_b)

Input to stage1: [xyz(3), res_fwd(3), res_bwd(3), t(1)] = 10-ch.
res_fwd[0] = 0 (no prior frame); res_bwd[T-1] = 0 (no next frame).
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionRigidResFBQ" in src:
    start = src.find("\n\nclass MotionRigidResFBQ")
    src = src[:start]
    print("stripped existing MotionRigidResFBQ")

snippet = '''

def _rrfbq_hamilton(a, b):
    aw,ax,ay,az = a[...,0], a[...,1], a[...,2], a[...,3]
    bw,bx,by,bz = b[...,0], b[...,1], b[...,2], b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _rrfbq_rot_to_quat(R):
    """Batched R (B,3,3) -> q (B,4) via Shepperd's branchless trick."""
    B = R.shape[0]
    m = R
    tr = m[:,0,0] + m[:,1,1] + m[:,2,2]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)
    # Use tr>0 branch for the majority; handle edge via fallback
    s = torch.sqrt(tr.clamp(min=-0.999) + 1.0) * 2.0
    s = s.clamp(min=1e-6)
    q[:,0] = 0.25 * s
    q[:,1] = (m[:,2,1] - m[:,1,2]) / s
    q[:,2] = (m[:,0,2] - m[:,2,0]) / s
    q[:,3] = (m[:,1,0] - m[:,0,1]) / s
    # Fallback for tr<=0 rows
    bad = tr <= 0
    if bad.any():
        idx = bad.nonzero(as_tuple=True)[0]
        for i in idx.tolist():
            mi = m[i]
            if (mi[0,0] > mi[1,1]) and (mi[0,0] > mi[2,2]):
                s2 = torch.sqrt(1 + mi[0,0] - mi[1,1] - mi[2,2]).clamp(min=1e-6) * 2
                q[i,0] = (mi[2,1] - mi[1,2]) / s2
                q[i,1] = 0.25 * s2
                q[i,2] = (mi[0,1] + mi[1,0]) / s2
                q[i,3] = (mi[0,2] + mi[2,0]) / s2
            elif mi[1,1] > mi[2,2]:
                s2 = torch.sqrt(1 + mi[1,1] - mi[0,0] - mi[2,2]).clamp(min=1e-6) * 2
                q[i,0] = (mi[0,2] - mi[2,0]) / s2
                q[i,1] = (mi[0,1] + mi[1,0]) / s2
                q[i,2] = 0.25 * s2
                q[i,3] = (mi[1,2] + mi[2,1]) / s2
            else:
                s2 = torch.sqrt(1 + mi[2,2] - mi[0,0] - mi[1,1]).clamp(min=1e-6) * 2
                q[i,0] = (mi[1,0] - mi[0,1]) / s2
                q[i,1] = (mi[0,2] + mi[2,0]) / s2
                q[i,2] = (mi[1,2] + mi[2,1]) / s2
                q[i,3] = 0.25 * s2
    q = torch.nn.functional.normalize(q, dim=-1)
    # hemisphere pin
    sign = torch.where(q[:,0:1] < 0, -torch.ones_like(q[:,0:1]), torch.ones_like(q[:,0:1]))
    return q * sign


def _rrfbq_quat_rotate(q, points):
    """q: (B,4) unit, points: (B,N,3) -> (B,N,3) via sandwich product."""
    B, N, _ = points.shape
    q_b = q.unsqueeze(1).expand(B, N, 4)
    pq = torch.cat([torch.zeros(B, N, 1, device=points.device, dtype=points.dtype), points], dim=-1)
    q_conj = torch.cat([q_b[...,0:1], -q_b[...,1:]], dim=-1)
    return _rrfbq_hamilton(_rrfbq_hamilton(q_b, pq), q_conj)[...,1:]


def _rrfbq_kabsch_quat(src, tgt, weights):
    """src,tgt: (B,N,3) weights: (B,N). Returns q (B,4), t (B,3) via quaternion."""
    device = src.device
    w = weights.float().clamp(min=0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum  # (B,1,3)
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
    q = _rrfbq_rot_to_quat(R)                                    # (B,4)
    rot_sm = _rrfbq_quat_rotate(q, sm)                           # (B,1,3)
    t = tm.squeeze(1) - rot_sm.squeeze(1)                        # (B,3)
    return q, t


class MotionRigidResFBQ(Motion):
    """PMamba + fwd+bwd quaternion Kabsch residuals as extra 6-ch stage1 input."""

    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        # 10-ch input for stage1 [xyz(3), res_fwd(3), res_bwd(3), t(1)]
        self.stage1 = MLPBlock([10, 32, 64], 2)

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
        xyz = coords[:, :3]
        time_ch = coords[:, 3:4]

        with torch.no_grad():
            res_fwd = torch.zeros_like(xyz)
            res_bwd = torch.zeros_like(xyz)
            if has_corr and corr_matched is not None:
                xyz_p = xyz.permute(0, 2, 3, 1)                       # (B,T,P,3)
                for t in range(timestep - 1):
                    src = xyz_p[:, t]
                    tgt = xyz_p[:, t + 1]
                    w = corr_matched[:, t].float()
                    # forward: src->tgt, residual stored at t+1
                    q_f, tr_f = _rrfbq_kabsch_quat(src, tgt, w)
                    rigid_f = _rrfbq_quat_rotate(q_f, src) + tr_f.unsqueeze(1)
                    rf = tgt - rigid_f
                    res_fwd[:, :, t + 1] = rf.permute(0, 2, 1)
                    # backward: tgt->src, residual stored at t
                    q_b, tr_b = _rrfbq_kabsch_quat(tgt, src, w)
                    rigid_b = _rrfbq_quat_rotate(q_b, tgt) + tr_b.unsqueeze(1)
                    rb = src - rigid_b
                    res_bwd[:, :, t] = rb.permute(0, 2, 1)
                # res_fwd[0] and res_bwd[T-1] stay zero

        coords10 = torch.cat([xyz, res_fwd, res_bwd, time_ch], dim=1)

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords10, array2=coords10,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 10, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )
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
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)
'''

MOTION.write_text(src.rstrip() + snippet + "\n", encoding="utf-8")
print("added MotionRigidResFBQ to models/motion.py")
