"""Add MotionDQCC: PMamba + full DQCC aux.
- Predict DQ per frame pair from per-frame pooled features
- L_anchor = MSE(DQ_pred, DQ_obs from Kabsch)
- L_cycle = MSE(composed DQ over all 31 pairs, observed cumulative DQ(0, 31))
Total aux = anchor_weight * L_anchor + cycle_weight * L_cycle.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionDQCC" in src:
    start = src.find("\n\nclass MotionDQCC")
    src = src[:start]

snippet = '''

def _dqcc_hamilton(a, b):
    aw,ax,ay,az = a[...,0],a[...,1],a[...,2],a[...,3]
    bw,bx,by,bz = b[...,0],b[...,1],b[...,2],b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _dqcc_dq_mul(p_r, p_d, q_r, q_d):
    return _dqcc_hamilton(p_r, q_r), _dqcc_hamilton(p_r, q_d) + _dqcc_hamilton(p_d, q_r)


def _dqcc_rot_to_quat(R):
    orig = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    m00,m01,m02 = Rf[:,0,0], Rf[:,0,1], Rf[:,0,2]
    m10,m11,m12 = Rf[:,1,0], Rf[:,1,1], Rf[:,1,2]
    m20,m21,m22 = Rf[:,2,0], Rf[:,2,1], Rf[:,2,2]
    tr = m00 + m11 + m22
    B = Rf.shape[0]; device = Rf.device
    q = torch.zeros(B, 4, device=device, dtype=Rf.dtype)
    m1 = tr > 0
    if m1.any():
        s = torch.sqrt(tr[m1].clamp(min=-0.999) + 1.0) * 2
        q[m1,0]=0.25*s; q[m1,1]=(m21[m1]-m12[m1])/s
        q[m1,2]=(m02[m1]-m20[m1])/s; q[m1,3]=(m10[m1]-m01[m1])/s
    rem = ~m1
    m2a = rem & (m00>m11) & (m00>m22)
    if m2a.any():
        s = torch.sqrt(1+m00[m2a]-m11[m2a]-m22[m2a]).clamp(min=1e-8)*2
        q[m2a,0]=(m21[m2a]-m12[m2a])/s; q[m2a,1]=0.25*s
        q[m2a,2]=(m01[m2a]+m10[m2a])/s; q[m2a,3]=(m02[m2a]+m20[m2a])/s
    m2b = rem & (~m2a) & (m11>m22)
    if m2b.any():
        s = torch.sqrt(1+m11[m2b]-m00[m2b]-m22[m2b]).clamp(min=1e-8)*2
        q[m2b,0]=(m02[m2b]-m20[m2b])/s; q[m2b,1]=(m01[m2b]+m10[m2b])/s
        q[m2b,2]=0.25*s; q[m2b,3]=(m12[m2b]+m21[m2b])/s
    m2c = rem & (~m2a) & (~m2b)
    if m2c.any():
        s = torch.sqrt(1+m22[m2c]-m00[m2c]-m11[m2c]).clamp(min=1e-8)*2
        q[m2c,0]=(m10[m2c]-m01[m2c])/s; q[m2c,1]=(m02[m2c]+m20[m2c])/s
        q[m2c,2]=(m12[m2c]+m21[m2c])/s; q[m2c,3]=0.25*s
    return F.normalize(q, dim=-1).reshape(*orig, 4)


def _dqcc_kabsch_rt(src, tgt, weights):
    shape = src.shape[:-2]
    N = src.shape[-2]
    sf = src.reshape(-1, N, 3); tf = tgt.reshape(-1, N, 3)
    B = sf.shape[0]; device = sf.device
    w = weights.reshape(-1, N).float().clamp(min=0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (sf * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tf * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = sf - sm; tc = tf - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    q_r = _dqcc_rot_to_quat(R)
    t = tm.squeeze(1) - torch.bmm(R, sm.transpose(-1, -2)).squeeze(-1)
    return q_r.reshape(*shape, 4), t.reshape(*shape, 3)


class MotionDQCC(Motion):
    """PMamba + full DQCC aux: anchor + cycle."""

    def __init__(self, num_classes, pts_size, anchor_weight=0.05,
                 cycle_weight=0.02, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.anchor_weight = anchor_weight
        self.cycle_weight = cycle_weight
        feat_dim = 64
        self.dq_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 8),
        )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

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

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        active = self.training and has_corr and (self.anchor_weight > 0 or self.cycle_weight > 0)
        if active:
            feat_frame = fea1_raw.mean(dim=-1).transpose(1, 2)      # (B, T, 64)
            T = feat_frame.shape[1]
            # Predict DQ per pair
            pair_in = torch.cat([feat_frame[:, :-1], feat_frame[:, 1:]], dim=-1)
            dq_pred = self.dq_head(pair_in)                         # (B, T-1, 8)
            q_r_pred = F.normalize(dq_pred[..., :4], dim=-1)
            q_d_pred = dq_pred[..., 4:]
            q_d_pred = q_d_pred - (q_d_pred * q_r_pred).sum(-1, keepdim=True) * q_r_pred

            # Observable DQ per pair
            with torch.no_grad():
                xyz_p = coords[:, :3].permute(0, 2, 3, 1)           # (B, T, P, 3)
                q_r_list, t_list = [], []
                for t in range(T - 1):
                    q_r, tr = _dqcc_kabsch_rt(xyz_p[:, t], xyz_p[:, t + 1],
                                              corr_matched[:, t])
                    q_r_list.append(q_r); t_list.append(tr)
                q_r_obs = torch.stack(q_r_list, dim=1)
                t_obs = torch.stack(t_list, dim=1)
                zero = torch.zeros_like(t_obs[..., :1])
                t_quat = torch.cat([zero, t_obs], dim=-1)
                q_d_obs = 0.5 * _dqcc_hamilton(t_quat, q_r_obs)

            # Sign-align q_r double cover
            cos_r = (q_r_pred * q_r_obs).sum(-1, keepdim=True)
            sign = torch.where(cos_r >= 0, torch.ones_like(cos_r), -torch.ones_like(cos_r))
            q_r_signed = q_r_pred * sign
            q_d_signed = q_d_pred * sign

            # Anchor loss
            anchor_loss = (F.mse_loss(q_r_signed, q_r_obs)
                           + F.mse_loss(q_d_signed, q_d_obs))

            # Cycle loss: compose all predicted + compare to cumulative observed
            cum_r_pred, cum_d_pred = q_r_signed[:, 0], q_d_signed[:, 0]
            cum_r_obs, cum_d_obs = q_r_obs[:, 0], q_d_obs[:, 0]
            for t in range(1, T - 1):
                cum_r_pred, cum_d_pred = _dqcc_dq_mul(
                    cum_r_pred, cum_d_pred, q_r_signed[:, t], q_d_signed[:, t])
                cum_r_pred = F.normalize(cum_r_pred, dim=-1)
                with torch.no_grad():
                    cum_r_obs, cum_d_obs = _dqcc_dq_mul(
                        cum_r_obs, cum_d_obs, q_r_obs[:, t], q_d_obs[:, t])
                    cum_r_obs = F.normalize(cum_r_obs, dim=-1)
            cycle_loss = (F.mse_loss(cum_r_pred, cum_r_obs)
                          + F.mse_loss(cum_d_pred, cum_d_obs))

            total = self.anchor_weight * anchor_loss + self.cycle_weight * cycle_loss
            self.latest_aux_loss = total
            self.latest_aux_metrics = {
                "qcc_raw": total.detach(),
                "qcc_forward": anchor_loss.detach(),
                "qcc_backward": cycle_loss.detach(),
                "qcc_valid_ratio": corr_matched.float().mean().detach(),
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
print("added MotionDQCC to models/motion.py")
