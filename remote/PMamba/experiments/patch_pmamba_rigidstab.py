"""Add MotionRigidStabilize: PMamba over rigid-motion-removed pointcloud.

Idea: undo whole-hand rigid motion globally using forward quaternion Kabsch.
PMamba sees xyz in frame-0 reference coords; only articulation (finger curl,
thumb tuck) remains in geometry. PMamba's st_group_points kNN now groups
truly corresponding points across time instead of chasing the moving hand.

Per pair (t, t+1): (q_pair, tr_pair) = kabsch_quat(p[t], p[t+1], matched[t])
Chain: q_t composes as Hamilton(q_pair, q_{t-1});  T_t = quat_rot(q_pair, T_{t-1}) + tr_pair
Stabilize: xyz_stable[t] = quat_rot(conj(q_t), xyz[t] - T_t)
Frame 0 anchor: q_0 = identity, T_0 = 0  ->  xyz_stable[0] = xyz[0]

Output to stage1 is plain 4-ch [xyz_stable, t] — vanilla PMamba interface.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionRigidStabilize" in src:
    start = src.find("\n\nclass MotionRigidStabilize")
    src = src[:start]
    print("stripped existing MotionRigidStabilize")

snippet = '''

class MotionRigidStabilize(Motion):
    """PMamba over rigid-motion-removed pointcloud.

    Replaces xyz with frame-0-referenced (stabilized) xyz before PMamba.
    Reuses _rrfbq_* quaternion math from MotionRigidResFBQ.
    Stage1 input is 4-ch [xyz_stable, t] — vanilla PMamba shape.
    """

    def _corr_sample(self, points, aux_input):
        # Same correspondence-aware sampler as MotionRigidResFBQ
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

    def _stabilize(self, xyz_p, matched):
        """xyz_p: (B,T,P,3), matched: (B,T-1,P) -> stabilized (B,T,P,3) in frame-0 coords."""
        B, T, P, _ = xyz_p.shape
        device = xyz_p.device
        # accumulated transform (q_t, T_t): xyz[t] = quat_rot(q_t, xyz[0]) + T_t
        q_acc = torch.zeros(B, 4, device=device, dtype=xyz_p.dtype)
        q_acc[:, 0] = 1.0  # identity quat (1,0,0,0)
        T_acc = torch.zeros(B, 3, device=device, dtype=xyz_p.dtype)
        out = torch.zeros_like(xyz_p)
        out[:, 0] = xyz_p[:, 0]  # frame 0 anchor
        for t in range(T - 1):
            src = xyz_p[:, t]
            tgt = xyz_p[:, t + 1]
            w = matched[:, t].float()
            # If a sample has too few matches, kabsch fit is unreliable
            n_match = w.sum(-1)  # (B,)
            ok = n_match >= 4  # need >=4 points for stable fit
            q_pair, tr_pair = _rrfbq_kabsch_quat(src, tgt, w)
            # Compose: q_{t+1} = q_pair * q_acc;  T_{t+1} = rot(q_pair, T_acc) + tr_pair
            q_new = _rrfbq_hamilton(q_pair, q_acc)
            q_new = torch.nn.functional.normalize(q_new, dim=-1)
            T_new = _rrfbq_quat_rotate(q_pair, T_acc.unsqueeze(1)).squeeze(1) + tr_pair
            # Fallback for unreliable rows: keep prev accumulated transform
            ok3 = ok.float().unsqueeze(-1)
            q_acc = ok3 * q_new + (1 - ok3) * q_acc
            q_acc = torch.nn.functional.normalize(q_acc, dim=-1)
            T_acc = ok3 * T_new + (1 - ok3) * T_acc
            # Apply inverse to map xyz[t+1] back into frame-0 coords
            q_conj = torch.cat([q_acc[:, 0:1], -q_acc[:, 1:]], dim=-1)
            shifted = xyz_p[:, t + 1] - T_acc.unsqueeze(1)
            stabilized = _rrfbq_quat_rotate(q_conj, shifted)
            out[:, t + 1] = stabilized
        return out

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
        xyz = coords[:, :3]                                          # (B,3,T,P)
        time_ch = coords[:, 3:4]                                     # (B,1,T,P)

        with torch.no_grad():
            if has_corr and corr_matched is not None:
                xyz_p = xyz.permute(0, 2, 3, 1).contiguous()         # (B,T,P,3)
                xyz_stab = self._stabilize(xyz_p, corr_matched)       # (B,T,P,3)
                xyz = xyz_stab.permute(0, 3, 1, 2).contiguous()       # (B,3,T,P)

        coords4 = torch.cat([xyz, time_ch], dim=1)
        coords = coords4  # downstream stages key xyz/time off coords[:, :4]

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords4, array2=coords4,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 4, timestep * pts_num, -1)
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
print("added MotionRigidStabilize to models/motion.py")
