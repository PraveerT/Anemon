"""Add MotionStrain: PMamba with per-point Green-Lagrange strain as 2-ch input.

Per frame-pair (t, t+1) using correspondence-aligned points:
  1. kNN at t (k=12 neighbors).
  2. Deformation gradient F_i = (sum dtgt @ dsrc^T) (sum dsrc @ dsrc^T + reg)^-1
     where reg = 1e-2 * trace(B)/3 (Tikhonov, scaled by trace).
  3. Strain eps_i = 0.5 * (F_i^T F_i - I).
  4. Two scalars: log1p(|frob|), log1p(|trace|), each clamped [-3, 3].

Frame 0 strain = zero (no prior).
Stage1 input: [xyz(3), strain_frob, strain_trace, t(1)] = 6-ch.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionStrain" in src:
    start = src.find("\n\nclass MotionStrain")
    end = src.find("\n\nclass ", start + 1)
    if end == -1:
        src = src[:start]
    else:
        src = src[:start] + src[end:]
    print("stripped existing MotionStrain")

snippet = '''


def _strain_field(p_t, p_tp1, k=12):
    """Per-point Green-Lagrange strain, reduced to (frob, trace) scalars.

    p_t, p_tp1: (B, P, 3) correspondence-aligned point clouds.
    Returns: (B, P, 2) with log1p-compressed (frob, trace) clamped to [-3, 3].
    """
    B, P, _ = p_t.shape
    device = p_t.device
    d = torch.cdist(p_t, p_t)                          # (B, P, P)
    _, knn_idx = d.topk(k + 1, dim=-1, largest=False)
    knn_idx = knn_idx[..., 1:]                         # drop self -> (B, P, k)
    batch_idx = torch.arange(B, device=device)[:, None, None].expand(-1, P, k)
    nb_t = p_t[batch_idx, knn_idx]                     # (B, P, k, 3)
    nb_tp1 = p_tp1[batch_idx, knn_idx]                 # (B, P, k, 3)
    dsrc = nb_t - p_t.unsqueeze(2)                     # (B, P, k, 3)
    dtgt = nb_tp1 - p_tp1.unsqueeze(2)                 # (B, P, k, 3)
    A = torch.einsum('bpki,bpkj->bpij', dtgt, dsrc)    # (B, P, 3, 3)
    Bm = torch.einsum('bpki,bpkj->bpij', dsrc, dsrc)   # (B, P, 3, 3)
    trace_B = Bm.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    reg = (1e-2 * trace_B / 3.0).clamp(min=1e-3)
    eye3 = torch.eye(3, device=device).view(1, 1, 3, 3)
    Bm_reg = Bm + reg * eye3
    F_grad = A @ torch.linalg.inv(Bm_reg)              # (B, P, 3, 3)
    eps = 0.5 * (F_grad.transpose(-1, -2) @ F_grad - eye3)
    eps_frob = eps.flatten(-2).norm(dim=-1, keepdim=True)              # (B, P, 1)
    eps_trace = eps.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)   # (B, P, 1)
    s = torch.cat([
        torch.sign(eps_frob)  * torch.log1p(eps_frob.abs()),
        torch.sign(eps_trace) * torch.log1p(eps_trace.abs()),
    ], dim=-1)                                                         # (B, P, 2)
    return s.clamp(min=-3.0, max=3.0)


class MotionStrain(Motion):
    """PMamba + per-point Green-Lagrange strain as 2-ch stage1 input."""

    def __init__(self, num_classes, pts_size, strain_knn=12, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.strain_knn = strain_knn
        # 6-ch input for stage1 [xyz, strain_frob, strain_trace, t]
        self.stage1 = MLPBlock([6, 32, 64], 2)

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

        # Compute strain field per-frame-pair (needs correspondence)
        with torch.no_grad():
            strain = torch.zeros(batchsize, 2, timestep, pts_num,
                                 device=xyz.device, dtype=xyz.dtype)
            if has_corr and corr_matched is not None:
                xyz_p = xyz.permute(0, 2, 3, 1)               # (B, T, P, 3)
                for t in range(timestep - 1):
                    s = _strain_field(
                        xyz_p[:, t], xyz_p[:, t + 1], k=self.strain_knn,
                    )                                          # (B, P, 2)
                    strain[:, :, t + 1] = s.permute(0, 2, 1)   # (B, 2, P)
                # invalid points: zero out strain
                if corr_matched is not None:
                    valid_mask = corr_matched.float()          # (B, T-1, P)
                    strain[:, :, 1:] = strain[:, :, 1:] * valid_mask.unsqueeze(1)

        coords6 = torch.cat([xyz, strain, time_ch], dim=1)    # (B, 6, T, P)

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords6, array2=coords6,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, 6, timestep * pts_num, -1)
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
print("added MotionStrain to models/motion.py")
