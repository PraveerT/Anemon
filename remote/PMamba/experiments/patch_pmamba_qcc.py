"""Add MotionQCCAnchored: PMamba + anchored quaternion-cycle-consistency aux.

Q1-style but on PMamba backbone:
- Per-frame pooled features from stage1 output
- Pair-quaternion head: predict q(t, t+1) from [f_t, f_t+1] concat
- L_anchor = 1 - (q_pred · q_obs)^2 (sign-ambiguous), q_obs from correspondence-
  aligned Kabsch → non-trivial target, no collapse (Mittal-style anchoring)
- L_trans = 1 - cos^2 between q_pred(t, t+2) and q(t,t+1) ⊗ q(t+1,t+2)

Uses NvidiaQuaternionQCCParityLoader + correspondence-guided sampling.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionQCCAnchored" in src:
    start = src.find("\n\nclass MotionQCCAnchored")
    src = src[:start]

snippet = '''

def _qcc_hamilton(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _qcc_rot_to_quat(R):
    """Batched rotation matrix -> quaternion [w,x,y,z], Shepperd-style."""
    orig_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    B = R_flat.shape[0]
    device = R_flat.device
    dtype = R_flat.dtype

    m00, m01, m02 = R_flat[:, 0, 0], R_flat[:, 0, 1], R_flat[:, 0, 2]
    m10, m11, m12 = R_flat[:, 1, 0], R_flat[:, 1, 1], R_flat[:, 1, 2]
    m20, m21, m22 = R_flat[:, 2, 0], R_flat[:, 2, 1], R_flat[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(B, 4, device=device, dtype=dtype)

    m1 = tr > 0
    if m1.any():
        s = torch.sqrt(tr[m1].clamp(min=-0.999) + 1.0) * 2.0
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (m21[m1] - m12[m1]) / s
        q[m1, 2] = (m02[m1] - m20[m1]) / s
        q[m1, 3] = (m10[m1] - m01[m1]) / s

    rem = ~m1
    m2a = rem & (m00 > m11) & (m00 > m22)
    if m2a.any():
        s = torch.sqrt(1.0 + m00[m2a] - m11[m2a] - m22[m2a]).clamp(min=1e-8) * 2.0
        q[m2a, 0] = (m21[m2a] - m12[m2a]) / s
        q[m2a, 1] = 0.25 * s
        q[m2a, 2] = (m01[m2a] + m10[m2a]) / s
        q[m2a, 3] = (m02[m2a] + m20[m2a]) / s

    m2b = rem & (~m2a) & (m11 > m22)
    if m2b.any():
        s = torch.sqrt(1.0 + m11[m2b] - m00[m2b] - m22[m2b]).clamp(min=1e-8) * 2.0
        q[m2b, 0] = (m02[m2b] - m20[m2b]) / s
        q[m2b, 1] = (m01[m2b] + m10[m2b]) / s
        q[m2b, 2] = 0.25 * s
        q[m2b, 3] = (m12[m2b] + m21[m2b]) / s

    m2c = rem & (~m2a) & (~m2b)
    if m2c.any():
        s = torch.sqrt(1.0 + m22[m2c] - m00[m2c] - m11[m2c]).clamp(min=1e-8) * 2.0
        q[m2c, 0] = (m10[m2c] - m01[m2c]) / s
        q[m2c, 1] = (m02[m2c] + m20[m2c]) / s
        q[m2c, 2] = (m12[m2c] + m21[m2c]) / s
        q[m2c, 3] = 0.25 * s

    q = F.normalize(q, dim=-1)
    return q.reshape(*orig_shape, 4)


def _qcc_kabsch_quat(src, tgt, weights=None):
    """Batched Kabsch -> unit quaternion. src, tgt: (..., N, 3)."""
    shape = src.shape[:-2]
    N = src.shape[-2]
    src_f = src.reshape(-1, N, 3)
    tgt_f = tgt.reshape(-1, N, 3)
    B = src_f.shape[0]
    device = src_f.device

    if weights is not None:
        w = weights.reshape(-1, N).float()
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        src_mean = (src_f * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
        tgt_mean = (tgt_f * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
        src_c = src_f - src_mean
        tgt_c = tgt_f - tgt_mean
        H = torch.einsum("bn,bni,bnj->bij", w, src_c, tgt_c)
    else:
        src_c = src_f - src_f.mean(1, keepdim=True)
        tgt_c = tgt_f - tgt_f.mean(1, keepdim=True)
        H = torch.einsum("bni,bnj->bij", src_c, tgt_c)

    H = H + 1e-6 * torch.eye(3, device=device, dtype=H.dtype).unsqueeze(0)
    try:
        U, S, Vh = torch.linalg.svd(H)
    except Exception:
        R = torch.eye(3, device=device, dtype=H.dtype).unsqueeze(0).expand(B, 3, 3).contiguous()
        return _qcc_rot_to_quat(R).reshape(*shape, 4)

    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    bad = ~torch.isfinite(R).all(dim=-1).all(dim=-1)
    if bad.any():
        eye = torch.eye(3, device=device, dtype=R.dtype).unsqueeze(0).expand_as(R)
        R = torch.where(bad.unsqueeze(-1).unsqueeze(-1), eye, R)
    q = _qcc_rot_to_quat(R)
    return q.reshape(*shape, 4)


class MotionQCCAnchored(Motion):
    """PMamba + Mittal-anchored quaternion-pair + transitivity aux.

    Features: stage1 output pooled per-frame. Head predicts q(t, t+1); anchor
    is Kabsch q_obs from correspondence-aligned sampled points.
    """

    def __init__(self, num_classes, pts_size, qcc_weight=0.05,
                 anchor_weight=1.0, trans_weight=0.5, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.qcc_weight = qcc_weight
        self.anchor_weight = anchor_weight
        self.trans_weight = trans_weight
        # Stage1 per-point dim is 64 (after pool1 + cat with coords=68, but we
        # use raw stage1 output before cat, which is 64).
        feat_dim = 64
        self.qcc_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4),
        )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _correspondence_guided_sample(self, points, aux_input):
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device
        if sample_size == pts_per_frame:
            corr_matched = torch.ones(batch_size, num_frames - 1, pts_per_frame,
                                      dtype=torch.bool, device=device)
            return points, corr_matched

        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames

        sampled = torch.zeros(batch_size, num_frames, sample_size, channels,
                              device=device, dtype=points.dtype)
        corr_matched = torch.zeros(batch_size, num_frames - 1, sample_size,
                                   dtype=torch.bool, device=device)

        for b in range(batch_size):
            if self.training:
                idx = torch.randperm(pts_per_frame, device=device)[:sample_size]
            else:
                idx = torch.linspace(0, pts_per_frame - 1, sample_size,
                                     device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()
            for t in range(num_frames - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(pts_per_frame, device=device)
                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]
                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))
                next_idx = torch.randint(0, pts_per_frame, (sample_size,), device=device)
                next_idx[valid] = tgt_pos[valid]
                sampled[b, t + 1] = points[b, t + 1, next_idx]
                corr_matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()
        return sampled, corr_matched

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
            aux_input = inputs
            has_corr = ("orig_flat_idx" in aux_input
                        and "corr_full_target_idx" in aux_input
                        and "corr_full_weight" in aux_input)
        else:
            points_raw = inputs
            aux_input = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._correspondence_guided_sample(
                points_raw[..., :4], aux_input,
            )
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
        )                                                        # (B, 64, T, P)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.qcc_weight > 0 and has_corr:
            q_loss, q_metrics = self._qcc_aux(fea1_raw, coords[:, :3], corr_matched)
            self.latest_aux_loss = self.qcc_weight * q_loss
            q_metrics["qcc_raw"] = q_loss.detach()
            q_metrics["qcc_forward"] = q_metrics.get("anchor_raw", q_loss.detach())
            q_metrics["qcc_backward"] = q_metrics.get("trans_raw", q_loss.detach())
            q_metrics["qcc_valid_ratio"] = corr_matched.float().mean().detach()
            self.latest_aux_metrics = q_metrics

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

    def _qcc_aux(self, features, xyz, corr_matched):
        """features (B, C, T, P), xyz (B, 3, T, P). Per-frame pooled features
        predict pair quaternions; anchor via Kabsch on corr-aligned xyz."""
        B, C, T, P = features.shape
        device = features.device
        feat = features.mean(dim=-1).transpose(1, 2)             # (B, T, C)
        xyz_p = xyz.permute(0, 2, 3, 1)                           # (B, T, P, 3)

        def pred(f_src, f_tgt):
            q = self.qcc_head(torch.cat([f_src, f_tgt], dim=-1))
            return F.normalize(q, dim=-1)

        q_preds, q_obs = [], []
        for t in range(T - 1):
            q_preds.append(pred(feat[:, t], feat[:, t + 1]))
            with torch.no_grad():
                w = corr_matched[:, t].float()
                q_obs.append(_qcc_kabsch_quat(xyz_p[:, t], xyz_p[:, t + 1], w))
        q_preds_t = torch.stack(q_preds, dim=1)                   # (B, T-1, 4)
        q_obs_t = torch.stack(q_obs, dim=1)

        cos = (q_preds_t * q_obs_t).sum(dim=-1)
        anchor_loss = (1.0 - cos ** 2).mean()

        trans_loss = features.new_zeros(())
        n_trip = 0
        for t in range(T - 2):
            q_skip = pred(feat[:, t], feat[:, t + 2])
            q_comp = _qcc_hamilton(q_preds_t[:, t], q_preds_t[:, t + 1])
            cos_t = (q_skip * q_comp).sum(dim=-1)
            trans_loss = trans_loss + (1.0 - cos_t ** 2).mean()
            n_trip += 1
        if n_trip > 0:
            trans_loss = trans_loss / n_trip

        total = self.anchor_weight * anchor_loss + self.trans_weight * trans_loss
        return total, {
            "anchor_raw": anchor_loss.detach(),
            "trans_raw": trans_loss.detach(),
            "q_cos_mean": cos.abs().mean().detach(),
        }
'''
SRC_NEW = src.rstrip() + snippet + "\n"
MOTION.write_text(SRC_NEW, encoding="utf-8")
print("added MotionQCCAnchored to models/motion.py")
