"""Q1 — Add AnchoredQCCBearingMotion to models/reqnn_motion.py.

Per QUATERNION_GAPS_PLAN.md Q1. Extends VelocityPolarBearingQCCFeatureMotion
(velpolar 79.67% solo baseline) with:
  - Pair-quaternion head predicting q(t, t+1) from concatenated per-frame
    pooled features.
  - L_anchor: 1 - (q_pred . q_obs)^2 where q_obs is Kabsch quaternion from
    correspondence-aligned points (observable anchor — no collapse).
  - L_trans: for each triplet (t, t+1, t+2), 1 - (q_pred(t,t+2) .
    (q_pred(t,t+1) o q_pred(t+1,t+2)))^2.
  - Total aux = anchor_weight * L_anchor + trans_weight * L_trans.

Loss target q_obs is non-trivial and observable; cycle/transitivity rides on
top as regulariser. Mittal-style anchoring.
"""
from pathlib import Path

SRC = Path("models/reqnn_motion.py")
src = SRC.read_text(encoding="utf-8")
if "class AnchoredQCCBearingMotion" in src:
    print("AnchoredQCCBearingMotion already present")
else:
    snippet = '''

def _rotation_matrix_to_quaternion(R):
    """Batched rotation matrix (..., 3, 3) -> unit quaternion (..., 4) [w,x,y,z].

    Shepperd-style: pick the largest diagonal term for numerical stability.
    """
    orig_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    B = R_flat.shape[0]
    device = R_flat.device
    dtype = R_flat.dtype

    m00 = R_flat[:, 0, 0]; m01 = R_flat[:, 0, 1]; m02 = R_flat[:, 0, 2]
    m10 = R_flat[:, 1, 0]; m11 = R_flat[:, 1, 1]; m12 = R_flat[:, 1, 2]
    m20 = R_flat[:, 2, 0]; m21 = R_flat[:, 2, 1]; m22 = R_flat[:, 2, 2]
    trace = m00 + m11 + m22

    q = torch.zeros(B, 4, device=device, dtype=dtype)

    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1].clamp(min=-0.999) + 1.0) * 2.0
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (m21[mask1] - m12[mask1]) / s
        q[mask1, 2] = (m02[mask1] - m20[mask1]) / s
        q[mask1, 3] = (m10[mask1] - m01[mask1]) / s

    rem = ~mask1
    mask2a = rem & (m00 > m11) & (m00 > m22)
    if mask2a.any():
        s = torch.sqrt(1.0 + m00[mask2a] - m11[mask2a] - m22[mask2a]).clamp(min=1e-8) * 2.0
        q[mask2a, 0] = (m21[mask2a] - m12[mask2a]) / s
        q[mask2a, 1] = 0.25 * s
        q[mask2a, 2] = (m01[mask2a] + m10[mask2a]) / s
        q[mask2a, 3] = (m02[mask2a] + m20[mask2a]) / s

    mask2b = rem & (~mask2a) & (m11 > m22)
    if mask2b.any():
        s = torch.sqrt(1.0 + m11[mask2b] - m00[mask2b] - m22[mask2b]).clamp(min=1e-8) * 2.0
        q[mask2b, 0] = (m02[mask2b] - m20[mask2b]) / s
        q[mask2b, 1] = (m01[mask2b] + m10[mask2b]) / s
        q[mask2b, 2] = 0.25 * s
        q[mask2b, 3] = (m12[mask2b] + m21[mask2b]) / s

    mask2c = rem & (~mask2a) & (~mask2b)
    if mask2c.any():
        s = torch.sqrt(1.0 + m22[mask2c] - m00[mask2c] - m11[mask2c]).clamp(min=1e-8) * 2.0
        q[mask2c, 0] = (m10[mask2c] - m01[mask2c]) / s
        q[mask2c, 1] = (m02[mask2c] + m20[mask2c]) / s
        q[mask2c, 2] = (m12[mask2c] + m21[mask2c]) / s
        q[mask2c, 3] = 0.25 * s

    q = F.normalize(q, dim=-1)
    return q.reshape(*orig_shape, 4)


def _batched_kabsch_quaternion(src, tgt):
    """Batched Kabsch -> quaternion.

    src, tgt: (..., N, 3) correspondence-aligned point sets.
    Returns unit quaternion (..., 4) rotating src to tgt (rigid, best-fit).
    """
    shape = src.shape[:-2]
    N = src.shape[-2]
    src_f = src.reshape(-1, N, 3)
    tgt_f = tgt.reshape(-1, N, 3)
    B = src_f.shape[0]
    device = src_f.device

    src_c = src_f - src_f.mean(dim=1, keepdim=True)
    tgt_c = tgt_f - tgt_f.mean(dim=1, keepdim=True)
    H = torch.einsum("bni,bnj->bij", src_c, tgt_c)
    H = H + 1e-6 * torch.eye(3, device=device, dtype=H.dtype).unsqueeze(0)

    try:
        U, S, Vh = torch.linalg.svd(H)
    except Exception:
        R = torch.eye(3, device=device, dtype=H.dtype).unsqueeze(0).expand(B, 3, 3).contiguous()
        return _rotation_matrix_to_quaternion(R).reshape(*shape, 4)

    V = Vh.transpose(-1, -2)
    det = torch.det(torch.matmul(V, U.transpose(-1, -2)))
    D_diag = torch.stack(
        [torch.ones_like(det), torch.ones_like(det), det], dim=-1,
    )
    D = torch.diag_embed(D_diag)
    R = torch.matmul(V, torch.matmul(D, U.transpose(-1, -2)))
    bad = ~torch.isfinite(R).all(dim=-1).all(dim=-1)
    if bad.any():
        R = torch.where(
            bad.unsqueeze(-1).unsqueeze(-1),
            torch.eye(3, device=device, dtype=R.dtype).unsqueeze(0).expand_as(R),
            R,
        )
    q = _rotation_matrix_to_quaternion(R)
    return q.reshape(*shape, 4)


class _AnchoredQuaternionPairLoss(nn.Module):
    """Predict q(t, t+1) from per-frame pooled features; anchor to Kabsch q_obs.

    Also predicts q(t, t+2) and pushes it toward q(t,t+1) o q(t+1,t+2) via
    a sign-ambiguous cos^2 transitivity penalty.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.quat_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4),
        )

    def _predict_pair(self, f_src, f_tgt):
        q = self.quat_head(torch.cat([f_src, f_tgt], dim=-1))
        return F.normalize(q, dim=-1)

    def forward(self, encoded, num_frames, pts_per_frame, points_xyz):
        # encoded:   (B, feat_dim, num_points) — num_points = F * P
        # points_xyz: (B, F, P, 3) correspondence-aligned
        B, feat_dim, _ = encoded.shape
        device = encoded.device
        feat = (encoded
                .permute(0, 2, 1)
                .reshape(B, num_frames, pts_per_frame, feat_dim)
                .mean(dim=2))                              # (B, F, feat_dim)

        # Consecutive pairs.
        q_consec = []
        q_obs = []
        for t in range(num_frames - 1):
            q_consec.append(self._predict_pair(feat[:, t], feat[:, t + 1]))
            with torch.no_grad():
                q_obs.append(_batched_kabsch_quaternion(
                    points_xyz[:, t], points_xyz[:, t + 1]))
        q_consec_t = torch.stack(q_consec, dim=1)          # (B, F-1, 4)
        q_obs_t = torch.stack(q_obs, dim=1)                # (B, F-1, 4)

        # Anchor loss: 1 - cos^2 (sign-ambiguous).
        cos = (q_consec_t * q_obs_t).sum(dim=-1)
        anchor_loss = (1.0 - cos ** 2).mean()

        # Transitivity loss over skip-2 pairs.
        trans_loss = torch.tensor(0.0, device=device)
        n_trip = 0
        for t in range(num_frames - 2):
            q_skip = self._predict_pair(feat[:, t], feat[:, t + 2])
            q_comp = _hamilton_product(q_consec_t[:, t], q_consec_t[:, t + 1])
            cos_t = (q_skip * q_comp).sum(dim=-1)
            trans_loss = trans_loss + (1.0 - cos_t ** 2).mean()
            n_trip += 1
        if n_trip > 0:
            trans_loss = trans_loss / n_trip

        metrics = {
            "anchor_raw": anchor_loss.detach(),
            "trans_raw": trans_loss.detach(),
            "q_cos_mean": cos.abs().mean().detach(),
        }
        return anchor_loss, trans_loss, metrics


class AnchoredQCCBearingMotion(VelocityPolarBearingQCCFeatureMotion):
    """Velpolar + anchored quaternion-pair supervision + transitivity (Q1).

    Loss: CE + anchor_weight * L_anchor + trans_weight * L_trans.
    Internal rigidity/correspondence sampling unchanged from velpolar.
    """

    def __init__(self, *args, anchor_weight=0.1, trans_weight=0.05,
                 hidden_dims=(64, 256), **kwargs):
        kwargs.setdefault("qcc_weight", 0.0)
        super().__init__(*args, hidden_dims=hidden_dims, **kwargs)
        _, hidden2 = hidden_dims
        self.anchor_weight = anchor_weight
        self.trans_weight = trans_weight
        self.anchored_qcc = _AnchoredQuaternionPairLoss(feat_dim=hidden2)

    def extract_features(self, inputs, aux_input=None):
        if isinstance(inputs, dict):
            points = inputs["points"]
            aux_unpacked = inputs
        else:
            points = inputs
            aux_unpacked = None

        has_corr = (aux_unpacked is not None
                    and "orig_flat_idx" in aux_unpacked
                    and "corr_full_target_idx" in aux_unpacked
                    and "corr_full_weight" in aux_unpacked)

        if has_corr and not self.decouple_sampling:
            sampled, corr_matched = self._correspondence_guided_sample(
                points[..., :4], aux_unpacked)
        else:
            sampled = self._sample_points(points[..., :4])
            corr_matched = None

        sampled = sampled[..., :4]
        B, F_, P, _ = sampled.shape

        rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
            sampled, F_, knn_k=self.bearing_knn_k,
            corr_matched=corr_matched,
        )

        xyz = sampled[..., :3]
        time_ch = sampled[..., 3:4]
        centroid = xyz.mean(dim=2, keepdim=True)
        rel = xyz - centroid
        magnitude = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = (rel / magnitude).detach()
        vel = torch.zeros_like(xyz)
        vel[:, :-1] = xyz[:, 1:] - xyz[:, :-1]
        vel[:, -1] = xyz[:, -1] - xyz[:, -2]

        sampled_8 = torch.cat([vel, direction, magnitude, time_ch], dim=-1)
        point_features = sampled_8.reshape(B, -1, 8).transpose(1, 2).contiguous()
        encoded = self._encode_to_pre_merge(point_features)

        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            encoded = encoded * (1.0 + modulation)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and (self.anchor_weight > 0 or self.trans_weight > 0):
            anchor_loss, trans_loss, metrics = self.anchored_qcc(
                encoded, F_, P, xyz,
            )
            total = (self.anchor_weight * anchor_loss
                     + self.trans_weight * trans_loss)
            metrics["qcc_raw"] = total.detach()
            metrics["qcc_forward"] = anchor_loss.detach()
            metrics["qcc_backward"] = trans_loss.detach()
            metrics["qcc_valid_ratio"] = torch.tensor(
                corr_valid_ratio, device=encoded.device)
            self.latest_aux_loss = total
            self.latest_aux_metrics = metrics

        encoded = self.merge_proj(self.merge_quaternions(encoded))
        pooled_max = encoded.max(dim=-1).values
        attention = torch.softmax(self.readout_attention(encoded), dim=-1)
        pooled_attn = torch.sum(encoded * attention, dim=-1)
        return torch.cat((pooled_max, pooled_attn), dim=1)
'''
    src = src.rstrip() + snippet + "\n"
    SRC.write_text(src, encoding="utf-8")
    print("appended AnchoredQCCBearingMotion to models/reqnn_motion.py")
