"""Q3 — Add DualQuaternionQCCBearingMotion to models/reqnn_motion.py.

Per QUATERNION_GAPS_PLAN.md Q3. Extends Q1 to SE(3): predict dual quaternion
DQ = q_r + eps*q_d per frame pair, anchor both components to observable
Kabsch (R, t). Transitivity on DQ composition.

Hypothesis: translation component (q_d) carries gesture info (grip closing,
finger extension, hand translation) that rotation-only anchoring missed.

Reuses: _hamilton_product, _rotation_matrix_to_quaternion (from Q1 patch).
"""
from pathlib import Path

SRC = Path("models/reqnn_motion.py")
src = SRC.read_text(encoding="utf-8")
if "class DualQuaternionQCCBearingMotion" in src:
    print("DualQuaternionQCCBearingMotion already present")
else:
    snippet = '''

def _batched_kabsch_rigid(src, tgt):
    """Batched Kabsch -> (R, t). src, tgt: (..., N, 3). Returns R (..., 3, 3), t (..., 3)."""
    shape = src.shape[:-2]
    N = src.shape[-2]
    src_f = src.reshape(-1, N, 3)
    tgt_f = tgt.reshape(-1, N, 3)
    B = src_f.shape[0]
    device = src_f.device

    src_mean = src_f.mean(dim=1, keepdim=True)
    tgt_mean = tgt_f.mean(dim=1, keepdim=True)
    src_c = src_f - src_mean
    tgt_c = tgt_f - tgt_mean
    H = torch.einsum("bni,bnj->bij", src_c, tgt_c)
    H = H + 1e-6 * torch.eye(3, device=device, dtype=H.dtype).unsqueeze(0)

    try:
        U, S, Vh = torch.linalg.svd(H)
    except Exception:
        R = torch.eye(3, device=device, dtype=H.dtype).unsqueeze(0).expand(B, 3, 3).contiguous()
        t = (tgt_mean - src_mean).squeeze(1)
        return R.reshape(*shape, 3, 3), t.reshape(*shape, 3)

    V = Vh.transpose(-1, -2)
    det = torch.det(torch.matmul(V, U.transpose(-1, -2)))
    D_diag = torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1)
    D = torch.diag_embed(D_diag)
    R = torch.matmul(V, torch.matmul(D, U.transpose(-1, -2)))
    bad = ~torch.isfinite(R).all(dim=-1).all(dim=-1)
    if bad.any():
        R = torch.where(
            bad.unsqueeze(-1).unsqueeze(-1),
            torch.eye(3, device=device, dtype=R.dtype).unsqueeze(0).expand_as(R),
            R,
        )
    t = tgt_mean.squeeze(1) - torch.bmm(R, src_mean.transpose(-1, -2)).squeeze(-1)
    return R.reshape(*shape, 3, 3), t.reshape(*shape, 3)


def _translation_to_dual_quat(t, q_r):
    """Dual part q_d = 0.5 * t_quat * q_r where t_quat = [0, tx, ty, tz]."""
    zero = torch.zeros_like(t[..., :1])
    t_quat = torch.cat([zero, t], dim=-1)
    return 0.5 * _hamilton_product(t_quat, q_r)


def _dq_multiply(p_r, p_d, q_r, q_d):
    """DQ multiplication: (p_r, p_d) * (q_r, q_d) = (p_r*q_r, p_r*q_d + p_d*q_r)."""
    r = _hamilton_product(p_r, q_r)
    d = _hamilton_product(p_r, q_d) + _hamilton_product(p_d, q_r)
    return r, d


class _DualQuaternionPairLoss(nn.Module):
    """Predict dual quaternion (q_r, q_d) per pair from per-frame pooled features.

    Anchor both components to observable Kabsch SE(3). Transitivity over skip-2.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.dq_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 8),
        )

    def _predict_pair(self, f_src, f_tgt):
        dq = self.dq_head(torch.cat([f_src, f_tgt], dim=-1))
        q_r = F.normalize(dq[..., :4], dim=-1)
        q_d = dq[..., 4:]
        # Orthogonality: q_d orthogonal to q_r on unit DQ manifold.
        q_d = q_d - (q_d * q_r).sum(dim=-1, keepdim=True) * q_r
        return q_r, q_d

    def forward(self, encoded, num_frames, pts_per_frame, points_xyz):
        B, feat_dim, _ = encoded.shape
        device = encoded.device
        feat = (encoded
                .permute(0, 2, 1)
                .reshape(B, num_frames, pts_per_frame, feat_dim)
                .mean(dim=2))                                  # (B, F, feat)

        qr_p, qd_p = [], []
        qr_o, qd_o = [], []
        for t in range(num_frames - 1):
            q_r, q_d = self._predict_pair(feat[:, t], feat[:, t + 1])
            qr_p.append(q_r); qd_p.append(q_d)
            with torch.no_grad():
                R, tr = _batched_kabsch_rigid(points_xyz[:, t], points_xyz[:, t + 1])
                q_r_o = _rotation_matrix_to_quaternion(R)
                q_d_o = _translation_to_dual_quat(tr, q_r_o)
            qr_o.append(q_r_o); qd_o.append(q_d_o)

        qr_p = torch.stack(qr_p, dim=1)                        # (B, F-1, 4)
        qd_p = torch.stack(qd_p, dim=1)
        qr_o_t = torch.stack(qr_o, dim=1)
        qd_o_t = torch.stack(qd_o, dim=1)

        # Rotation anchor (sign-ambiguous cos^2).
        cos_r = (qr_p * qr_o_t).sum(dim=-1)                    # (B, F-1)
        anchor_r = (1.0 - cos_r ** 2).mean()

        # Translation anchor: sign-align q_d with the rotation sign choice.
        sign = torch.sign(cos_r).unsqueeze(-1)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        qd_p_signed = qd_p * sign
        anchor_d = F.mse_loss(qd_p_signed, qd_o_t)

        # Transitivity: pred DQ(t, t+2) vs DQ(t,t+1) o DQ(t+1,t+2).
        trans_r = torch.tensor(0.0, device=device)
        trans_d = torch.tensor(0.0, device=device)
        n_trip = 0
        for t in range(num_frames - 2):
            q_r_skip, q_d_skip = self._predict_pair(feat[:, t], feat[:, t + 2])
            q_r_comp, q_d_comp = _dq_multiply(
                qr_p[:, t], qd_p[:, t], qr_p[:, t + 1], qd_p[:, t + 1],
            )
            cos_rt = (q_r_skip * q_r_comp).sum(dim=-1)
            trans_r = trans_r + (1.0 - cos_rt ** 2).mean()
            sgn_t = torch.sign(cos_rt).unsqueeze(-1)
            sgn_t = torch.where(sgn_t == 0, torch.ones_like(sgn_t), sgn_t)
            trans_d = trans_d + F.mse_loss(q_d_skip * sgn_t, q_d_comp)
            n_trip += 1
        if n_trip > 0:
            trans_r = trans_r / n_trip
            trans_d = trans_d / n_trip

        metrics = {
            "anchor_r": anchor_r.detach(),
            "anchor_d": anchor_d.detach(),
            "trans_r": trans_r.detach(),
            "trans_d": trans_d.detach(),
            "cos_r_mean": cos_r.abs().mean().detach(),
        }
        return anchor_r, anchor_d, trans_r, trans_d, metrics


class DualQuaternionQCCBearingMotion(VelocityPolarBearingQCCFeatureMotion):
    """Velpolar + dual-quaternion SE(3) anchor + transitivity (Q3)."""

    def __init__(self, *args,
                 anchor_r_weight=0.1, anchor_d_weight=0.05,
                 trans_r_weight=0.05, trans_d_weight=0.02,
                 hidden_dims=(64, 256), **kwargs):
        kwargs.setdefault("qcc_weight", 0.0)
        super().__init__(*args, hidden_dims=hidden_dims, **kwargs)
        _, hidden2 = hidden_dims
        self.anchor_r_weight = anchor_r_weight
        self.anchor_d_weight = anchor_d_weight
        self.trans_r_weight = trans_r_weight
        self.trans_d_weight = trans_d_weight
        self.dq_qcc = _DualQuaternionPairLoss(feat_dim=hidden2)

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
                points[..., :4], aux_unpacked,
            )
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
        any_weight = (self.anchor_r_weight > 0 or self.anchor_d_weight > 0
                      or self.trans_r_weight > 0 or self.trans_d_weight > 0)
        if self.training and any_weight:
            a_r, a_d, t_r, t_d, metrics = self.dq_qcc(encoded, F_, P, xyz)
            total = (self.anchor_r_weight * a_r
                     + self.anchor_d_weight * a_d
                     + self.trans_r_weight * t_r
                     + self.trans_d_weight * t_d)
            metrics["qcc_raw"] = total.detach()
            metrics["qcc_forward"] = a_r.detach()
            metrics["qcc_backward"] = a_d.detach()
            metrics["qcc_valid_ratio"] = torch.tensor(
                corr_valid_ratio, device=encoded.device,
            )
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
    print("appended DualQuaternionQCCBearingMotion to models/reqnn_motion.py")
