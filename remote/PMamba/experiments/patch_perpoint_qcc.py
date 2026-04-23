"""Q2 — Add PerPointQCCBearingMotion to models/reqnn_motion.py.

Per QUATERNION_GAPS_PLAN.md Q2. Dense per-point quaternion field with
observable shortest-arc anchor + ARAP-style neighbor smoothness.

- Per-point quaternion head: Conv1d(feat*2, feat) -> Conv1d(feat, 4), normalize.
- Input: concat of (feat_t, feat_{t+1}) per correspondence-aligned point.
- Observable anchor q_obs per point = shortest-arc between (p_i - c_t) and
  (p_i - c_{t+1}) centroid-relative directions. Non-trivial, observable.
- L_anchor = 1 - (q_pred . q_obs)^2 mean over points and pairs.
- L_arap  = pairwise sign-corrected MSE of q_pred across kNN neighbors at
  frame t. Encourages spatially smooth rotation field (rigid-body-like).
- Total aux = anchor_weight * L_anchor + arap_weight * L_arap.

Dense target (B * (F-1) * P quaternion labels) — expected to shape features
more than the per-pair head in Q1 (which plateaued at 79.25 vs velpolar 79.67).
"""
from pathlib import Path

SRC = Path("models/reqnn_motion.py")
src = SRC.read_text(encoding="utf-8")
if "class PerPointQCCBearingMotion" in src:
    print("PerPointQCCBearingMotion already present")
else:
    snippet = '''

def _shortest_arc_quaternion(v0, v1):
    """Shortest-arc unit quaternion rotating unit vector v0 to v1.

    v0, v1: (..., 3). Returns (..., 4) in [w,x,y,z]. Handles v0=-v1 degenerate
    by picking a perpendicular axis.
    """
    dot = (v0 * v1).sum(dim=-1, keepdim=True)                 # (..., 1)
    cross = torch.linalg.cross(v0, v1, dim=-1)                # (..., 3)
    w = 1.0 + dot
    q = torch.cat([w, cross], dim=-1)                         # (..., 4)

    degenerate = (dot.squeeze(-1) < -0.9999)
    if degenerate.any():
        ax = torch.zeros_like(v0)
        v0_abs = v0.abs()
        # Pick x-axis unless v0 is ~parallel to x; in that case use y.
        use_y = v0_abs[..., 0] > v0_abs[..., 1]
        ax[..., 0] = (~use_y).to(ax.dtype)
        ax[..., 1] = use_y.to(ax.dtype)
        fallback = torch.cat([torch.zeros_like(w), ax], dim=-1)
        q = torch.where(degenerate.unsqueeze(-1), fallback, q)

    return F.normalize(q, dim=-1)


class _PerPointQuaternionFieldLoss(nn.Module):
    """Dense per-point quaternion field with observable anchor + ARAP smoothness."""

    def __init__(self, feat_dim, arap_knn=8):
        super().__init__()
        self.arap_knn = arap_knn
        self.quat_head = nn.Sequential(
            nn.Conv1d(feat_dim * 2, feat_dim, 1),
            nn.GELU(),
            nn.Conv1d(feat_dim, 4, 1),
        )

    def forward(self, encoded, num_frames, pts_per_frame, points_xyz):
        """encoded: (B, feat_dim, F*P); points_xyz: (B, F, P, 3)."""
        B, feat_dim, _ = encoded.shape
        device = encoded.device
        feat = encoded.permute(0, 2, 1).reshape(
            B, num_frames, pts_per_frame, feat_dim,
        )                                                     # (B, F, P, feat)

        centroid = points_xyz.mean(dim=2, keepdim=True)       # (B, F, 1, 3)
        rel = points_xyz - centroid
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rel_u = rel / rel_norm                                # (B, F, P, 3)

        anchor_sum = torch.tensor(0.0, device=device)
        arap_sum = torch.tensor(0.0, device=device)
        n_pairs = 0

        for t in range(num_frames - 1):
            f_cat = torch.cat([feat[:, t], feat[:, t + 1]], dim=-1)  # (B, P, 2*feat)
            q_pred = self.quat_head(f_cat.transpose(1, 2)).transpose(1, 2)  # (B, P, 4)
            q_pred = F.normalize(q_pred, dim=-1)

            with torch.no_grad():
                q_obs = _shortest_arc_quaternion(rel_u[:, t], rel_u[:, t + 1])

            cos = (q_pred * q_obs).sum(dim=-1)                # (B, P)
            anchor_sum = anchor_sum + (1.0 - cos ** 2).mean()

            # ARAP: kNN over frame-t centroid-relative positions. Neighbor
            # quaternions should be sign-corrected close (rigid object local
            # consistency).
            k = min(self.arap_knn, pts_per_frame - 1)
            pos = rel[:, t]                                    # (B, P, 3)
            d2 = ((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2).sum(dim=-1)  # (B,P,P)
            _, nn_idx = d2.topk(k + 1, dim=-1, largest=False)  # includes self
            nn_idx = nn_idx[..., 1:]                           # drop self -> (B,P,k)

            # Gather neighbor quaternions.
            nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 4)
            q_nn = torch.gather(
                q_pred.unsqueeze(2).expand(-1, -1, k, -1),
                1, nn_idx_exp,
            )                                                  # (B, P, k, 4)
            q_ref = q_pred.unsqueeze(2).expand(-1, -1, k, -1)  # (B, P, k, 4)

            # Sign-correct neighbor quats before MSE.
            sign = torch.sign((q_ref * q_nn).sum(dim=-1, keepdim=True))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            q_nn_aligned = q_nn * sign
            arap_sum = arap_sum + ((q_ref - q_nn_aligned) ** 2).sum(dim=-1).mean()

            n_pairs += 1

        anchor_loss = anchor_sum / max(n_pairs, 1)
        arap_loss = arap_sum / max(n_pairs, 1)

        metrics = {
            "anchor_raw": anchor_loss.detach(),
            "arap_raw": arap_loss.detach(),
        }
        return anchor_loss, arap_loss, metrics


class PerPointQCCBearingMotion(VelocityPolarBearingQCCFeatureMotion):
    """Velpolar + dense per-point quaternion field (Q2)."""

    def __init__(self, *args, anchor_weight=0.1, arap_weight=0.02,
                 arap_knn=8, hidden_dims=(64, 256), **kwargs):
        kwargs.setdefault("qcc_weight", 0.0)
        super().__init__(*args, hidden_dims=hidden_dims, **kwargs)
        _, hidden2 = hidden_dims
        self.anchor_weight = anchor_weight
        self.arap_weight = arap_weight
        self.perpoint_qcc = _PerPointQuaternionFieldLoss(
            feat_dim=hidden2, arap_knn=arap_knn,
        )

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
        if self.training and (self.anchor_weight > 0 or self.arap_weight > 0):
            anchor_loss, arap_loss, metrics = self.perpoint_qcc(
                encoded, F_, P, xyz,
            )
            total = (self.anchor_weight * anchor_loss
                     + self.arap_weight * arap_loss)
            metrics["qcc_raw"] = total.detach()
            metrics["qcc_forward"] = anchor_loss.detach()
            metrics["qcc_backward"] = arap_loss.detach()
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
    print("appended PerPointQCCBearingMotion to models/reqnn_motion.py")
