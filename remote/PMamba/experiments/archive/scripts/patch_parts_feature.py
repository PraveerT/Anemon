"""Parts-as-feature (v18a): K=6 rotation features, NO aux loss.

Key shift from v16a/v17a: instead of using the K rotations to drive an
auxiliary loss (rigidity residual or cycle drift), we compute them purely
as a feature that modulates the classifier. The gradient to learn the
soft-assignment comes only from classification, flowing back through the
differentiable weighted Procrustes SVD.

Module outputs per frame-pair per part:
  - quaternion (4 numbers, derived from Procrustes)
  - rigidity residual (1 number, scalar fit quality)
Concatenated to 5 features per part → K*5 = 30 features per frame-pair.
Mean-pooled over F-1 pairs → 30 features per sample → projected → modulates
encoded multiplicatively.

Only regularization: a tiny entropy-collapse penalty on the averaged
assignment distribution (so parts stay distinct). No reconstruction loss,
no cycle loss, no rigidity loss.

Activate via:
  qcc_variant: parts_feature
  qcc_weight: 1.0       (scales the entropy reg; internal weight is tiny)

Paired with Hungarian correspondence (assignment_mode: hungarian).
"""
import re
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

module_code = '''
class _PartsFeatureProcrustes(nn.Module):
    """K-part Procrustes rotation + rigidity residual, used as features only.

    Returns (aux_loss, features):
      aux_loss: scalar entropy-collapse penalty (tiny) -- keeps parts distinct.
      features: (B, F-1, K, 5) with [qw, qx, qy, qz, rigidity_residual] per part.
    """

    def __init__(self, feat_dim, num_parts=6, entropy_weight=0.01):
        super().__init__()
        self.num_parts = num_parts
        self.entropy_weight = entropy_weight
        self.assign_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, num_parts),
        )

    @staticmethod
    def _rot_to_quat(R):
        m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
        m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
        m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
        tr = m00 + m11 + m22
        eps = 1e-8
        s1 = torch.sqrt(torch.clamp(1 + tr, min=eps)) * 2
        q1 = torch.stack([0.25 * s1, (m21 - m12) / s1, (m02 - m20) / s1, (m10 - m01) / s1], dim=-1)
        s2 = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=eps)) * 2
        q2 = torch.stack([(m21 - m12) / s2, 0.25 * s2, (m01 + m10) / s2, (m02 + m20) / s2], dim=-1)
        s3 = torch.sqrt(torch.clamp(1 + m11 - m00 - m22, min=eps)) * 2
        q3 = torch.stack([(m02 - m20) / s3, (m01 + m10) / s3, 0.25 * s3, (m12 + m21) / s3], dim=-1)
        s4 = torch.sqrt(torch.clamp(1 + m22 - m00 - m11, min=eps)) * 2
        q4 = torch.stack([(m10 - m01) / s4, (m02 + m20) / s4, (m12 + m21) / s4, 0.25 * s4], dim=-1)
        cond1 = tr > 0
        cond2 = (m00 >= m11) & (m00 >= m22)
        cond3 = m11 >= m22
        q_nt = torch.where(cond2.unsqueeze(-1), q2,
                           torch.where(cond3.unsqueeze(-1), q3, q4))
        q = torch.where(cond1.unsqueeze(-1), q1, q_nt)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=eps)

    def forward(self, encoded, points_xyz_flat, num_frames, pts_per_frame, corr_matched):
        B, D, _ = encoded.shape
        F_, P = num_frames, pts_per_frame
        K = self.num_parts
        device = encoded.device

        feat = encoded.transpose(1, 2).contiguous()
        logits = self.assign_head(feat)                  # (B, F*P, K)
        assign = torch.softmax(logits, dim=-1).view(B, F_, P, K)

        xyz = points_xyz_flat.view(B, F_, P, 3)

        feats_list = []
        I3 = torch.eye(3, device=device).view(1, 1, 3, 3)

        for t in range(F_ - 1):
            src = xyz[:, t]
            tgt = xyz[:, t + 1]
            mask = corr_matched[:, t].float() if corr_matched is not None \\
                else torch.ones(B, P, device=device)
            sa = assign[:, t]
            ta = assign[:, t + 1]

            w_k = (sa * ta).permute(0, 2, 1) * mask.unsqueeze(1)       # (B, K, P)
            w_sum = w_k.sum(dim=-1, keepdim=True).clamp(min=1e-6)      # (B, K, 1)

            src_b = src.unsqueeze(1).expand(B, K, P, 3)
            tgt_b = tgt.unsqueeze(1).expand(B, K, P, 3)
            src_mean = (w_k.unsqueeze(-1) * src_b).sum(dim=-2) / w_sum
            tgt_mean = (w_k.unsqueeze(-1) * tgt_b).sum(dim=-2) / w_sum
            src_c = src_b - src_mean.unsqueeze(-2)
            tgt_c = tgt_b - tgt_mean.unsqueeze(-2)

            H = torch.einsum("bkp,bkpi,bkpj->bkij", w_k, src_c, tgt_c)
            H = H + 1e-6 * I3

            try:
                U, S, Vh = torch.linalg.svd(H)
            except Exception:
                R_used = I3.expand(B, K, 3, 3).contiguous()
                quats = self._rot_to_quat(R_used)
                residuals = torch.zeros(B, K, device=device)
                feats_list.append(torch.cat([quats, residuals.unsqueeze(-1)], dim=-1))
                continue

            V = Vh.transpose(-1, -2)
            det = torch.det(torch.matmul(V, U.transpose(-1, -2)))
            D_diag = torch.ones(B, K, 3, device=device)
            D_diag[..., -1] = det
            D_mat = torch.diag_embed(D_diag)
            R = torch.matmul(V, torch.matmul(D_mat, U.transpose(-1, -2)))

            # Detach R to avoid differentiable-SVD gradient explosion. Since
            # we're using the rotations only as FEATURES (no aux loss), this
            # is fine: the classifier gets a data-dependent rotation readout,
            # and the soft-assignment still gets gradient via:
            #   (i) the entropy collapse penalty (aux_loss)
            #   (ii) the centroid path (w_k governs src_mean/tgt_mean which
            #        flow into residual_per_part)
            R_used = R.detach()

            bad = ~torch.isfinite(R_used).all(dim=-1).all(dim=-1)
            if bad.any():
                R_used = torch.where(bad.unsqueeze(-1).unsqueeze(-1), I3.expand_as(R_used), R_used)

            # Rigidity residual per part (how well the single rotation fits).
            pred = torch.einsum("bkij,bkpj->bkpi", R_used, src_c)
            residual_per_point = ((pred - tgt_c) ** 2).sum(dim=-1)       # (B, K, P)
            residual_per_part = (w_k * residual_per_point).sum(dim=-1) / w_sum.squeeze(-1)  # (B, K)

            quats = self._rot_to_quat(R_used)                            # (B, K, 4)
            feats_list.append(torch.cat([quats, residual_per_part.unsqueeze(-1)], dim=-1))  # (B, K, 5)

        if feats_list:
            features = torch.stack(feats_list, dim=1)                    # (B, F-1, K, 5)
        else:
            features = torch.zeros(B, 0, K, 5, device=device)

        # Entropy collapse penalty (tiny)
        mean_assign = assign.mean(dim=(0, 1, 2))
        entropy = -(mean_assign * mean_assign.clamp(min=1e-8).log()).sum()
        max_entropy = torch.log(torch.tensor(float(K), device=device))
        collapse = (max_entropy - entropy).clamp(min=0.0)
        aux_loss = self.entropy_weight * collapse

        return aux_loss, features

'''

anchor = "class _TemporalPredictionLoss(nn.Module):"
assert anchor in src, "anchor missing"
src = src.replace(anchor, module_code + anchor, 1)

# ----------------------------------------------------- init registration
init_head = "        self.prediction_loss = _TemporalPredictionLoss(feat_dim=hidden2)"
parts_feature_init = (
    init_head
    + "\n        # Parts-feature (v18a, no aux loss path)\n"
    + "        num_parts_f = 6\n"
    + "        self.parts_feature_module = _PartsFeatureProcrustes(feat_dim=hidden2, num_parts=num_parts_f)\n"
    + "        self.parts_feature_proj = nn.Sequential(\n"
    + "            nn.Linear(num_parts_f * 5, hidden2),\n"
    + "            nn.GELU(),\n"
    + "            nn.Linear(hidden2, hidden2),\n"
    + "        )\n"
)
assert init_head in src, "init anchor missing"
src = src.replace(init_head, parts_feature_init, 1)

# ----------------------------------------------------- eager modulation + dispatch
eager_block = '''
        # Parts-feature modulation (always on train + eval). No aux loss beyond
        # a tiny entropy collapse penalty returned by the module.
        self._parts_feature_aux_loss = None
        if 'parts_feature' in self.qcc_variants and corr_matched is not None:
            xyz_flat_pf = sampled[..., :3].reshape(batch_size, num_frames * pts_per_frame, 3)
            pf_loss, pf_features = self.parts_feature_module(
                encoded, xyz_flat_pf, num_frames, pts_per_frame, corr_matched,
            )
            self._parts_feature_aux_loss = pf_loss
            if pf_features.shape[1] > 0:
                pf_pooled = pf_features.mean(dim=1).reshape(batch_size, -1)  # (B, K*5)
                pf_mod = self.parts_feature_proj(pf_pooled)                   # (B, hidden2)
                encoded = encoded * (1.0 + pf_mod.unsqueeze(-1))
'''

rigidity_end_anchor = (
    "            encoded = encoded * (1.0 + modulation)\n\n        # Auxiliary losses"
)
assert rigidity_end_anchor in src, "rigidity anchor missing"
src = src.replace(
    rigidity_end_anchor,
    "            encoded = encoded * (1.0 + modulation)\n" + eager_block + "\n        # Auxiliary losses",
    1,
)

dispatch_tail_anchor = (
    "                    elif variant == 'contrastive':\n                        qcc_loss = self.corr_contrastive("
)
dispatch_branch = (
    "                    elif variant == 'parts_feature':\n"
    "                        qcc_loss = self._parts_feature_aux_loss if self._parts_feature_aux_loss is not None else torch.tensor(0.0, device=encoded.device)\n"
    "                    elif variant == 'contrastive':\n"
    "                        qcc_loss = self.corr_contrastive("
)
assert dispatch_tail_anchor in src, "dispatch anchor missing"
src = src.replace(dispatch_tail_anchor, dispatch_branch, 1)

PATH.write_text(src, encoding="utf-8")
print("patched models/reqnn_motion.py (parts_feature: K=6 rotations + residuals as features, no aux loss)")
