"""Add K-part rigidity QCC: weighted Procrustes per part + rotation modulation.

Implements Options A+B from the K=6 parts discussion (2026-04-19/20):
  A) Rigidity-residual loss: network predicts a soft assignment of each point
     to K parts; for each frame-pair and each part, a weighted Procrustes
     solves the best rigid rotation; residual of that fit is the loss. Drives
     the soft assignment to pick locally rigid groupings (palm, 5 fingers).
  B) Feed-forward modulation: the K derived quaternions are pooled over time,
     projected, and multiplicatively modulate `encoded` before the classifier.

Adds to nvidia_dataloader's sister file `models/reqnn_motion.py`:
  - class _PartsRigidityProcrustes (soft-assign head + diff SVD)
  - BearingQCCFeatureMotion.__init__ registers parts_rigidity_loss + parts_rot_proj
  - Forward dispatch: modulates encoded by parts, adds loss in training

Activate via:
  qcc_variant: parts_rigidity
  qcc_weight: 0.1
  num_parts: 6

Cache prereq: Hungarian correspondence (assignment_mode: hungarian on loader).
"""
import re
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

# -------------------------------------------------------------------- module
module_code = '''
class _PartsRigidityProcrustes(nn.Module):
    """K-part rigidity decomposition via weighted Procrustes.

    Predicts a soft assignment of every point to K parts. For each consecutive
    frame pair and each part, solves weighted Procrustes (differentiable SVD)
    to find the best rigid rotation that maps the part's source points to the
    target points. Returns:
      - residual loss (weighted squared reconstruction error across parts)
      - K quaternions per frame-pair, used by the model for feature modulation
    Entropy regularization keeps all K parts active.
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
        # R: (..., 3, 3) -> quat (..., 4) as (w, x, y, z); Shepperd's method.
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
        # encoded: (B, D, F*P); points_xyz_flat: (B, F*P, 3); corr_matched: (B, F-1, P) bool
        B, D, _ = encoded.shape
        F_, P = num_frames, pts_per_frame
        K = self.num_parts
        device = encoded.device

        feat = encoded.transpose(1, 2).contiguous()          # (B, F*P, D)
        logits = self.assign_head(feat)                       # (B, F*P, K)
        assign = torch.softmax(logits, dim=-1).view(B, F_, P, K)

        xyz = points_xyz_flat.view(B, F_, P, 3)

        loss_total = torch.zeros((), device=device)
        count = 0
        quats_list = []
        I3 = torch.eye(3, device=device).view(1, 1, 3, 3)

        for t in range(F_ - 1):
            src = xyz[:, t]                 # (B, P, 3)
            tgt = xyz[:, t + 1]
            if corr_matched is not None:
                mask = corr_matched[:, t].float()  # (B, P)
            else:
                mask = torch.ones(B, P, device=device)
            sa = assign[:, t]               # (B, P, K)
            ta = assign[:, t + 1]

            w_k = (sa * ta).permute(0, 2, 1) * mask.unsqueeze(1)   # (B, K, P)
            w_sum = w_k.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, K, 1)

            src_b = src.unsqueeze(1).expand(B, K, P, 3)
            tgt_b = tgt.unsqueeze(1).expand(B, K, P, 3)
            src_mean = (w_k.unsqueeze(-1) * src_b).sum(dim=-2) / w_sum   # (B, K, 3)
            tgt_mean = (w_k.unsqueeze(-1) * tgt_b).sum(dim=-2) / w_sum
            src_c = src_b - src_mean.unsqueeze(-2)
            tgt_c = tgt_b - tgt_mean.unsqueeze(-2)

            H = torch.einsum("bkp,bkpi,bkpj->bkij", w_k, src_c, tgt_c)
            H = H + 1e-6 * I3  # numerical stability

            try:
                U, S, Vh = torch.linalg.svd(H)
            except Exception:
                # catastrophic fallback: identity rotations, zero loss contribution
                R = I3.expand(B, K, 3, 3).contiguous()
                quats_list.append(self._rot_to_quat(R))
                continue

            V = Vh.transpose(-1, -2)
            det = torch.det(torch.matmul(V, U.transpose(-1, -2)))    # (B, K)
            D_diag = torch.ones(B, K, 3, device=device)
            D_diag[..., -1] = det
            D_mat = torch.diag_embed(D_diag)
            R = torch.matmul(V, torch.matmul(D_mat, U.transpose(-1, -2)))  # (B, K, 3, 3)

            # Detach R to avoid differentiable-SVD gradient explosion. The
            # gradient still reaches the soft-assignment through the w_k
            # prefactor in the residual, which is the intended signal
            # ("assign points so that fitted rotations give low residual").
            R_used = R.detach()

            # Replace any NaN/Inf in R with identity (safety net for rare
            # degenerate SVD).
            bad = ~torch.isfinite(R_used).all(dim=-1).all(dim=-1)
            if bad.any():
                R_used = torch.where(
                    bad.unsqueeze(-1).unsqueeze(-1),
                    I3.expand_as(R_used),
                    R_used,
                )

            pred = torch.einsum("bkij,bkpj->bkpi", R_used, src_c)
            residual = ((pred - tgt_c) ** 2).sum(dim=-1)             # (B, K, P)
            loss_pair = (w_k * residual).sum(dim=-1) / w_sum.squeeze(-1)  # (B, K)
            # Only include parts with meaningful weight in the mean.
            active = (w_sum.squeeze(-1) > 1e-4).float()
            active_count = active.sum(dim=-1).clamp(min=1.0)
            loss_total = loss_total + ((loss_pair * active).sum(dim=-1) / active_count).mean()
            count += 1

            quats_list.append(self._rot_to_quat(R_used))              # (B, K, 4)

        if count > 0:
            loss_total = loss_total / count

        # Entropy regularization: penalize low entropy (collapse) instead of
        # adding negative entropy directly, so total loss stays non-negative.
        mean_assign = assign.mean(dim=(0, 1, 2))                      # (K,)
        entropy = -(mean_assign * mean_assign.clamp(min=1e-8).log()).sum()
        max_entropy = torch.log(torch.tensor(float(K), device=device))
        collapse_penalty = (max_entropy - entropy).clamp(min=0.0)
        loss_total = loss_total + self.entropy_weight * collapse_penalty

        if quats_list:
            quats = torch.stack(quats_list, dim=1)                    # (B, F-1, K, 4)
        else:
            quats = torch.zeros(B, 0, K, 4, device=device)
        return loss_total, quats

'''

# Insert module before the existing _TemporalPredictionLoss definition.
anchor = "class _TemporalPredictionLoss(nn.Module):"
assert anchor in src, "anchor not found: _TemporalPredictionLoss"
src = src.replace(anchor, module_code + anchor, 1)

# ------------------------------------------------------- init registration
init_head = "        self.prediction_loss = _TemporalPredictionLoss(feat_dim=hidden2)"
parts_init_block = (
    init_head
    + "\n        # K-part Procrustes (parts_rigidity variant)\n"
    + "        num_parts = 6\n"
    + "        self.parts_rigidity_loss = _PartsRigidityProcrustes(feat_dim=hidden2, num_parts=num_parts)\n"
    + "        self.parts_rot_proj = nn.Sequential(\n"
    + "            nn.Linear(num_parts * 4, hidden2),\n"
    + "            nn.GELU(),\n"
    + "            nn.Linear(hidden2, hidden2),\n"
    + "        )\n"
)
assert init_head in src, "init anchor not found"
src = src.replace(init_head, parts_init_block, 1)

# --------------------------- eager modulation (works at both train and eval)
eager_block = '''
        # Parts-rigidity modulation (runs at train and eval so features are consistent).
        # Also stashes (loss, quats) for the aux dispatch below.
        self._parts_aux_loss = None
        if 'parts_rigidity' in self.qcc_variants and corr_matched is not None:
            xyz_flat_for_parts = sampled[..., :3].reshape(batch_size, num_frames * pts_per_frame, 3)
            parts_loss, parts_quats = self.parts_rigidity_loss(
                encoded, xyz_flat_for_parts, num_frames, pts_per_frame, corr_matched,
            )
            self._parts_aux_loss = parts_loss
            if parts_quats.shape[1] > 0:
                parts_feat = parts_quats.reshape(batch_size, parts_quats.shape[1], -1).mean(dim=1)  # (B, K*4)
                parts_mod = self.parts_rot_proj(parts_feat)  # (B, hidden2)
                encoded = encoded * (1.0 + parts_mod.unsqueeze(-1))
'''

# Find the end of the rigidity modulation block to inject after it.
rigidity_end_anchor = (
    "            encoded = encoded * (1.0 + modulation)\n\n        # Auxiliary losses"
)
assert rigidity_end_anchor in src, "rigidity end anchor not found"
src = src.replace(
    rigidity_end_anchor,
    "            encoded = encoded * (1.0 + modulation)\n" + eager_block + "\n        # Auxiliary losses",
    1,
)

# ---------------------------- aux dispatch branch: consume stashed parts_loss
dispatch_tail_anchor = "                    elif variant == 'contrastive':\n                        qcc_loss = self.corr_contrastive("
# Inject a parts_rigidity branch just before 'contrastive' inside the `elif corr_matched is not None:` block.
dispatch_branch = (
    "                    elif variant == 'parts_rigidity':\n"
    "                        qcc_loss = self._parts_aux_loss if self._parts_aux_loss is not None else torch.tensor(0.0, device=encoded.device)\n"
    "                    elif variant == 'contrastive':\n"
    "                        qcc_loss = self.corr_contrastive("
)
assert dispatch_tail_anchor in src, "dispatch anchor not found"
src = src.replace(dispatch_tail_anchor, dispatch_branch, 1)

PATH.write_text(src, encoding="utf-8")
print("patched models/reqnn_motion.py (K=6 parts_rigidity variant + feed-forward modulation)")
