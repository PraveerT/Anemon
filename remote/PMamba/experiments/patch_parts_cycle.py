"""Option C: K=6 predicted-rotation QCC with real cycle consistency.

Adds qcc_variant='parts_cycle' to BearingQCCFeatureMotion.

Design:
  - Soft-assignment head predicts (B, F, P, K) part memberships.
  - Per frame-pair (t, t+1), per part k, features are pooled by weighted
    average (weight = part-k membership * correspondence mask).
  - rot_head_fwd(pooled_feat_t, pooled_feat_{t+1}) -> 4D quaternion q_fwd_k.
  - rot_head_bwd(pooled_feat_{t+1}, pooled_feat_t) -> 4D quaternion q_bwd_k.
    (Separate heads so forward and backward are free predictions, not inverses
    by construction -- this is what gives cycle consistency a gradient.)
  - Losses:
      * reconstruction: apply quat_to_rot(q_fwd_k) to centered source points;
        minimize weighted residual to centered target points. Keeps rotations
        grounded in real geometry.
      * cycle: q_bwd_k ∘ q_fwd_k should equal identity (1,0,0,0). Measured as
        (1 - |w_component|).
      * entropy: penalize low entropy of averaged assignment (anti-collapse).

Forward-through: the K predicted forward quats are mean-pooled over time and
multiplicatively modulate `encoded` (same feed-forward (B) idea from v16a).

Activate via:
  qcc_variant: parts_cycle
  qcc_weight: 0.1

Paired with Hungarian correspondence (assignment_mode: hungarian).
"""
import re
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

# -------------------------------------------------------------------- module
module_code = '''
def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _quat_to_rot(q):
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ], dim=-1)
    return R.view(*R.shape[:-1], 3, 3)


class _PartsCycleQCC(nn.Module):
    """K-part learned-rotation QCC with reconstruction + cycle consistency."""

    def __init__(self, feat_dim, num_parts=6, entropy_weight=0.01,
                 cycle_weight=1.0, recon_weight=1.0):
        super().__init__()
        self.num_parts = num_parts
        self.entropy_weight = entropy_weight
        self.cycle_weight = cycle_weight
        self.recon_weight = recon_weight

        self.assign_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, num_parts),
        )
        # Rotation heads take concatenated (pooled_t, pooled_{t+1}) features.
        self.rot_head_fwd = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4),
        )
        self.rot_head_bwd = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4),
        )

    def forward(self, encoded, points_xyz_flat, num_frames, pts_per_frame, corr_matched):
        B, D, _ = encoded.shape
        F_, P = num_frames, pts_per_frame
        K = self.num_parts
        device = encoded.device

        feat = encoded.transpose(1, 2).contiguous().view(B, F_, P, D)
        logits = self.assign_head(feat)
        assign = torch.softmax(logits, dim=-1)  # (B, F, P, K)
        xyz = points_xyz_flat.view(B, F_, P, 3)

        loss_total = torch.zeros((), device=device)
        count = 0
        fwd_quats = []

        for t in range(F_ - 1):
            feat_t = feat[:, t]              # (B, P, D)
            feat_t1 = feat[:, t + 1]
            sa = assign[:, t]                # (B, P, K)
            ta = assign[:, t + 1]
            mask = corr_matched[:, t].float() if corr_matched is not None \
                else torch.ones(B, P, device=device)

            w_k = (sa * ta).permute(0, 2, 1) * mask.unsqueeze(1)     # (B, K, P)
            w_sum = w_k.sum(dim=-1, keepdim=True).clamp(min=1e-6)     # (B, K, 1)

            # per-part pooled feature
            feat_t_exp = feat_t.unsqueeze(1).expand(B, K, P, D)
            feat_t1_exp = feat_t1.unsqueeze(1).expand(B, K, P, D)
            pooled_t = (w_k.unsqueeze(-1) * feat_t_exp).sum(dim=-2) / w_sum
            pooled_t1 = (w_k.unsqueeze(-1) * feat_t1_exp).sum(dim=-2) / w_sum

            joint_fwd = torch.cat([pooled_t, pooled_t1], dim=-1)      # (B, K, 2D)
            joint_bwd = torch.cat([pooled_t1, pooled_t], dim=-1)
            q_fwd = self.rot_head_fwd(joint_fwd)
            q_bwd = self.rot_head_bwd(joint_bwd)
            q_fwd = q_fwd / q_fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            q_bwd = q_bwd / q_bwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            # cycle drift: (q_bwd ∘ q_fwd) should equal identity (1,0,0,0)
            q_cycle = _quat_mul(q_bwd, q_fwd)
            cycle_drift = (1.0 - q_cycle[..., 0].abs()).mean()

            # reconstruction: rotate centered source by quat_to_rot(q_fwd), compare to centered target
            R_fwd = _quat_to_rot(q_fwd)                 # (B, K, 3, 3)
            src_b = xyz[:, t].unsqueeze(1).expand(B, K, P, 3)
            tgt_b = xyz[:, t + 1].unsqueeze(1).expand(B, K, P, 3)
            src_mean = (w_k.unsqueeze(-1) * src_b).sum(dim=-2) / w_sum
            tgt_mean = (w_k.unsqueeze(-1) * tgt_b).sum(dim=-2) / w_sum
            src_c = src_b - src_mean.unsqueeze(-2)
            tgt_c = tgt_b - tgt_mean.unsqueeze(-2)
            pred = torch.einsum("bkij,bkpj->bkpi", R_fwd, src_c)
            residual = ((pred - tgt_c) ** 2).sum(dim=-1)
            active = (w_sum.squeeze(-1) > 1e-4).float()
            recon_pair = (w_k * residual).sum(dim=-1) / w_sum.squeeze(-1)
            recon_drift = ((recon_pair * active).sum(dim=-1) / active.sum(dim=-1).clamp(min=1.0)).mean()

            loss_pair = self.recon_weight * recon_drift + self.cycle_weight * cycle_drift
            loss_total = loss_total + loss_pair
            count += 1
            fwd_quats.append(q_fwd)

        if count > 0:
            loss_total = loss_total / count

        mean_assign = assign.mean(dim=(0, 1, 2))
        entropy = -(mean_assign * mean_assign.clamp(min=1e-8).log()).sum()
        max_entropy = torch.log(torch.tensor(float(K), device=device))
        collapse_penalty = (max_entropy - entropy).clamp(min=0.0)
        loss_total = loss_total + self.entropy_weight * collapse_penalty

        if fwd_quats:
            quats = torch.stack(fwd_quats, dim=1)
        else:
            quats = torch.zeros(B, 0, K, 4, device=device)
        return loss_total, quats

'''

anchor = "class _TemporalPredictionLoss(nn.Module):"
assert anchor in src, "anchor not found"
src = src.replace(anchor, module_code + anchor, 1)

# --------------------------------------- init registration (after prediction_loss)
init_head = "        self.prediction_loss = _TemporalPredictionLoss(feat_dim=hidden2)"
parts_cycle_init = (
    init_head
    + "\n        # Parts-cycle QCC (Option C)\n"
    + "        num_parts_c = 6\n"
    + "        self.parts_cycle_loss = _PartsCycleQCC(feat_dim=hidden2, num_parts=num_parts_c)\n"
    + "        self.parts_cycle_rot_proj = nn.Sequential(\n"
    + "            nn.Linear(num_parts_c * 4, hidden2),\n"
    + "            nn.GELU(),\n"
    + "            nn.Linear(hidden2, hidden2),\n"
    + "        )\n"
)
assert init_head in src, "init anchor missing"
src = src.replace(init_head, parts_cycle_init, 1)

# --------------------------------------- eager modulation + dispatch
eager_block = '''
        # Parts-cycle modulation (train + eval for consistent features).
        self._parts_cycle_aux_loss = None
        if 'parts_cycle' in self.qcc_variants and corr_matched is not None:
            xyz_flat_pc = sampled[..., :3].reshape(batch_size, num_frames * pts_per_frame, 3)
            pc_loss, pc_quats = self.parts_cycle_loss(
                encoded, xyz_flat_pc, num_frames, pts_per_frame, corr_matched,
            )
            self._parts_cycle_aux_loss = pc_loss
            if pc_quats.shape[1] > 0:
                pc_feat = pc_quats.reshape(batch_size, pc_quats.shape[1], -1).mean(dim=1)
                pc_mod = self.parts_cycle_rot_proj(pc_feat)
                encoded = encoded * (1.0 + pc_mod.unsqueeze(-1))
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
    "                    elif variant == 'parts_cycle':\n"
    "                        qcc_loss = self._parts_cycle_aux_loss if self._parts_cycle_aux_loss is not None else torch.tensor(0.0, device=encoded.device)\n"
    "                    elif variant == 'contrastive':\n"
    "                        qcc_loss = self.corr_contrastive("
)
assert dispatch_tail_anchor in src, "dispatch anchor missing"
src = src.replace(dispatch_tail_anchor, dispatch_branch, 1)

PATH.write_text(src, encoding="utf-8")
print("patched models/reqnn_motion.py (parts_cycle QCC, option C)")
