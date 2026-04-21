"""Patch reqnn_motion.py: add _BearingRotationQCCLoss (literal per-point QCC).

This is the true quaternion cross-correlation we never tried. Predicts per-point
bearing rotation quaternion (rotation taking bearing_t to bearing_t+1) from
encoded features. Loss = geodesic (1 - |q_pred . q_gt|^2).

Discriminative signal: directional motion and finger orientations — targets the
worst v8a classes (24 OK sign, 13 two fingers, 7 fingers-down, 3 hand-down,
6 fingers-up, 1 hand-right, 5 two-fingers-right) which confuse with each other
on direction/finger-count.

Token: qr  (qcc_variant='bearing_rot')
"""
import re
import sys
from pathlib import Path

PATH = Path('models/reqnn_motion.py')
src = PATH.read_text(encoding='utf-8')

NEW_CLASS = r'''

class _BearingRotationQCCLoss(nn.Module):
    """True per-point bearing-rotation QCC.

    Predicts per-point quaternion that rotates bearing(t) to bearing(t+1) from
    encoded features. This is the literal quaternion cross-correlation: a
    quaternion-valued prediction supervised against the geometric quaternion
    rotating the point's bearing vector between frames.

    Token: ``qr``.
    """

    def __init__(self, feat_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 4),
        )

    @staticmethod
    def _bearing(xyz):
        # xyz: (B, F, P, 3). Center per-frame then normalize to unit vectors.
        bbox_min = xyz.min(dim=2).values
        bbox_max = xyz.max(dim=2).values
        centroids = (bbox_min + bbox_max) / 2
        d = xyz - centroids.unsqueeze(2)
        return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    @staticmethod
    def _rotation_quat(a, b):
        # Quaternion rotating unit vector a -> unit vector b.
        # q = [1 + a.b, a x b], normalized. Falls back to 180-deg handling.
        dot = (a * b).sum(dim=-1, keepdim=True)
        cross = torch.cross(a, b, dim=-1)
        w = 1.0 + dot
        q = torch.cat([w, cross], dim=-1)
        # 180-deg case: a ~= -b, use any perpendicular axis.
        near_opposite = (w.squeeze(-1) < 1e-6)
        if near_opposite.any():
            # Pick axis perpendicular to a.
            ex = torch.zeros_like(a); ex[..., 0] = 1.0
            ey = torch.zeros_like(a); ey[..., 1] = 1.0
            use_ey = (a[..., 0].abs() > 0.9).unsqueeze(-1)
            axis = torch.where(use_ey, ey, ex)
            perp = axis - (axis * a).sum(dim=-1, keepdim=True) * a
            perp = perp / perp.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            q_flip = torch.cat([torch.zeros_like(dot), perp], dim=-1)
            q = torch.where(near_opposite.unsqueeze(-1), q_flip, q)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def forward(self, encoded, points_xyz, num_frames, pts_per_frame):
        # encoded: (B, D, F*P); points_xyz: (B, F, P, 3)
        B, D, _ = encoded.shape
        feat = encoded.view(B, D, num_frames, pts_per_frame)
        bearings = self._bearing(points_xyz)  # (B, F, P, 3)

        loss = torch.tensor(0.0, device=encoded.device)
        count = 0
        for t in range(num_frames - 1):
            feat_t = feat[:, :, t].permute(0, 2, 1)  # (B, P, D)
            a = bearings[:, t]        # (B, P, 3)
            b = bearings[:, t + 1]    # (B, P, 3)
            q_gt = self._rotation_quat(a, b).detach()  # (B, P, 4)

            q_pred = self.predictor(feat_t)  # (B, P, 4)
            q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            # Geodesic loss: 1 - (q_pred . q_gt)^2 handles q ~ -q equivalence.
            dot = (q_pred * q_gt).sum(dim=-1)
            loss = loss + (1.0 - dot.pow(2)).mean()
            count += 1
        return loss / max(count, 1)

'''

# Insert new class right after _GroundedDisplacementBidirLoss
anchor_line = 'class _TemporalPredictionLoss'
idx = src.find(anchor_line)
if idx == -1:
    print('ERR: could not find anchor class _TemporalPredictionLoss')
    sys.exit(1)

new_src = src[:idx] + NEW_CLASS.lstrip('\n') + '\n' + src[idx:]

# Instantiate in BearingQCCFeatureMotion.__init__
# Anchor: after self.grounded_disp_bidir_loss line
anchor = 'self.grounded_disp_bidir_loss = _GroundedDisplacementBidirLoss(feat_dim=hidden2)'
inst_line = '        self.bearing_rot_qcc_loss = _BearingRotationQCCLoss(feat_dim=hidden2)'
if anchor not in new_src:
    print('ERR: could not find grounded_disp_bidir_loss instantiation')
    sys.exit(1)
new_src = new_src.replace(anchor, anchor + '\n' + inst_line, 1)

# Dispatch: extend grounded_disp* branch to include 'bearing_rot'
old_branch = "elif variant in ('grounded_disp', 'grounded_disp_dir', 'grounded_disp_bidir'):"
new_branch = "elif variant in ('grounded_disp', 'grounded_disp_dir', 'grounded_disp_bidir', 'bearing_rot'):"
if old_branch not in new_src:
    print('ERR: could not find dispatch branch')
    sys.exit(1)
new_src = new_src.replace(old_branch, new_branch, 1)

# Add case inside the branch
disp_case = "elif variant == 'grounded_disp_bidir':\n                        qcc_loss = self.grounded_disp_bidir_loss(\n                            encoded, sampled[..., :3], num_frames, pts_per_frame)"
replacement = disp_case + """
                    elif variant == 'bearing_rot':
                        # Reshape sampled xyz for _BearingRotationQCCLoss: need (B, F, P, 3)
                        xyz_raw = sampled[..., :3].view(sampled.shape[0], num_frames, pts_per_frame, 3)
                        qcc_loss = self.bearing_rot_qcc_loss(
                            encoded, xyz_raw, num_frames, pts_per_frame)"""
if disp_case not in new_src:
    print('ERR: could not find disp_bidir case')
    sys.exit(1)
new_src = new_src.replace(disp_case, replacement, 1)

PATH.write_text(new_src, encoding='utf-8')
print('OK: patched reqnn_motion.py')
print('  Added class _BearingRotationQCCLoss')
print('  Added self.bearing_rot_qcc_loss instantiation')
print('  Extended dispatch to qcc_variant="bearing_rot"')
