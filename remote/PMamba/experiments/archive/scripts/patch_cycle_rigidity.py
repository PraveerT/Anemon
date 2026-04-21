"""Add cycle-consistency quaternion rigidity to reqnn_motion.py.

Per-point rigidity channel from temporal smoothness of bearing quaternions:
- For each interior frame t: q_prev = rotation(b[t-1, i] -> b[t, i])
                              q_next = rotation(b[t, i] -> b[t+1, i])
- cycle_score = |q_prev . q_next|^2 in [0, 1]
  high = smooth motion (swipe, push), low = oscillating/jittery (shake)

Concatenated with existing single-scale geometric rigidity -> 2-channel input
to rigidity_proj.  Channel 0 (geometric) loads from v8a weights;  channel 1
(cycle) reinits zero and is learned during warm-restart.

qcc_variant token: 'cycle_rigidity'
"""
import re
from pathlib import Path

PATH = Path('models/reqnn_motion.py')
src = PATH.read_text(encoding='utf-8')

CYCLE_FN = r'''

def _compute_cycle_consistency_rigidity(points_4d, num_frames):
    """Cycle-consistency score per point from bearing-quaternion triplets.

    For each interior frame t in [1, F-2], for each point i, compute
    q_prev = rotation(bearing[t-1, i] -> bearing[t, i]) and q_next = rotation
    (bearing[t, i] -> bearing[t+1, i]).  Score = (q_prev . q_next)^2 in [0, 1].
    Score is high when motion is smooth (same rotation continued), low when
    motion oscillates or reverses.  Boundary frames copy nearest interior
    score.

    Args:
        points_4d: (B, F, P, 4)
        num_frames: int
    Returns:
        (B, 1, F*P) in [0, 1]
    """
    B, F, P, _ = points_4d.shape
    device = points_4d.device
    xyz = points_4d[..., :3]

    bbox_min = xyz.min(dim=2).values
    bbox_max = xyz.max(dim=2).values
    centroids = (bbox_min + bbox_max) / 2
    d = xyz - centroids.unsqueeze(2)
    bearings = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # (B, F, P, 3)

    def rotation_quat(a, b):
        # Unit-vector rotation: q = [1 + a.b, a x b], normalized.
        dot = (a * b).sum(dim=-1, keepdim=True)
        cross = torch.cross(a, b, dim=-1)
        q = torch.cat([1.0 + dot, cross], dim=-1)
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    # Need F >= 3 for triplets. If F < 3 fall back to constant 1.
    if F < 3:
        return torch.ones(B, 1, F * P, device=device), 1.0

    scores = torch.zeros(B, F, P, device=device)
    for t in range(1, F - 1):
        a = bearings[:, t - 1]
        m = bearings[:, t]
        b = bearings[:, t + 1]
        q_prev = rotation_quat(a, m)
        q_next = rotation_quat(m, b)
        dot = (q_prev * q_next).sum(dim=-1)  # (B, P)
        scores[:, t] = dot.pow(2)
    # Boundary frames: copy nearest interior
    scores[:, 0] = scores[:, 1]
    scores[:, F - 1] = scores[:, F - 2]

    return scores.reshape(B, 1, -1), 1.0

'''

# Insert before multiscale function
anchor = 'def _compute_bearing_qcc_multiscale'
idx = src.find(anchor)
if idx == -1:
    raise SystemExit('ERR: anchor not found')
src = src[:idx] + CYCLE_FN.lstrip('\n') + '\n' + src[idx:]

# Update rig_channels logic
old_rig = """        is_multiscale = 'multiscale' in self.qcc_variants
        rig_channels = len(rigidity_scales) if is_multiscale else 1"""
new_rig = """        is_multiscale = 'multiscale' in self.qcc_variants
        is_cycle_rig = 'cycle_rigidity' in self.qcc_variants
        if is_multiscale:
            rig_channels = len(rigidity_scales)
        elif is_cycle_rig:
            rig_channels = 2
        else:
            rig_channels = 1"""
if old_rig not in src:
    raise SystemExit('ERR: rig_channels block not found')
src = src.replace(old_rig, new_rig, 1)

# Update dispatch in forward
old_dispatch = """        # Bearing QCC rigidity
        if self.qcc_variant == 'multiscale':
            rigidity, corr_valid_ratio = _compute_bearing_qcc_multiscale(
                sampled, num_frames, scales=self.rigidity_scales,
                corr_matched=corr_matched)
        else:
            rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)"""
new_dispatch = """        # Bearing QCC rigidity
        if self.qcc_variant == 'multiscale':
            rigidity, corr_valid_ratio = _compute_bearing_qcc_multiscale(
                sampled, num_frames, scales=self.rigidity_scales,
                corr_matched=corr_matched)
        elif self.qcc_variant == 'cycle_rigidity':
            geom_rig, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)
            cyc_rig, _ = _compute_cycle_consistency_rigidity(sampled, num_frames)
            rigidity = torch.cat([geom_rig, cyc_rig], dim=1)
        else:
            rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)"""
if old_dispatch not in src:
    raise SystemExit('ERR: dispatch block not found')
src = src.replace(old_dispatch, new_dispatch, 1)

PATH.write_text(src, encoding='utf-8')
print('OK: added _compute_cycle_consistency_rigidity')
print('OK: updated rig_channels for cycle_rigidity variant')
print('OK: wired dispatch for qcc_variant="cycle_rigidity"')
