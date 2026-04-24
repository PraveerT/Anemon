"""Verify R-based residual == quaternion-based residual (numerically).

For the paper, we want quaternion math literal. But we need to confirm the
numerics are IDENTICAL so all prior results carry over unchanged.
"""
import torch
import torch.nn.functional as F


def rot_to_quat(R):
    m00,m01,m02 = R[0,0], R[0,1], R[0,2]
    m10,m11,m12 = R[1,0], R[1,1], R[1,2]
    m20,m21,m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = torch.sqrt(tr.clamp(min=-0.999) + 1) * 2
        q = torch.stack([0.25*s, (m21-m12)/s, (m02-m20)/s, (m10-m01)/s])
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1 + m00 - m11 - m22).clamp(min=1e-8) * 2
        q = torch.stack([(m21-m12)/s, 0.25*s, (m01+m10)/s, (m02+m20)/s])
    elif m11 > m22:
        s = torch.sqrt(1 + m11 - m00 - m22).clamp(min=1e-8) * 2
        q = torch.stack([(m02-m20)/s, (m01+m10)/s, 0.25*s, (m12+m21)/s])
    else:
        s = torch.sqrt(1 + m22 - m00 - m11).clamp(min=1e-8) * 2
        q = torch.stack([(m10-m01)/s, (m02+m20)/s, (m12+m21)/s, 0.25*s])
    q = F.normalize(q, dim=-1)
    if q[0] < 0: q = -q
    return q


def hamilton(a, b):
    aw,ax,ay,az = a[...,0], a[...,1], a[...,2], a[...,3]
    bw,bx,by,bz = b[...,0], b[...,1], b[...,2], b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def quat_rotate(q, points):
    """Rotate points (N,3) by unit quaternion q (4,). Returns (N,3)."""
    # Broadcast q to all N points
    N = points.shape[0]
    q_b = q.unsqueeze(0).expand(N, -1)
    # Treat points as pure quaternions: (0, x, y, z)
    pq = torch.cat([torch.zeros(N, 1, device=points.device, dtype=points.dtype), points], dim=-1)
    # Conjugate: q* = (w, -x, -y, -z)
    q_conj = torch.cat([q_b[:, 0:1], -q_b[:, 1:]], dim=-1)
    # Rotate: q * pq * q*
    temp = hamilton(q_b, pq)
    rotated = hamilton(temp, q_conj)
    return rotated[:, 1:]  # drop scalar part


# Test: compare R·p vs q·p·q* numerically
print("Testing R-based vs quaternion-based rotation on random rotations + points...")
torch.manual_seed(42)
N_trials = 1000
N_points = 128
max_diff = 0.0
max_rel = 0.0

for trial in range(N_trials):
    # Random rotation matrix via Kabsch on random point sets
    p1 = torch.randn(N_points, 3)
    # Random rotation
    axis = torch.randn(3); axis = axis / axis.norm()
    angle = torch.rand(1) * 6.28
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    t_vec = torch.randn(3) * 0.1
    p2 = p1 @ R.T + t_vec + torch.randn(N_points, 3) * 0.01  # noisy

    # Kabsch fit (normally done on real data)
    p1_c = p1 - p1.mean(0, keepdim=True)
    p2_c = p2 - p2.mean(0, keepdim=True)
    H = p1_c.T @ p2_c
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.T
    det = torch.det(V @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, det.item()]))
    R_fit = V @ D @ U.T
    t_fit = p2.mean(0) - R_fit @ p1.mean(0)

    # R-based rigid prediction
    rigid_R = p1 @ R_fit.T + t_fit

    # Quaternion-based rigid prediction
    q_fit = rot_to_quat(R_fit)
    rigid_q = quat_rotate(q_fit, p1 - p1.mean(0, keepdim=True)) + p2.mean(0)

    diff = (rigid_R - rigid_q).abs().max().item()
    rel = diff / (rigid_R.abs().max().item() + 1e-8)
    if diff > max_diff:
        max_diff = diff
    if rel > max_rel:
        max_rel = rel

print(f"Max absolute difference across {N_trials} trials: {max_diff:.2e}")
print(f"Max relative difference: {max_rel:.2e}")
print(f"Numerically identical (diff < 1e-5)? {max_diff < 1e-5}")
