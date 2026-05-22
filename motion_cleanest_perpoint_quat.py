"""Per-point quaternion temporal mechanism on the v2 AE canonical.

Inputs are the baked canonical point clouds (K=1024) where index k tracks
the same anatomical region across frames thanks to correspondence. For
each k we have a trajectory (T, 3). We compute the "turning quaternion"
between consecutive velocity directions, Hamilton-product chain across
t, and read out via a small MLP.

This is the first real-mechanism quaternion aux on NVGesture: the math
is only meaningful because correspondence holds, distinguishing it from
the decorative inertia quat-head (which is correspondence-free and gradient-zeroes
itself at inference).

Architecture:
  raw canonical input (B, T, K, 3) [from CanonicalNvidiaLoader]
    -> velocities v_k(t) = canonical[t+1, k] - canonical[t, k]
    -> unit directions d_k(t)
    -> turning quaternions q_k(t) from d_k(t-1) to d_k(t)
    -> Hamilton chain across t -> Q_k (B, K, 4)
    -> per-K projection MLP (4 -> 32)
    -> mean + max pool over K -> 64-d feature
    -> Linear(64, num_classes) -> aux logits
    -> ensembled with main: out = main + perpoint_scale * aux
"""
import torch
import torch.nn as nn

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead


def hamilton_product(a, b):
    """a, b: (..., 4) -> (..., 4). Both quaternion components in (w, x, y, z)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def per_point_quaternion_chain(canonical, eps=1e-6):
    """Compute the Hamilton-product chain of turning quaternions per index.

    canonical: (B, T, K, 3)
    Returns: (B, K, 4) cumulative rotation quaternion per index.
    """
    v = canonical[:, 1:] - canonical[:, :-1]                            # (B, T-1, K, 3)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    d = v / v_norm                                                       # unit directions

    a = d[:, :-1]                                                        # (B, T-2, K, 3)
    b = d[:, 1:]
    dot = (a * b).sum(-1).clamp(min=-1 + eps, max=1 - eps)               # (B, T-2, K)
    cross = torch.cross(a, b, dim=-1)                                    # (B, T-2, K, 3)
    # Halfway quaternion form: q = (1 + dot, cross) then normalize.
    q_w = (1.0 + dot).unsqueeze(-1)
    q = torch.cat([q_w, cross], dim=-1)                                  # (B, T-2, K, 4)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)

    # Hamilton chain across t. Renormalise each step so numerical error
    # doesn't compound the magnitude (the product of unit quaternions
    # should be unit, but float32 + 30 multiplications drifts toward 0).
    Q = q[:, 0]                                                          # (B, K, 4)
    for t in range(1, q.shape[1]):
        Q = hamilton_product(Q, q[:, t])
        Q = Q / Q.norm(dim=-1, keepdim=True).clamp(min=eps)
    return Q


class MotionCleanestLinXLQuatHeadPerPoint(MotionCleanestLinXLQuatHead):
    """Main path = MotionCleanestLinXLQuatHead on the canonical input
    (same as the canonical-replace baseline). Adds a per-point quaternion
    chain aux that uses ALL K=1024 canonical indices (not just the 172
    the spatial pipeline samples). The aux is ensembled with main via a
    learnable scalar like the existing decorative quat-head."""

    def __init__(self, *args,
                 perpoint_proj_dim=32,
                 perpoint_quat_scale=0.3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.perpoint_proj = nn.Sequential(
            nn.Linear(4, perpoint_proj_dim),
            nn.GELU(),
            nn.Linear(perpoint_proj_dim, perpoint_proj_dim),
        )
        # mean + max pool concatenated -> 2 * proj_dim
        self.perpoint_head = nn.Sequential(
            nn.LayerNorm(2 * perpoint_proj_dim),
            nn.Linear(2 * perpoint_proj_dim, self.num_classes),
        )
        self.perpoint_quat_scale = nn.Parameter(torch.tensor(float(perpoint_quat_scale)))

    def forward(self, inputs):
        # Main path (returns logits + decorative quat-head ensemble already).
        main_out = super().forward(inputs)

        if isinstance(inputs, dict):
            raw = inputs['points']
        else:
            raw = inputs
        # Input is canonical (CanonicalNvidiaLoader) with 4 channels (xyz + time).
        canonical_xyz = raw[..., :3]                                     # (B, T, K, 3)
        Q = per_point_quaternion_chain(canonical_xyz)                     # (B, K, 4)
        proj = self.perpoint_proj(Q)                                      # (B, K, P)
        mean_pool = proj.mean(dim=1)                                      # (B, P)
        max_pool = proj.max(dim=1)[0]                                     # (B, P)
        feat = torch.cat([mean_pool, max_pool], dim=-1)                   # (B, 2P)
        perpoint_logits = self.perpoint_head(feat)

        return main_out + self.perpoint_quat_scale * perpoint_logits
