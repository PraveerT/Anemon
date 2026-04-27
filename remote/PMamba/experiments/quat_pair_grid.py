"""Test variant W (rotation-plane bivector, N3) on multiple PMamba confusion pairs.

For each (class_a, class_b) pair, build binary subset, train tiny PointNet from
scratch with variant A (xyzt baseline) and W (xyzt + bivector). 5 seeds each,
report mean ± std deltas. Tests whether W generalizes beyond 3↔16.
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/notebooks/PMamba/experiments")
import nvidia_dataloader as nd

DEV = torch.device("cuda")

PAIRS = [
    (3, 16),   # Move down ↔ Push down
    (18, 9),   # Pull in / Call
    (5, 4),    # Two fingers R/L
    (1, 0),    # Hand right/left
    (14, 13),  # Three fingers / Two fingers
    (8, 12),   # Click index / Show index
]

# ============= quaternion / Kabsch utils =============
def hamilton(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def rot_to_quat(R):
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    s = torch.sqrt((tr + 1).clamp(min=1e-6)) * 2
    qw = 0.25 * s
    qx = (m21 - m12) / s
    qy = (m02 - m20) / s
    qz = (m10 - m01) / s
    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return F.normalize(q, dim=-1)


def kabsch_qf(xyz):
    """xyz: (B, T, N, 3) -> qf (B, T, 4)"""
    B, T, N, _ = xyz.shape
    src = xyz[:, :-1]; tgt = xyz[:, 1:]
    src_c = src - src.mean(dim=2, keepdim=True)
    tgt_c = tgt - tgt.mean(dim=2, keepdim=True)
    H = torch.einsum("btni,btnj->btij", src_c, tgt_c)
    H = H + 1e-6 * torch.eye(3, device=xyz.device).view(1, 1, 3, 3)
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R_fwd = V @ D @ U.transpose(-1, -2)
    qf = rot_to_quat(R_fwd)
    qf = torch.cat([qf, qf[:, -1:].clone()], dim=1)
    return qf


def kabsch_qf_t(xyz):
    """Like kabsch_qf but also returns translation t (B, T, 3)."""
    B, T, N, _ = xyz.shape
    src = xyz[:, :-1]; tgt = xyz[:, 1:]
    src_mu = src.mean(dim=2, keepdim=True); tgt_mu = tgt.mean(dim=2, keepdim=True)
    src_c = src - src_mu; tgt_c = tgt - tgt_mu
    H = torch.einsum("btni,btnj->btij", src_c, tgt_c)
    H = H + 1e-6 * torch.eye(3, device=xyz.device).view(1, 1, 3, 3)
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    qf = rot_to_quat(R)
    # translation
    t_fwd = tgt_mu.squeeze(2) - torch.einsum("btij,btj->bti", R, src_mu.squeeze(2))
    qf = torch.cat([qf, qf[:, -1:].clone()], dim=1)
    t_fwd = torch.cat([t_fwd, t_fwd[:, -1:].clone()], dim=1)
    return qf, t_fwd


def build_features(xyzt, variant):
    xyz = xyzt[..., :3]
    if variant == "A":
        return xyzt
    qf = kabsch_qf(xyz)
    if variant == "W":
        v = qf[..., 1:]
        nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        nh = v / nrm
        nx, ny, nz = nh[..., 0:1], nh[..., 1:2], nh[..., 2:3]
        b11 = nx * nx - 1.0/3; b22 = ny * ny - 1.0/3
        b12 = nx * ny; b13 = nx * nz; b23 = ny * nz
        beta = torch.cat([b11, b22, b12, b13, b23], dim=-1)
        beta_p = beta.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, beta_p], dim=-1)

    # AA: dual quaternion (rotation + translation in 8-d)
    if variant == "AA":
        qf2, t_fwd = kabsch_qf_t(xyz)
        # dual part q' = 0.5 * [0, t] * q  (quat product of pure-translation quat with q)
        qt = torch.cat([torch.zeros_like(t_fwd[..., 0:1]), t_fwd], dim=-1)  # (B, T, 4)
        q_dual = 0.5 * hamilton(qt, qf2)
        dq = torch.cat([qf2, q_dual], dim=-1)                                 # (B, T, 8)
        dq_p = dq.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, dq_p], dim=-1)                                 # 4+8=12

    # AB: Hopf swing-twist decomposition (twist around z=(0,0,1))
    if variant == "AB":
        # twist around z: q_twist = (cos(phi/2), 0, 0, sin(phi/2))
        # extract: phi = 2 * atan2(q_z, q_w)
        qw = qf[..., 0:1]; qz = qf[..., 3:4]
        phi = 2 * torch.atan2(qz, qw + 1e-9)
        cos_p = torch.cos(phi/2); sin_p = torch.sin(phi/2)
        # swing = q * conj(q_twist) ; q_twist_conj = (cos(phi/2), 0, 0, -sin(phi/2))
        q_twist_conj = torch.cat([cos_p, torch.zeros_like(cos_p), torch.zeros_like(cos_p), -sin_p], dim=-1)
        q_swing = hamilton(qf, q_twist_conj)                                   # 4-d
        # output: (swing_xyz=3, twist_cos=1, twist_sin=1) = 5-d
        feat = torch.cat([q_swing[..., 1:], cos_p, sin_p], dim=-1)             # (B, T, 5)
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                               # 4+5=9

    # AC: SLERP smoothness residual
    if variant == "AC":
        # q_slerp_mid = SLERP(q_{t-1}, q_{t+1}, 0.5)
        # for unit quats: SLERP(p, q, 0.5) = (p+q) / ||p+q||  if dot>0 else (p-q)/||p-q||
        B, T, _ = qf.shape
        q_prev = torch.cat([qf[:, :1].clone(), qf[:, :-1]], dim=1)
        q_next = torch.cat([qf[:, 1:], qf[:, -1:].clone()], dim=1)
        dot = (q_prev * q_next).sum(-1, keepdim=True)
        sign = torch.sign(dot + 1e-9)
        q_mid = q_prev + sign * q_next
        q_mid = F.normalize(q_mid, dim=-1)
        residual = qf - q_mid                                                  # (B, T, 4)
        residual_p = residual.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, residual_p], dim=-1)                           # 4+4=8

    # AD: Cayley parameters c = q_xyz / (1 + q_w)
    if variant == "AD":
        qw = qf[..., 0:1]
        c = qf[..., 1:] / (1 + qw.abs() + 1e-9)
        c_p = c.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, c_p], dim=-1)                                  # 4+3=7

    # AE: anchor-quaternion inner products (8 reference quats)
    if variant == "AE":
        # deterministic anchors via Halton sampling on S^3 (use seed 0)
        torch.manual_seed(7)
        anchors = torch.randn(8, 4, device=xyz.device)
        anchors = F.normalize(anchors, dim=-1)                                 # (8, 4)
        # |<qf, q_k>| for each k
        prod = (qf.unsqueeze(2) * anchors.view(1, 1, 8, 4)).sum(-1).abs()      # (B, T, 8)
        prod_p = prod.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, prod_p], dim=-1)                               # 4+8=12

    # BA: pure log map to so(3) — 3-d Lie algebra coord
    if variant == "BA":
        w = qf[..., 0:1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        v = qf[..., 1:]
        nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        theta = 2 * torch.acos(w.abs())
        xi = theta * v / nrm                                                   # (B, T, 3)
        xi_p = xi.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, xi_p], dim=-1)                                 # 4+3=7

    # BC: cross-covariance singular values (Fisher-info-like)
    if variant == "BC":
        # recompute H per frame and get its singular values (3 per frame)
        B, T, N, _ = xyz.shape
        src = xyz[:, :-1]; tgt = xyz[:, 1:]
        src_c = src - src.mean(dim=2, keepdim=True)
        tgt_c = tgt - tgt.mean(dim=2, keepdim=True)
        H = torch.einsum("btni,btnj->btij", src_c, tgt_c)
        H = H + 1e-6 * torch.eye(3, device=xyz.device).view(1, 1, 3, 3)
        _, S, _ = torch.linalg.svd(H)                                          # (B, T-1, 3) sorted descending
        # log-scale and pad
        S_log = torch.log(S + 1e-6)
        S_log = torch.cat([S_log, S_log[:, -1:].clone()], dim=1)              # (B, T, 3)
        S_p = S_log.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, S_p], dim=-1)                                  # 4+3=7

    # BH: quaternion trajectory curvature on S^3 (discrete 2nd derivative)
    if variant == "BH":
        B, T, _ = qf.shape
        # zero-pad ends; q_{t+1} - 2 q_t + q_{t-1}
        q_prev = torch.cat([qf[:, :1].clone(), qf[:, :-1]], dim=1)
        q_next = torch.cat([qf[:, 1:], qf[:, -1:].clone()], dim=1)
        kappa_v = q_next - 2 * qf + q_prev                                     # (B, T, 4)
        kappa_mag = kappa_v.norm(dim=-1, keepdim=True)                         # (B, T, 1)
        feat = torch.cat([kappa_v, kappa_mag], dim=-1)                         # (B, T, 5)
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                               # 4+5=9

    # BI: binary-tetrahedral lattice distance + nearest-element 4-vector
    if variant == "BI":
        # 24 elements of 2T:
        e = torch.zeros(24, 4, device=xyz.device)
        # ±1, ±i, ±j, ±k
        for i, idx in enumerate([(0, 1), (0, -1), (1, 1), (1, -1),
                                  (2, 1), (2, -1), (3, 1), (3, -1)]):
            e[i, idx[0]] = idx[1]
        # 16 of the form (±1, ±1, ±1, ±1)/2
        sign_combos = []
        for s0 in (-1, 1):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    for s3 in (-1, 1):
                        sign_combos.append([s0, s1, s2, s3])
        for k, sg in enumerate(sign_combos):
            e[8 + k] = torch.tensor(sg, device=xyz.device, dtype=torch.float32) * 0.5
        # cosine similarity per element |q . g|, take max -> nearest
        # qf: (B, T, 4); e: (24, 4)
        sims = (qf.unsqueeze(2) * e.view(1, 1, 24, 4)).sum(-1).abs()           # (B, T, 24)
        max_sim, max_idx = sims.max(dim=-1)                                    # (B, T)
        # geodesic distance to nearest = 2*arccos(max_sim)
        dist = 2 * torch.acos(max_sim.clamp(-1+1e-6, 1-1e-6))                  # (B, T)
        # also include the nearest lattice element (4-d)
        max_idx_exp = max_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 4)
        nearest = e.view(1, 1, 24, 4).expand_as(sims.unsqueeze(-1).expand(-1, -1, -1, 4)).gather(2, max_idx_exp)[:, :, 0]
        feat = torch.cat([dist.unsqueeze(-1), nearest], dim=-1)               # (B, T, 5)
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                              # 4+5=9

    # CA: per-region Kabsch (K=4 quadrants)
    if variant == "CA":
        B, T, N, _ = xyz.shape
        # split into 4 quadrants by sign of x and y (per frame)
        feats = []
        # for each quadrant compute Kabsch between t and t+1
        src = xyz[:, :-1]; tgt = xyz[:, 1:]                                   # (B, T-1, N, 3)
        for sx in (1, -1):
            for sy in (1, -1):
                mask = ((torch.sign(src[..., 0]) == sx) & (torch.sign(src[..., 1]) == sy)).float()
                cnt = mask.sum(-1, keepdim=True).clamp(min=1.0)
                w = mask.unsqueeze(-1)
                src_mu = (src * w).sum(2, keepdim=True) / cnt.unsqueeze(-1)
                tgt_mu = (tgt * w).sum(2, keepdim=True) / cnt.unsqueeze(-1)
                src_c = src - src_mu; tgt_c = tgt - tgt_mu
                # weighted cross-cov
                H = torch.einsum("btn,btni,btnj->btij", w[..., 0], src_c, tgt_c)
                H = H + 1e-6 * torch.eye(3, device=xyz.device).view(1, 1, 3, 3)
                U, _, Vh = torch.linalg.svd(H)
                V = Vh.transpose(-1, -2)
                det = torch.det(V @ U.transpose(-1, -2))
                D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
                R = V @ D @ U.transpose(-1, -2)
                qq = rot_to_quat(R)                                            # (B, T-1, 4)
                qq = torch.cat([qq, qq[:, -1:].clone()], dim=1)               # (B, T, 4)
                feats.append(qq)
        all_q = torch.cat(feats, dim=-1)                                       # (B, T, 16)
        all_q_p = all_q.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, all_q_p], dim=-1)                              # 4+16=20

    # CB: screw axis (Plücker line representation)
    if variant == "CB":
        qf2, t_fwd = kabsch_qf_t(xyz)                                          # (B, T, 4), (B, T, 3)
        # axis direction = q_xyz / sin(theta/2); for theta=0 use any unit vector
        w = qf2[..., 0:1].clamp(-1.0+1e-6, 1.0-1e-6)
        v = qf2[..., 1:]
        s = torch.sqrt((1 - w * w).clamp(min=1e-9))
        axis = v / s.clamp(min=1e-6)                                           # (B, T, 3)
        theta = 2 * torch.acos(w.abs())                                        # (B, T, 1) rotation angle
        # pitch = (axis . t) / theta (translation along axis per radian)
        proj = (axis * t_fwd).sum(-1, keepdim=True)                            # (B, T, 1)
        pitch = proj / theta.clamp(min=1e-6)
        # moment = p × axis where p is a point on the screw axis
        # for screw motion: p = (1/2) (t - pitch * theta * axis) + (cot(theta/2)/2) (axis × t)
        # simplification using closed-form
        cot_half = w / s.clamp(min=1e-6)                                       # cot(theta/2)
        moment = 0.5 * (torch.cross(axis, t_fwd, dim=-1) * cot_half + (t_fwd - pitch * theta * axis))
        feat = torch.cat([axis, moment, pitch], dim=-1)                        # (B, T, 7)
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                               # 4+7=11

    # CE: trajectory PCA (top-2 right singular vectors of {q_t}_t)
    if variant == "CE":
        B, T, _ = qf.shape
        Q_centered = qf - qf.mean(dim=1, keepdim=True)
        # SVD: Q = U S V^T, V is (4, 4), top-2 columns
        U, S, Vh = torch.linalg.svd(Q_centered, full_matrices=False)
        V = Vh.transpose(-1, -2)                                               # (B, 4, 4)
        v12 = V[..., :2]                                                       # (B, 4, 2)
        v_flat = v12.reshape(B, 8)                                              # (B, 8)
        # broadcast to all frames + points
        v_p = v_flat.unsqueeze(1).unsqueeze(2).expand(-1, T, xyz.size(2), -1)
        return torch.cat([xyzt, v_p], dim=-1)                                   # 4+8=12

    # CK: angular velocity quaternion + acceleration
    if variant == "CK":
        B, T, _ = qf.shape
        # omega_t = q_t * conj(q_{t-1})
        q_prev = torch.cat([qf[:, :1].clone(), qf[:, :-1]], dim=1)
        q_prev_conj = torch.cat([q_prev[..., 0:1], -q_prev[..., 1:]], dim=-1)
        omega = hamilton(qf, q_prev_conj)
        omega = F.normalize(omega, dim=-1)                                     # (B, T, 4)
        # alpha_t = omega_t * conj(omega_{t-1})
        omega_prev = torch.cat([omega[:, :1].clone(), omega[:, :-1]], dim=1)
        omega_prev_conj = torch.cat([omega_prev[..., 0:1], -omega_prev[..., 1:]], dim=-1)
        alpha = hamilton(omega, omega_prev_conj)
        alpha = F.normalize(alpha, dim=-1)
        feat = torch.cat([omega, alpha], dim=-1)                              # (B, T, 8)
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                              # 4+8=12

    # CM: mean-subtracted quaternion trajectory
    if variant == "CM":
        q_mean = qf.mean(dim=1, keepdim=True)
        q_mean = F.normalize(q_mean, dim=-1)
        delta = qf - q_mean                                                    # (B, T, 4)
        delta_p = delta.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, delta_p], dim=-1)                              # 4+4=8

    # CN: multi-scale finite differences q_t - q_{t-k}, k=1,2,4,8
    if variant == "CN":
        B, T, _ = qf.shape
        diffs = []
        for k in (1, 2, 4, 8):
            q_lag = torch.cat([qf[:, :1].expand(-1, k, -1), qf[:, :-k]], dim=1)
            diffs.append(qf - q_lag)                                           # (B, T, 4)
        feat = torch.cat(diffs, dim=-1)                                        # (B, T, 16)
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                               # 4+16=20

    # CR: full Cfbq — per-point fwd + bwd Kabsch residuals (incl. translation)
    if variant == "CR":
        B, T, N, _ = xyz.shape
        qf2, t_fwd = kabsch_qf_t(xyz)                                          # (B, T, 4), (B, T, 3)
        # backward: same as fwd with src/tgt swapped
        src = xyz[:, 1:]; tgt = xyz[:, :-1]
        src_mu = src.mean(dim=2, keepdim=True); tgt_mu = tgt.mean(dim=2, keepdim=True)
        src_c = src - src_mu; tgt_c = tgt - tgt_mu
        Hb = torch.einsum("btni,btnj->btij", src_c, tgt_c)
        Hb = Hb + 1e-6 * torch.eye(3, device=xyz.device).view(1, 1, 3, 3)
        Ub, _, Vhb = torch.linalg.svd(Hb)
        Vb = Vhb.transpose(-1, -2)
        det_b = torch.det(Vb @ Ub.transpose(-1, -2))
        Db = torch.diag_embed(torch.stack([torch.ones_like(det_b), torch.ones_like(det_b), det_b], dim=-1))
        R_bwd = Vb @ Db @ Ub.transpose(-1, -2)                                 # (B, T-1, 3, 3) src=t+1 -> tgt=t
        t_bwd = tgt_mu.squeeze(2) - torch.einsum("btij,btj->bti", R_bwd, src_mu.squeeze(2))
        # for each frame t compute per-point residuals (using t-1 -> t fwd, t+1 -> t bwd)
        # rf[t, i] = xyz[t,i] - (R_fwd[t-1] @ xyz[t-1,i] + t_fwd[t-1])
        # rb[t, i] = xyz[t,i] - (R_bwd[t]   @ xyz[t+1,i] + t_bwd[t])
        # Use Kabsch (R_fwd, t_fwd) for fwd residual and (R_bwd, t_bwd) for bwd residual.
        R_fwd_full = torch.zeros(B, T, 3, 3, device=xyz.device)
        t_fwd_full = torch.zeros(B, T, 3, device=xyz.device)
        # we already have fwd Rs from kabsch_qf_t indirectly; recompute via SVD again for full T
        H_fwd = torch.einsum("btni,btnj->btij",
                             xyz[:, :-1] - xyz[:, :-1].mean(2, keepdim=True),
                             xyz[:, 1:] - xyz[:, 1:].mean(2, keepdim=True))
        H_fwd = H_fwd + 1e-6 * torch.eye(3, device=xyz.device).view(1, 1, 3, 3)
        Uf, _, Vhf = torch.linalg.svd(H_fwd)
        Vf = Vhf.transpose(-1, -2)
        det_f = torch.det(Vf @ Uf.transpose(-1, -2))
        Df = torch.diag_embed(torch.stack([torch.ones_like(det_f), torch.ones_like(det_f), det_f], dim=-1))
        R_fwd_pair = Vf @ Df @ Uf.transpose(-1, -2)                            # (B, T-1, 3, 3)
        R_fwd_full[:, 1:] = R_fwd_pair
        R_bwd_full = torch.zeros(B, T, 3, 3, device=xyz.device)
        R_bwd_full[:, :-1] = R_bwd
        t_fwd_full[:, 1:] = t_fwd[:, :-1]                                       # t_fwd applies t-1 -> t at position t
        t_bwd_full = torch.zeros(B, T, 3, device=xyz.device)
        t_bwd_full[:, :-1] = t_bwd                                              # t_bwd applies t+1 -> t at position t
        # apply rotations per-point
        # forward residual at frame t (>0): xyz[t] - (R_fwd[t] @ xyz[t-1] + t_fwd[t])
        rf = torch.zeros_like(xyz)
        rb = torch.zeros_like(xyz)
        for t_i in range(1, T):
            R = R_fwd_full[:, t_i]                                              # (B, 3, 3)
            tt = t_fwd_full[:, t_i].unsqueeze(1)                                # (B, 1, 3)
            pred = torch.einsum("bij,bnj->bni", R, xyz[:, t_i - 1]) + tt
            rf[:, t_i] = xyz[:, t_i] - pred
        for t_i in range(0, T - 1):
            R = R_bwd_full[:, t_i]
            tt = t_bwd_full[:, t_i].unsqueeze(1)
            pred = torch.einsum("bij,bnj->bni", R, xyz[:, t_i + 1]) + tt
            rb[:, t_i] = xyz[:, t_i] - pred
        return torch.cat([xyzt, rf, rb], dim=-1)                                # 4 + 3 + 3 = 10

    # CO: pairwise distance pooling (per-frame anomaly statistics)
    if variant == "CO":
        # ||q_t - q_s||^2 for all (t, s), then per-row stats
        # qf: (B, T, 4)
        diff = qf.unsqueeze(2) - qf.unsqueeze(1)                                # (B, T, T, 4)
        d2 = (diff * diff).sum(-1)                                              # (B, T, T)
        # mask diagonal (self)
        T_ = qf.shape[1]
        eye = torch.eye(T_, device=xyz.device).bool()
        d2 = d2.masked_fill(eye.unsqueeze(0), float('inf'))
        d_min = d2.min(dim=-1).values                                          # (B, T)
        d2 = d2.masked_fill(eye.unsqueeze(0), 0)
        d_max = d2.max(dim=-1).values
        d_mean = d2.sum(-1) / max(1, T_ - 1)
        d_var = ((d2 - d_mean.unsqueeze(-1)) ** 2).sum(-1) / max(1, T_ - 1)
        feat = torch.stack([d_min, d_max, d_mean, d_var], dim=-1)              # (B, T, 4)
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                               # 4+4=8

    # CL: quaternion outer product (10-d symmetric)
    if variant == "CL":
        # Q_ij = q_i q_j; symmetric 4x4 -> upper-triangular 10 entries
        q = qf
        outer = q.unsqueeze(-1) * q.unsqueeze(-2)                              # (B, T, 4, 4)
        # extract upper-triangular incl. diagonal
        i, j = torch.triu_indices(4, 4)
        flat = outer[..., i, j]                                                 # (B, T, 10)
        flat_p = flat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, flat_p], dim=-1)                                # 4+10=14

    # BJ: quaternion power decomposition (q^{1/2} and q^2)
    if variant == "BJ":
        # q^{1/2}: half-rotation. SLERP(identity, q, 0.5)
        # cos(theta/2) -> cos(theta/4), axis preserved, scale.
        w = qf[..., 0:1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        v = qf[..., 1:]
        nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        half_theta = torch.acos(w.abs())                                      # = theta/2
        quarter_theta = half_theta / 2
        q_half_w = torch.cos(quarter_theta)
        q_half_v = (v / nrm) * torch.sin(quarter_theta)
        q_half = torch.cat([q_half_w, q_half_v], dim=-1)                       # 4-d
        # q^2 via Hamilton product q*q
        q_sq = hamilton(qf, qf)                                                # 4-d
        feat = torch.cat([q_half, q_sq], dim=-1)                              # 8-d
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                              # 4+8=12

    raise ValueError(variant)


# ============= model =============
class TinyPointNet(nn.Module):
    def __init__(self, in_ch, num_classes=2, hidden=128):
        super().__init__()
        self.pt = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.tc = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden//2, num_classes),
        )

    def forward(self, x):
        h = self.pt(x)
        per_frame = torch.cat([h.max(2).values, h.mean(2)], dim=-1)
        h_seq = self.tc(per_frame.transpose(1, 2))
        return self.head(h_seq.max(-1).values)


# ============= data =============
def load_subset(phase, classes):
    ds = nd.NvidiaLoader(framerate=32, phase=phase)
    pts_list, lbl_list = [], []
    for i in range(len(ds)):
        x, y, _ = ds[i]
        if int(y) not in classes:
            continue
        if hasattr(x, "numpy"):
            x = x.numpy()
        pts_list.append(np.asarray(x[..., 4:8], dtype=np.float32))
        lbl_list.append(classes.index(int(y)))
    X = np.stack(pts_list, 0)
    y = np.array(lbl_list, dtype=np.int64)
    return X, y


def normalize(X, mu=None, sd=None):
    flat = X.reshape(-1, 4)
    if mu is None:
        mu = flat.mean(0); sd = flat.std(0).clip(min=1.0)
    return (X - mu) / sd, mu, sd


def run_one(variant, seed, Xtr_t, ytr_t, Xte_t, yte_t, epochs=60, bs=8, P=128):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    dummy = build_features(Xtr_t[:1, :, :P], variant)
    in_ch = dummy.shape[-1]
    model = TinyPointNet(in_ch, num_classes=2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Ntr = Xtr_t.shape[0]
    best = 0.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Ntr, device=DEV)
        ridx = torch.randperm(Xtr_t.shape[2], device=DEV)[:P]
        Xtr_use = Xtr_t.index_select(2, ridx)
        for i in range(0, Ntr, bs):
            idx = perm[i:i+bs]
            x_in = build_features(Xtr_use[idx], variant)
            logits = model(x_in)
            loss = F.cross_entropy(logits, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        te_idx = torch.linspace(0, Xte_t.shape[2] - 1, P, device=DEV).long()
        Xte_use = Xte_t.index_select(2, te_idx)
        with torch.no_grad():
            logits = model(build_features(Xte_use, variant))
            acc = (logits.argmax(-1) == yte_t).float().mean().item()
        if acc > best: best = acc
    return best


VARIANTS_TO_TEST = ["A", "CR", "CM", "CN", "CO"]


def run_pair(pair, n_seeds=5):
    classes = list(pair)
    Xtr, ytr = load_subset("train", classes)
    Xte, yte = load_subset("test", classes)
    Xtr, mu, sd = normalize(Xtr); Xte, _, _ = normalize(Xte, mu, sd)
    Xtr_t = torch.from_numpy(Xtr).to(DEV); ytr_t = torch.from_numpy(ytr).to(DEV)
    Xte_t = torch.from_numpy(Xte).to(DEV); yte_t = torch.from_numpy(yte).to(DEV)
    out = {}
    for v in VARIANTS_TO_TEST:
        accs = [run_one(v, s, Xtr_t, ytr_t, Xte_t, yte_t) for s in range(n_seeds)]
        out[v] = (float(np.mean(accs)), float(np.std(accs)))
    return Xtr.shape[0], Xte.shape[0], out


def main():
    cols = "  ".join(f"{v:>10s}" for v in VARIANTS_TO_TEST)
    print(f"PAIR    n_tr n_te   {cols}")
    print("-" * 110)
    grid = {}
    for (a, b) in PAIRS:
        t0 = time.time()
        ntr, nte, res = run_pair((a, b))
        cells = "  ".join(f"{m*100:5.2f}±{s*100:.2f}" for m, s in res.values())
        print(f"{a:2d}-{b:2d}     {ntr:3d}  {nte:3d}    {cells}  ({time.time()-t0:.0f}s)")
        grid[(a, b)] = res

    print("\n=== summary: mean delta (variant - A) per variant across pairs ===")
    a_means = {p: g["A"][0] for p, g in grid.items()}
    for v in VARIANTS_TO_TEST:
        if v == "A": continue
        deltas = [grid[p][v][0] - a_means[p] for p in grid]
        n_pos = sum(1 for d in deltas if d > 0)
        n_strong = sum(1 for d in deltas if d >= 0.02)
        print(f"  {v}: mean Δ {np.mean(deltas)*100:+.2f}pp  median {np.median(deltas)*100:+.2f}pp  pos {n_pos}/{len(deltas)}  ≥+2pp {n_strong}/{len(deltas)}")


if __name__ == "__main__":
    main()
