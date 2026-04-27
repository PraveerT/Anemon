"""Quaternion-cycle sweep on class-3 vs class-16 binary subset.

Class 3 = Moving hand down, Class 16 = Pushing hand down — the hardest confusion pair.
PMamba: 7/10 wrong class-3 samples are predicted as class-16.

This script iterates over several quaternion-cycle feature variants, trains a tiny
PointNet-style classifier from scratch on the 2-class subset, reports test acc.

Goal: find any feature variant that gives a meaningful binary signal beyond
the xyzt-only baseline.

Variants:
  A. xyzt baseline                          (4ch  per-point)
  B. xyzt + Kabsch fwd quaternion           (4 + 4 broadcast per frame)
  C. xyzt + cycle residual quaternion       (qf*qb - I, 4 broadcast per frame)
  D. xyzt + dual-quat SE(3) (qf, t)         (4+3=7 broadcast per frame)
  E. xyzt + per-point cycle residual        (4 + 3, per-point trivial cycle)
  F. cycle-only (no xyz)                    (4 broadcast — quat features alone)
"""
import os, math, sys, time, requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/notebooks/PMamba/experiments")
import nvidia_dataloader as nd

DEV = torch.device("cuda")
TARGET_CLASSES = [3, 16]   # default binary subset (overridden via env CLASS_PAIR)
WORK = "/notebooks/PMamba/experiments/work_dir/quat_cycle_sweep"
os.makedirs(WORK, exist_ok=True)
TG = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"


def tg(msg):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            cid = r["result"][-1]["message"]["chat"]["id"]
            requests.post(f"https://api.telegram.org/bot{TG}/sendMessage",
                data={"chat_id": cid, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


# ---------- quaternion utilities ----------
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
    # R: (..., 3, 3)
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


def kabsch_quat_per_frame(xyz):
    """xyz: (B, T, N, 3). Compute (qf, qb, t) per consecutive frame pair.
    Returns qf (B, T, 4), qb (B, T, 4), tr (B, T, 3). Last frame copies prev (no t+1).
    """
    B, T, N, _ = xyz.shape
    src = xyz[:, :-1]                              # (B, T-1, N, 3)
    tgt = xyz[:, 1:]
    src_c = src - src.mean(dim=2, keepdim=True)
    tgt_c = tgt - tgt.mean(dim=2, keepdim=True)
    H = torch.einsum("btni,btnj->btij", src_c, tgt_c)
    H = H + 1e-6 * torch.eye(3, device=xyz.device).view(1, 1, 3, 3)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R_fwd = V @ D @ U.transpose(-1, -2)            # (B, T-1, 3, 3) src->tgt
    R_bwd = R_fwd.transpose(-1, -2)
    qf = rot_to_quat(R_fwd)
    qb = rot_to_quat(R_bwd)
    # translation
    t_fwd = tgt.mean(dim=2) - torch.einsum("btij,btj->bti", R_fwd, src.mean(dim=2))
    # pad last frame
    qf = torch.cat([qf, qf[:, -1:].clone()], dim=1)
    qb = torch.cat([qb, qb[:, -1:].clone()], dim=1)
    t_fwd = torch.cat([t_fwd, t_fwd[:, -1:].clone()], dim=1)
    return qf, qb, t_fwd


# ---------- feature builders ----------
def build_features(xyzt, variant):
    """xyzt: (B, T, N, 4). Returns x: (B, T, N, C) suitable for the model."""
    xyz = xyzt[..., :3]
    t   = xyzt[..., 3:4]
    if variant == "A":
        return xyzt                                        # 4ch
    qf, qb, tr = kabsch_quat_per_frame(xyz)               # (B, T, 4), (B, T, 4), (B, T, 3)
    if variant == "B":
        qf_p = qf.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, qf_p], dim=-1)             # 4+4=8
    if variant == "C":
        cyc = hamilton(qf, qb)                             # ~identity
        ident = torch.tensor([1, 0, 0, 0], device=xyz.device, dtype=cyc.dtype)
        cyc = cyc - ident
        cyc_p = cyc.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, cyc_p], dim=-1)            # 4+4=8
    if variant == "D":
        qf_p = qf.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        tr_p = tr.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, qf_p, tr_p], dim=-1)       # 4+4+3=11
    if variant == "E":
        # per-point fwd cycle residual: align xyz_t -> xyz_{t+1} then cycle back
        # use the global qf rotation per frame (broadcast)
        # residual = xyz_{t+1} - rotate(qf, xyz_t)
        # this is the per-point Kabsch residual
        B, T, N, _ = xyz.shape
        rot = torch.zeros(B, T, N, 3, device=xyz.device)
        for ti in range(T - 1):
            q = qf[:, ti]                                   # (B, 4)
            v = xyz[:, ti]                                  # (B, N, 3)
            qv = torch.cat([torch.zeros(B, N, 1, device=xyz.device), v], dim=-1)
            qb_local = torch.cat([q[:, 0:1], -q[:, 1:]], dim=-1)
            rotated = hamilton(hamilton(q.unsqueeze(1).expand(-1, N, -1), qv),
                               qb_local.unsqueeze(1).expand(-1, N, -1))[..., 1:]
            rot[:, ti + 1] = xyz[:, ti + 1] - rotated
        return torch.cat([xyzt, rot], dim=-1)              # 4+3=7
    if variant == "F":
        qf_p = qf.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([t, qf_p], dim=-1)                # 1+4=5  (no xyz)

    # G: angle + axis decomposition
    if variant == "G":
        # angle = 2 * acos(qw), axis = q[1:] / sin(angle/2)
        qw = qf[..., 0:1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        ang = 2 * torch.acos(qw.abs())                                    # (B, T, 1)
        axis = F.normalize(qf[..., 1:], dim=-1)                            # (B, T, 3)
        feat = torch.cat([ang, axis], dim=-1)                              # (B, T, 4)
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                           # 4+4=8

    # H: just rotation magnitude (motion energy)
    if variant == "H":
        qw = qf[..., 0:1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        ang = 2 * torch.acos(qw.abs())                                    # (B, T, 1)
        ang_p = ang.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, ang_p], dim=-1)                            # 4+1=5

    # I: cumulative rotation from frame 0 (trajectory)
    if variant == "I":
        # walk forward composing qf[0..t]
        B, T, _ = qf.shape
        cum = torch.zeros_like(qf)
        cum[:, 0] = torch.tensor([1.0, 0, 0, 0], device=xyz.device).expand(B, -1)
        for ti in range(1, T):
            cum[:, ti] = hamilton(cum[:, ti - 1], qf[:, ti - 1])
            cum[:, ti] = F.normalize(cum[:, ti], dim=-1)
        cum_p = cum.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, cum_p], dim=-1)                            # 4+4=8

    # J: non-rigid divergence — per-point displacement minus rigid-body fit
    if variant == "J":
        B, T, N, _ = xyz.shape
        # rigid-body fit per-frame xyz_t -> xyz_{t+1} using qf
        nonrigid = torch.zeros(B, T, N, 3, device=xyz.device)
        for ti in range(T - 1):
            q = qf[:, ti]
            v = xyz[:, ti]
            qv = torch.cat([torch.zeros(B, N, 1, device=xyz.device), v], dim=-1)
            qb_local = torch.cat([q[:, 0:1], -q[:, 1:]], dim=-1)
            rotated = hamilton(hamilton(q.unsqueeze(1).expand(-1, N, -1), qv),
                               qb_local.unsqueeze(1).expand(-1, N, -1))[..., 1:]
            # nonrigid = obs - (rotated + translation)
            t_fwd = tr[:, ti].unsqueeze(1)                                # (B, 1, 3)
            nonrigid[:, ti] = xyz[:, ti + 1] - (rotated + t_fwd)
        return torch.cat([xyzt, nonrigid], dim=-1)                         # 4+3=7

    # L: per-frame velocity (xyz_{t+1} - xyz_t per point)
    if variant == "L":
        B, T, N, _ = xyz.shape
        vel = torch.zeros_like(xyz)
        vel[:, :-1] = xyz[:, 1:] - xyz[:, :-1]
        return torch.cat([xyzt, vel], dim=-1)                              # 4+3=7

    # M: per-frame acceleration
    if variant == "M":
        B, T, N, _ = xyz.shape
        acc = torch.zeros_like(xyz)
        acc[:, 1:-1] = xyz[:, 2:] - 2 * xyz[:, 1:-1] + xyz[:, :-2]
        return torch.cat([xyzt, acc], dim=-1)                              # 4+3=7

    # N: K + velocity (xyzt + cum + nonrigid + velocity)
    if variant == "N":
        B, T, N, _ = xyz.shape
        cum = torch.zeros_like(qf)
        cum[:, 0] = torch.tensor([1.0, 0, 0, 0], device=xyz.device).expand(B, -1)
        for ti in range(1, T):
            cum[:, ti] = F.normalize(hamilton(cum[:, ti - 1], qf[:, ti - 1]), dim=-1)
        nonrigid = torch.zeros(B, T, N, 3, device=xyz.device)
        for ti in range(T - 1):
            q = qf[:, ti]; v = xyz[:, ti]
            qv = torch.cat([torch.zeros(B, N, 1, device=xyz.device), v], dim=-1)
            qb_local = torch.cat([q[:, 0:1], -q[:, 1:]], dim=-1)
            rotated = hamilton(hamilton(q.unsqueeze(1).expand(-1, N, -1), qv),
                               qb_local.unsqueeze(1).expand(-1, N, -1))[..., 1:]
            nonrigid[:, ti] = xyz[:, ti + 1] - (rotated + tr[:, ti].unsqueeze(1))
        vel = torch.zeros_like(xyz); vel[:, :-1] = xyz[:, 1:] - xyz[:, :-1]
        cum_p = cum.unsqueeze(2).expand(-1, -1, N, -1)
        return torch.cat([xyzt, cum_p, nonrigid, vel], dim=-1)             # 4+4+3+3=14

    # O: per-frame point spread (std of xyz per frame, broadcast)
    if variant == "O":
        spread = xyz.std(dim=2, keepdim=True).expand(-1, -1, xyz.size(2), -1)  # (B,T,N,3)
        return torch.cat([xyzt, spread], dim=-1)                           # 4+3=7

    # P: cumulative angle (just scalar magnitude trajectory)
    if variant == "P":
        B, T, _ = qf.shape
        cum = torch.zeros_like(qf)
        cum[:, 0] = torch.tensor([1.0, 0, 0, 0], device=xyz.device).expand(B, -1)
        for ti in range(1, T):
            cum[:, ti] = F.normalize(hamilton(cum[:, ti - 1], qf[:, ti - 1]), dim=-1)
        cum_w = cum[..., 0:1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        cum_ang = 2 * torch.acos(cum_w.abs())
        cum_ang_p = cum_ang.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, cum_ang_p], dim=-1)                        # 4+1=5

    # Q: skip-2 cumulative cycle (q_{t,t+2} vs q_{t,t+1}*q_{t+1,t+2})
    if variant == "Q":
        # compose qf[t] * qf[t+1] for skip-2 prediction
        B, T, _ = qf.shape
        skip = torch.zeros_like(qf)
        for ti in range(T - 1):
            skip[:, ti] = F.normalize(hamilton(qf[:, ti], qf[:, min(ti + 1, T - 1)]), dim=-1)
        skip[:, -1] = skip[:, -2]
        skip_p = skip.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, skip_p], dim=-1)                           # 4+4=8

    # R: rigid translation only (3ch broadcast)
    if variant == "R":
        tr_p = tr.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, tr_p], dim=-1)                             # 4+3=7

    # S: per-point Euclidean displacement magnitude
    if variant == "S":
        B, T, N, _ = xyz.shape
        disp_mag = torch.zeros(B, T, N, 1, device=xyz.device)
        disp_mag[:, :-1] = (xyz[:, 1:] - xyz[:, :-1]).norm(dim=-1, keepdim=True)
        return torch.cat([xyzt, disp_mag], dim=-1)                         # 4+1=5

    # ============================================================
    # 5 NOVEL QCC FORMULATIONS (N1..N5 → letters U,V,W,X,Y)
    # ============================================================

    # U  (N1): Karcher geodesic residual on S^3
    # local geodesic mean over [t-1, t, t+1], one Newton step on manifold
    if variant == "U":
        # ensure quat double-cover canonicalization for averaging
        qf2 = qf.clone()
        # log map at p: log_p(q) = (theta / sin(theta)) * (q - <p,q>p)
        # for one Newton step: take arithmetic mean of log_p(q_s), then exp_p
        B, T, _ = qf2.shape
        # for each t use window {t-1, t, t+1} clamped
        idxs = torch.stack([torch.clamp(torch.tensor([t - 1, t, t + 1]), 0, T - 1)
                             for t in range(T)], dim=0).to(xyz.device)  # (T, 3)
        win_q = qf2[:, idxs]                                              # (B, T, 3, 4)
        # canonicalize signs to align with q_t (avoid antipodal cancellation)
        center = qf2.unsqueeze(2)                                         # (B, T, 1, 4)
        signs = torch.sign((win_q * center).sum(-1, keepdim=True) + 1e-9) # (B, T, 3, 1)
        win_q = win_q * signs
        # mean and renormalize -> Newton step approximation
        qbar = win_q.mean(dim=2)
        qbar = F.normalize(qbar, dim=-1)
        # log_qbar(q_t) tangent residual ~ q_t - <qbar, q_t> qbar  (chord approx)
        proj = (qbar * qf2).sum(-1, keepdim=True)
        chord = qf2 - proj * qbar                                         # tangent at qbar
        # convert 4-d chord to 3-d via Householder mapping in tangent: drop the qbar-aligned coord
        # simplest: scale chord by 2/||chord+eps|| to get rotation magnitude scale; output 4d for stability
        residual = chord                                                  # (B, T, 4)
        residual_p = residual.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, residual_p], dim=-1)                      # 4+4=8

    # V  (N2): BCH curvature (cumulative-sum-in-algebra vs log-of-cumulative-product)
    if variant == "V":
        B, T, _ = qf.shape
        # log map: q -> 2*acos(w) * (x,y,z)/||(x,y,z)||
        def log_q(q):
            w = q[..., 0:1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            v = q[..., 1:]
            n = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            theta = torch.acos(w.abs())                                    # half-angle
            return 2 * theta * v / n                                       # (..., 3) axis-angle
        xi = log_q(qf)                                                     # (B, T, 3)
        Sigma = torch.cumsum(xi, dim=1) - xi                               # sum over s<t (B, T, 3)
        # cumulative product in S^3
        cum_prod = torch.zeros_like(qf)
        cum_prod[:, 0] = torch.tensor([1.0, 0, 0, 0], device=xyz.device).expand(B, -1)
        for ti in range(1, T):
            cum_prod[:, ti] = F.normalize(hamilton(qf[:, ti - 1], cum_prod[:, ti - 1]), dim=-1)
        log_cum_prod = log_q(cum_prod)                                     # (B, T, 3)
        kappa = Sigma - log_cum_prod                                       # BCH curvature (B, T, 3)
        kappa_p = kappa.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, kappa_p], dim=-1)                          # 4+3=7

    # W  (N3): rotation-plane bivector (5-d symmetric traceless)
    if variant == "W":
        # axis hat_n = q_xyz / sin(theta/2) ; for theta -> 0 use q_xyz directly (still defines plane)
        v = qf[..., 1:]
        nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        nh = v / nrm                                                       # (B, T, 3)
        # B = nh ⊗ nh - (1/3) I  -> vectorize 5 indep components
        nx, ny, nz = nh[..., 0:1], nh[..., 1:2], nh[..., 2:3]
        b11 = nx * nx - 1.0 / 3
        b22 = ny * ny - 1.0 / 3
        b12 = nx * ny
        b13 = nx * nz
        b23 = ny * nz
        beta = torch.cat([b11, b22, b12, b13, b23], dim=-1)                # (B, T, 5)
        beta_p = beta.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, beta_p], dim=-1)                           # 4+5=9

    # X  (N4): temporal Wigner-lite spectrum (DCT of quaternion components, K=4)
    if variant == "X":
        K = 4
        B, T, _ = qf.shape
        t_idx = torch.arange(T, device=xyz.device).float()
        k_idx = torch.arange(K, device=xyz.device).float()
        basis = torch.cos((torch.pi / T) * (t_idx.unsqueeze(0) + 0.5) * k_idx.unsqueeze(1))   # (K, T)
        # qf: (B, T, 4) -> for each component c, project onto K basis vectors
        spectrum = torch.einsum("kt,btc->bkc", basis, qf)                   # (B, K, 4)
        spec_flat = spectrum.reshape(B, K * 4)                              # (B, 16)
        # broadcast to per-frame, per-point
        spec_p = spec_flat.unsqueeze(1).unsqueeze(2).expand(-1, T, xyz.size(2), -1)
        return torch.cat([xyzt, spec_p], dim=-1)                            # 4 + 16 = 20

    # Y  (N5): rotor commutator (cross of consecutive rotation-vector parts)
    if variant == "Y":
        v = qf[..., 1:]                                                    # (B, T, 3)
        v_next = torch.cat([v[:, 1:], v[:, -1:].clone()], dim=1)
        cross = torch.cross(v, v_next, dim=-1)                             # (B, T, 3)
        # also include magnitude (= |sin(theta_a)*sin(theta_b)*sin(angle_between_axes)|)
        mag = cross.norm(dim=-1, keepdim=True)
        feat = torch.cat([cross, mag], dim=-1)                             # (B, T, 4)
        feat_p = feat.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        return torch.cat([xyzt, feat_p], dim=-1)                           # 4+4=8

    # T: K + velocity magnitude broadcast
    if variant == "T":
        B, T, N, _ = xyz.shape
        cum = torch.zeros_like(qf)
        cum[:, 0] = torch.tensor([1.0, 0, 0, 0], device=xyz.device).expand(B, -1)
        for ti in range(1, T):
            cum[:, ti] = F.normalize(hamilton(cum[:, ti - 1], qf[:, ti - 1]), dim=-1)
        vel_mag = (tr.norm(dim=-1, keepdim=True))                          # per-frame translation magnitude
        vel_p = vel_mag.unsqueeze(2).expand(-1, -1, N, -1)
        cum_p = cum.unsqueeze(2).expand(-1, -1, N, -1)
        return torch.cat([xyzt, cum_p, vel_p], dim=-1)                     # 4+4+1=9

    # K: I + J fused
    if variant == "K":
        B, T, N, _ = xyz.shape
        cum = torch.zeros_like(qf)
        cum[:, 0] = torch.tensor([1.0, 0, 0, 0], device=xyz.device).expand(B, -1)
        for ti in range(1, T):
            cum[:, ti] = F.normalize(hamilton(cum[:, ti - 1], qf[:, ti - 1]), dim=-1)
        nonrigid = torch.zeros(B, T, N, 3, device=xyz.device)
        for ti in range(T - 1):
            q = qf[:, ti]
            v = xyz[:, ti]
            qv = torch.cat([torch.zeros(B, N, 1, device=xyz.device), v], dim=-1)
            qb_local = torch.cat([q[:, 0:1], -q[:, 1:]], dim=-1)
            rotated = hamilton(hamilton(q.unsqueeze(1).expand(-1, N, -1), qv),
                               qb_local.unsqueeze(1).expand(-1, N, -1))[..., 1:]
            t_fwd = tr[:, ti].unsqueeze(1)
            nonrigid[:, ti] = xyz[:, ti + 1] - (rotated + t_fwd)
        cum_p = cum.unsqueeze(2).expand(-1, -1, N, -1)
        return torch.cat([xyzt, cum_p, nonrigid], dim=-1)                  # 4+4+3=11

    raise ValueError(variant)


# ---------- tiny model ----------
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
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, _ = x.shape
        h = self.pt(x)                                    # (B, T, N, H)
        per_frame = torch.cat([h.max(2).values, h.mean(2)], dim=-1)   # (B, T, 2H)
        h_seq = self.tc(per_frame.transpose(1, 2))        # (B, H, T)
        return self.head(h_seq.max(-1).values)


# ---------- data ----------
def load_subset(phase):
    ds = nd.NvidiaLoader(framerate=32, phase=phase)
    pts_list, lbl_list = [], []
    for i in range(len(ds)):
        x, y, _ = ds[i]
        if int(y) not in TARGET_CLASSES:
            continue
        if hasattr(x, "numpy"):
            x = x.numpy()
        # use channels 4..8 (xyzt — native 3D + time, after uvd2xyz)
        pts_list.append(np.asarray(x[..., 4:8], dtype=np.float32))
        lbl_list.append(TARGET_CLASSES.index(int(y)))
    X = np.stack(pts_list, 0)
    y = np.array(lbl_list, dtype=np.int64)
    return X, y


def normalize(X, mu=None, sd=None):
    flat = X.reshape(-1, 4)
    if mu is None:
        mu = flat.mean(0); sd = flat.std(0).clip(min=1.0)
    return (X - mu) / sd, mu, sd


def run_variant_one(variant, seed, Xtr_t, ytr_t, Xte_t, yte_t, epochs=60, bs=8, P=128):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    # Determine in_ch via dry-run on small slice
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
    return best, in_ch


def run_variant(variant, Xtr, ytr, Xte, yte, epochs=60, bs=8, P=128, n_seeds=5):
    Xtr_t = torch.from_numpy(Xtr).to(DEV)
    Xte_t = torch.from_numpy(Xte).to(DEV)
    ytr_t = torch.from_numpy(ytr).to(DEV)
    yte_t = torch.from_numpy(yte).to(DEV)
    accs = []
    in_ch_final = None
    for seed in range(n_seeds):
        acc, in_ch = run_variant_one(variant, seed, Xtr_t, ytr_t, Xte_t, yte_t, epochs, bs, P)
        accs.append(acc)
        in_ch_final = in_ch
    return float(np.mean(accs)), float(np.std(accs)), in_ch_final


def main():
    print("loading subset...")
    Xtr, ytr = load_subset("train")
    Xte, yte = load_subset("test")
    print(f"  train: {Xtr.shape}  labels: {np.bincount(ytr)}")
    print(f"  test : {Xte.shape}  labels: {np.bincount(yte)}")

    Xtr, mu, sd = normalize(Xtr)
    Xte, _, _ = normalize(Xte, mu, sd)
    print(f"normalized: mu={mu}  sd={sd}")

    tg(f"<b>QuatCycle Sweep</b> binary {TARGET_CLASSES}\n"
       f"train {Xtr.shape[0]}  test {Xte.shape[0]}")

    # Reduced focused sweep on baseline + the 5 novel formulations
    VARIANTS = ["A",          # baseline
                "U", "V", "W", "X", "Y"]   # N1..N5 novel
    results = {}
    for v in VARIANTS:
        t0 = time.time()
        try:
            mean_acc, std_acc, in_ch = run_variant(v, Xtr, ytr, Xte, yte, n_seeds=3)
        except Exception as e:
            print(f"variant {v} FAILED: {e}")
            tg(f"variant {v}: FAILED ({e})")
            results[v] = None
            continue
        dt = time.time() - t0
        results[v] = (mean_acc, std_acc, in_ch)
        line = f"variant {v}  in_ch={in_ch}  acc {mean_acc*100:.2f}±{std_acc*100:.2f}%  ({dt:.0f}s)"
        print(line)
        tg(f"<b>{v}</b> in={in_ch}  <b>{mean_acc*100:.2f}±{std_acc*100:.2f}%</b>  ({dt:.0f}s)")

    print("\n=== summary (sorted) ===")
    sorted_r = sorted([(v, r) for v, r in results.items() if r],
                      key=lambda x: x[1][0], reverse=True)
    lines = []
    for v, (m, s, ic) in sorted_r:
        ln = f"  {v}: {m*100:.2f}±{s*100:.2f}%  (in_ch={ic})"
        print(ln); lines.append(ln)
    tg(f"<b>Sweep done — sorted</b>\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
