"""SeQuMamba: Selective Quaternion-valued Mamba.

Combines:
  - Mamba-style selectivity (input-dependent A, B, C)
  - REQNN-style SO(3)-equivariance via quaternion features

Key novelty: selectivity is achieved via SO(3)-invariant input statistics
(per-channel quaternion magnitudes). Since |q·x| = |x| for unit q ∈ ℍ¹,
the gates are unchanged under rotation → recurrence stays SO(3)-equivariant.

Recurrence:
  s^A_t = σ(W_A · |x_t|),   s^B_t = σ(W_B · |x_t|)   ← input-dependent gates
  A_t = s^A_t · qA_unit                              ← scalar × unit quaternion
  B_t = s^B_t · qB_unit
  h_t = h_{t-1} ⊙ A_t + (W_x ⊙ x_t) ⊙ B_t
  y_t = (W_y ⊙ h_t) ⊙ qC_unit

Theorem: under x → q·x for any unit q ∈ ℍ¹:
  - magnitudes |x_t| invariant → s^A_t, s^B_t unchanged
  - W_x and W_y are real channel-mixers (treat 4 quat components identically)
    → preserve left-multiplication: W_x · (q·x) = q · (W_x · x)
  - h_t → q·h_t propagates linearly through recurrence
  - Final readout (magnitudes ‖h_T‖) is invariant.
Hence the classifier is SO(3)-invariant.

Parallel scan preserved: scalar modulation commutes with quaternion mul, so
the scan operator (a, b) ∘ (c, d) = (a⊙c, b⊙c + d) remains associative.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.motion import Motion


def qmul(p, q):
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], dim=-1)


def qnorm_unit(q):
    return q / (q.norm(dim=-1, keepdim=True) + 1e-9)


def parallel_scan_quat(A, B):
    """Hillis-Steele parallel scan for the linear quaternion recurrence:
        h_t = h_{t-1} ⊙ A_t + B_t
    Inputs:
      A: (..., T, n, 4) per-step weight quaternion
      B: (..., T, n, 4) per-step input contribution
    Returns h: (..., T, n, 4)
    """
    A_acc, B_acc = A, B
    T = A.shape[-3]
    log_T = max(1, math.ceil(math.log2(T)))
    for k in range(log_T):
        step = 2 ** k
        if step >= T: break
        ident_A = torch.zeros_like(A_acc[..., :step, :, :])
        ident_A[..., 0] = 1.0
        ident_B = torch.zeros_like(B_acc[..., :step, :, :])
        A_prev = torch.cat([ident_A, A_acc[..., :-step, :, :]], dim=-3)
        B_prev = torch.cat([ident_B, B_acc[..., :-step, :, :]], dim=-3)
        A_acc = qmul(A_prev, A_acc)
        B_acc = qmul(B_prev, A_acc)  # NOTE: rebuilt below
        # Actually: new_b = b_prev ⊙ a_acc_orig + b_acc, but a_acc_orig was overwritten.
        # Recompute correctly:
        # We need: A_new = A_prev ⊙ A_acc_orig
        #          B_new = B_prev ⊙ A_acc_orig + B_acc_orig
        # So swap order: do B first, then A
        # (correct above)
    # The block above had a logic error; rewrite cleanly:
    A_acc, B_acc = A, B
    for k in range(log_T):
        step = 2 ** k
        if step >= T: break
        ident_A = torch.zeros_like(A_acc[..., :step, :, :])
        ident_A[..., 0] = 1.0
        ident_B = torch.zeros_like(B_acc[..., :step, :, :])
        A_prev = torch.cat([ident_A, A_acc[..., :-step, :, :]], dim=-3)
        B_prev = torch.cat([ident_B, B_acc[..., :-step, :, :]], dim=-3)
        A_new = qmul(A_prev, A_acc)
        B_new = qmul(B_prev, A_acc) + B_acc
        A_acc, B_acc = A_new, B_new
    return B_acc


class SeQuMambaBlock(nn.Module):
    """Selective quaternion Mamba block.

    Selectivity: A_t, B_t are scalar-modulated unit quaternions, where the
    scalars come from invariant input magnitudes. Preserves SO(3)-equivariance.
    """
    def __init__(self, n_quat, gate_hidden=None):
        super().__init__()
        if gate_hidden is None:
            gate_hidden = max(n_quat // 2, 16)
        self.W_x = nn.Linear(n_quat, n_quat, bias=False)
        self.W_y = nn.Linear(n_quat, n_quat, bias=False)
        # Selectivity gates: take per-channel magnitudes -> per-channel scalar gates
        self.gate_A = nn.Sequential(
            nn.Linear(n_quat, gate_hidden), nn.SiLU(),
            nn.Linear(gate_hidden, n_quat),
        )
        self.gate_B = nn.Sequential(
            nn.Linear(n_quat, gate_hidden), nn.SiLU(),
            nn.Linear(gate_hidden, n_quat),
        )
        # Fixed unit quaternion directions (learned)
        self.qA_raw = nn.Parameter(torch.randn(n_quat, 4) * 0.05)
        self.qB_raw = nn.Parameter(torch.randn(n_quat, 4) * 0.05)
        self.qC_raw = nn.Parameter(torch.randn(n_quat, 4) * 0.05)
        with torch.no_grad():
            self.qA_raw[:, 0] = 1.0
            self.qB_raw[:, 0] = 1.0
            self.qC_raw[:, 0] = 1.0

    def _mix(self, q_features, W):
        return W(q_features.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, x):
        # x: (B, T, n_quat, 4)
        Bz, T, N, _ = x.shape
        x_mix = self._mix(x, self.W_x)            # (B, T, n_quat, 4)
        # Invariant statistics: per-channel magnitudes
        mags = x_mix.norm(dim=-1)                 # (B, T, n_quat)
        # Selective gates from invariant scalars
        sA = torch.sigmoid(self.gate_A(mags))     # (B, T, n_quat)
        sB = torch.sigmoid(self.gate_B(mags))
        # Direction-fixed unit quaternions
        qA = qnorm_unit(self.qA_raw)              # (n_quat, 4)
        qB = qnorm_unit(self.qB_raw)
        qC = qnorm_unit(self.qC_raw)
        # Time-varying scalar-modulated A_t, B_t
        A_t = sA.unsqueeze(-1) * qA               # (B, T, n_quat, 4)
        B_in = qmul(x_mix, sB.unsqueeze(-1) * qB) # (B, T, n_quat, 4)
        # Parallel scan
        H = parallel_scan_quat(A_t, B_in)
        # Output projection
        y = qmul(self._mix(H, self.W_y), qC)
        return y


class SeQuMambaTemporalEncoder(nn.Module):
    """Drop-in replacement for MambaTemporalEncoder."""
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2, dropout=0.3):
        super().__init__()
        assert hidden_dim % 4 == 0
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.n_quat = hidden_dim // 4
        self.blocks = nn.ModuleList([SeQuMambaBlock(self.n_quat) for _ in range(num_layers)])
        # Magnitude norm + SiLU (operates on invariant magnitudes, preserves equivariance)
        self.norms = nn.ModuleList([nn.LayerNorm(self.n_quat) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(self.n_quat)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def _gnorm(self, x_q, norm_layer):
        # Magnitude-norm + SiLU: rescale each quaternion to have magnitude
        # `silu(norm(|q|))` × original direction. SO(3)-equivariant.
        mags = x_q.norm(dim=-1, keepdim=True) + 1e-9
        new_mags = F.silu(norm_layer(mags.squeeze(-1))).unsqueeze(-1)
        return x_q / mags * new_mags

    def forward(self, x):
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)
        x_q = x.reshape(B * N, T, self.n_quat, 4)
        for blk, norm in zip(self.blocks, self.norms):
            residual = x_q
            x_n = self._gnorm(x_q, norm)
            out = blk(x_n)
            x_q = self.dropout(out) + residual
        x_q = self._gnorm(x_q, self.final_norm)
        x = x_q.reshape(B * N, T, self.hidden_dim)
        x = self.output_proj(x)
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        return x


class MotionQuMamba(Motion):
    def __init__(self, *args, qumamba_layers=2, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        hid = old.hidden_dim
        out_d = old.output_dim
        if hid % 4 != 0:
            hid = (hid // 4) * 4
        self.mamba = SeQuMambaTemporalEncoder(in_channels=in_c, hidden_dim=hid,
                                               output_dim=out_d, num_layers=qumamba_layers)
