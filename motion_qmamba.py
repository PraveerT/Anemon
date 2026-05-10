"""QMamba: Quaternion-valued Mamba (no equivariance constraint).

Uses quaternion algebra for:
  - Parameter sharing in linear layers (Hamilton-product structure: 4 weight
    matrices W_w, W_x, W_y, W_z mix all 4 input components into all 4 output
    components, total params = 4 * H_in * H_out vs real linear's (4H_in × 4H_out)
    = 16 H_in H_out — quaternion linear is 4× fewer parameters)
  - Quaternion-valued state for richer hidden dynamics (extends Mamba-3's
    complex state to 3D rotation in S^3)

Drops vs SeQuMamba:
  - No equivariance constraint (no channel-tying)
  - No magnitude-only readout (full 4-component output)
  - Vector selective gates (per-component) instead of scalar gates
  - No unit-quaternion constraint on A, B, C parameters

Bridges:
  - Mamba-3 (2026): complex-valued state -> 1D rotation
  - QMamba (this): quaternion-valued state -> 3D rotation, SE(3)-style dynamics
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


def parallel_scan_quat(A, B):
    """Hillis-Steele parallel scan: h_t = h_{t-1} ⊙ A_t + B_t."""
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
        A_new = qmul(A_prev, A_acc)
        B_new = qmul(B_prev, A_acc) + B_acc
        A_acc, B_acc = A_new, B_new
    return B_acc


class QuaternionLinear(nn.Module):
    """Hypercomplex linear layer: maps n_in quaternions to n_out quaternions
    via Hamilton-product structure. 4 weight matrices, 4× fewer parameters
    than equivalent real Linear(4 n_in, 4 n_out).

    For input q = (w, x, y, z) and weights (W_w, W_x, W_y, W_z), output:
      out_w = W_w @ q_w - W_x @ q_x - W_y @ q_y - W_z @ q_z
      out_x = W_w @ q_x + W_x @ q_w + W_y @ q_z - W_z @ q_y
      out_y = W_w @ q_y - W_x @ q_z + W_y @ q_w + W_z @ q_x
      out_z = W_w @ q_z + W_x @ q_y - W_y @ q_x + W_z @ q_w

    This is the Parcollet et al. (2018) quaternion linear formulation.
    """
    def __init__(self, n_in, n_out, bias=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        scale = 1.0 / math.sqrt(n_in)
        self.W_w = nn.Parameter(torch.randn(n_out, n_in) * scale)
        self.W_x = nn.Parameter(torch.randn(n_out, n_in) * scale)
        self.W_y = nn.Parameter(torch.randn(n_out, n_in) * scale)
        self.W_z = nn.Parameter(torch.randn(n_out, n_in) * scale)
        self.b = nn.Parameter(torch.zeros(n_out, 4)) if bias else None

    def forward(self, x):
        # x: (..., n_in, 4)
        qw, qx, qy, qz = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        # Each is (..., n_in); apply weight matrices
        ow = F.linear(qw, self.W_w) - F.linear(qx, self.W_x) - F.linear(qy, self.W_y) - F.linear(qz, self.W_z)
        ox = F.linear(qw, self.W_x) + F.linear(qx, self.W_w) + F.linear(qy, self.W_z) - F.linear(qz, self.W_y)
        oy = F.linear(qw, self.W_y) - F.linear(qx, self.W_z) + F.linear(qy, self.W_w) + F.linear(qz, self.W_x)
        oz = F.linear(qw, self.W_z) + F.linear(qx, self.W_y) - F.linear(qy, self.W_x) + F.linear(qz, self.W_w)
        out = torch.stack([ow, ox, oy, oz], dim=-1)
        if self.b is not None:
            out = out + self.b
        return out


class QMambaBlock(nn.Module):
    """Quaternion-valued Mamba block (no equivariance).

    Recurrence: h_t = h_{t-1} ⊙ A_t + B_t
    Where A_t, B_t are time-varying quaternions derived from input via:
      A_t = unconstrained quaternion (scalar gate × free quaternion direction)
      B_t = quaternion-mul of QuaternionLinear(x_t) with input-dependent weight
    """
    def __init__(self, n_quat, gate_hidden=None):
        super().__init__()
        if gate_hidden is None:
            gate_hidden = max(n_quat // 2, 16)
        # Quaternion linear projections (full hypercomplex, no channel-tying)
        self.Wx = QuaternionLinear(n_quat, n_quat)
        self.Wy = QuaternionLinear(n_quat, n_quat)
        # Selective gates: take FULL quaternion features (not just magnitudes)
        # Apply real linear to flattened input -> per-channel 4-vector gates
        self.gate_A = nn.Sequential(
            nn.Linear(4 * n_quat, gate_hidden), nn.SiLU(),
            nn.Linear(gate_hidden, 4 * n_quat),
        )
        self.gate_B = nn.Sequential(
            nn.Linear(4 * n_quat, gate_hidden), nn.SiLU(),
            nn.Linear(gate_hidden, 4 * n_quat),
        )
        # Free learned quaternions (no unit constraint)
        self.qA = nn.Parameter(torch.randn(n_quat, 4) * 0.05)
        self.qC = nn.Parameter(torch.randn(n_quat, 4) * 0.05)
        with torch.no_grad():
            self.qA[:, 0] = 0.9  # init to slow decay
            self.qC[:, 0] = 1.0

    def forward(self, x):
        # x: (B, T, n_quat, 4)
        Bz, T, N, _ = x.shape
        x_proj = self.Wx(x)                      # (B, T, n_quat, 4)
        # Selective gates from full input features
        x_flat = x_proj.flatten(-2)              # (B, T, n_quat*4)
        sA = torch.sigmoid(self.gate_A(x_flat)).reshape(Bz, T, N, 4)
        sB = torch.sigmoid(self.gate_B(x_flat)).reshape(Bz, T, N, 4)
        # Time-varying A, B: gate-modulated
        A_t = sA * self.qA                       # (B, T, n_quat, 4)
        B_in = sB * x_proj                       # (B, T, n_quat, 4)
        # Parallel scan
        H = parallel_scan_quat(A_t, B_in)        # (B, T, n_quat, 4)
        # Output projection (quaternion linear, no qC constraint needed)
        y = qmul(self.Wy(H), self.qC)
        return y


class QMambaTemporalEncoder(nn.Module):
    """Drop-in replacement for MambaTemporalEncoder."""
    def __init__(self, in_channels, hidden_dim, output_dim=None, num_layers=2, dropout=0.3):
        super().__init__()
        assert hidden_dim % 4 == 0
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.n_quat = hidden_dim // 4
        self.blocks = nn.ModuleList([QMambaBlock(self.n_quat) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)
        # Pack flat into quaternions for processing
        x_q = x.reshape(B * N, T, self.n_quat, 4)
        for blk, norm in zip(self.blocks, self.norms):
            residual = x_q
            x_flat = x_q.flatten(-2)
            x_n = norm(x_flat).reshape(B * N, T, self.n_quat, 4)
            out = blk(x_n)
            x_q = self.dropout(out) + residual
        x = self.final_norm(x_q.flatten(-2))
        x = self.output_proj(x)
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        return x


class MotionQMamba(Motion):
    def __init__(self, *args, qmamba_layers=2, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        hid = old.hidden_dim
        out_d = old.output_dim
        if hid % 4 != 0:
            hid = (hid // 4) * 4
        self.mamba = QMambaTemporalEncoder(in_channels=in_c, hidden_dim=hid,
                                            output_dim=out_d, num_layers=qmamba_layers)
