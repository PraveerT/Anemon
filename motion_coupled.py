"""Coupled Mamba: depth (N2 backbone, per-frame) + lattice (SeQuMamba, per-frame)
with state-level temporal coupling between streams.

Per-step recurrence (each stream has its own state h, but cross-influences):
    gate_d = sigmoid(W_gd x_d[t])
    gate_l = sigmoid(W_gl x_l[t])
    state_d = (1 - gate_d) * state_d + gate_d * (B_d x_d[t] + C_dl @ state_l)
    state_l = (1 - gate_l) * state_l + gate_l * (B_l x_l[t] + C_ld @ state_d)
    out_d[t] = O_d state_d
    out_l[t] = O_l state_l

Inspired by Coupled Mamba (OpenReview UXEo3uNNIX): inter-modal hidden-state
transition aggregates current states from all other streams. Tighter than
concat or cross-attention because coupling happens INSIDE the recurrence.

Streams:
- depth: N2 backbone with per-frame extract (modify pool5 to keep T axis)
- lattice: K=8 top-256 active arrows, SeQuMamba processed per-frame

Joint loss: CE on (depth_logits + lattice_logits) primary + aux CE on each.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.motion_qumamba import SeQuMambaBlock


def per_frame_extract(motion, depth_pts):
    """Run motion backbone but pool only over points (keep T axis).
    Returns (B, T, 1024) per-frame depth features.
    """
    coords = motion._sample_points(depth_pts)               # (B, C, T, P_sub)
    fea3 = motion._encode_sampled_points(coords)            # (B, C', T, P_smaller)
    output = motion.stage5(fea3)                            # (B, 1024, T, P)
    # Pool only over points (last dim) to keep T
    output = output.amax(dim=-1, keepdim=True)              # (B, 1024, T, 1)
    output = motion.global_bn(output)                       # (B, 1024, T, 1)
    output = output.squeeze(-1).transpose(1, 2)             # (B, T, 1024)
    return output


class LatticePerFrameEncoder(nn.Module):
    """Lattice arrow field -> per-frame features (B, T, D_l) via SeQuMamba (no time mean)."""
    def __init__(self, n_lattice=256, hidden=192, n_layers=4, dropout=0.2):
        super().__init__()
        assert hidden % 4 == 0
        self.n_quat = hidden // 4
        self.lift = nn.Linear(n_lattice, self.n_quat, bias=False)
        self.blocks = nn.ModuleList([SeQuMambaBlock(self.n_quat) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.n_quat) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(self.n_quat)
        self.feature_dim = self.n_quat  # invariant magnitudes per frame

    def _gnorm(self, x_q, n):
        m = x_q.norm(dim=-1, keepdim=True) + 1e-9
        new_m = F.silu(n(m.squeeze(-1))).unsqueeze(-1)
        return x_q / m * new_m

    def forward(self, x):
        # x: (B, T, n_lattice, 4)
        x = self.lift(x.transpose(-1, -2)).transpose(-1, -2)  # (B, T, n_quat, 4)
        for blk, norm in zip(self.blocks, self.norms):
            r = x; x = blk(self._gnorm(x, norm))
            x = self.dropout(x) + r
        x = self._gnorm(x, self.final_norm)
        # Per-frame magnitudes: (B, T, n_quat) — keeps T axis
        return x.norm(dim=-1)


class CoupledMambaBlock(nn.Module):
    """Coupled gated linear SSM over T. Two streams cross-update at each step.

    state_d, state_l: (B, H_d), (B, H_l)
    Inputs x_d, x_l: (B, T, D_d), (B, T, D_l)
    Outputs y_d, y_l: (B, T, D_d), (B, T, D_l)
    """
    def __init__(self, d_dim, l_dim, hidden_d=128, hidden_l=128):
        super().__init__()
        self.d_dim = d_dim
        self.l_dim = l_dim
        self.hidden_d = hidden_d
        self.hidden_l = hidden_l
        # Input projections (x -> state-space)
        self.B_d = nn.Linear(d_dim, hidden_d, bias=True)
        self.B_l = nn.Linear(l_dim, hidden_l, bias=True)
        # Cross-coupling projections (state_other -> state_self)
        self.C_dl = nn.Linear(hidden_l, hidden_d, bias=False)
        self.C_ld = nn.Linear(hidden_d, hidden_l, bias=False)
        # Selectivity gates (input-dependent)
        self.gate_d = nn.Linear(d_dim, hidden_d)
        self.gate_l = nn.Linear(l_dim, hidden_l)
        # Output projections (state -> stream output)
        self.O_d = nn.Linear(hidden_d, d_dim)
        self.O_l = nn.Linear(hidden_l, l_dim)
        # Init cross-coupling small so streams start nearly independent
        nn.init.zeros_(self.C_dl.weight)
        nn.init.zeros_(self.C_ld.weight)

    def forward(self, x_d, x_l):
        B, T, _ = x_d.shape
        h_d = torch.zeros(B, self.hidden_d, device=x_d.device, dtype=x_d.dtype)
        h_l = torch.zeros(B, self.hidden_l, device=x_l.device, dtype=x_l.dtype)
        out_d = []
        out_l = []
        # Use t-1 cross-state to avoid coupling within same step (causal)
        h_d_prev = h_d
        h_l_prev = h_l
        for t in range(T):
            xd_t = x_d[:, t]
            xl_t = x_l[:, t]
            g_d = torch.sigmoid(self.gate_d(xd_t))
            g_l = torch.sigmoid(self.gate_l(xl_t))
            inp_d = self.B_d(xd_t) + self.C_dl(h_l_prev)
            inp_l = self.B_l(xl_t) + self.C_ld(h_d_prev)
            h_d = (1 - g_d) * h_d_prev + g_d * inp_d
            h_l = (1 - g_l) * h_l_prev + g_l * inp_l
            out_d.append(self.O_d(h_d))
            out_l.append(self.O_l(h_l))
            h_d_prev = h_d
            h_l_prev = h_l
        return torch.stack(out_d, dim=1), torch.stack(out_l, dim=1)


class MotionCoupled(nn.Module):
    """N2 (depth) + Lattice (side) with state-coupled temporal Mamba."""
    def __init__(self, num_classes=25, pts_size=96, knn=(32, 24, 48, 24), topk=8,
                 multi_scale_num_scales=5,
                 lattice_n=256, lattice_hidden=192, lattice_layers=4,
                 coupled_hidden=256,
                 lattice_init_weights=None,
                 aux_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.pts_size = pts_size
        self.lattice_n = lattice_n
        self.aux_weight = aux_weight

        # Depth backbone (full N2)
        self.main = Motion(num_classes=num_classes, pts_size=pts_size,
                           knn=list(knn), topk=topk,
                           multi_scale_num_scales=multi_scale_num_scales)
        d_dim = 1024  # N2 stage5 output channels

        # Lattice per-frame encoder
        self.side = LatticePerFrameEncoder(n_lattice=lattice_n,
                                           hidden=lattice_hidden,
                                           n_layers=lattice_layers)
        l_dim = self.side.feature_dim  # = lattice_hidden // 4

        # Coupled temporal block
        self.coupled = CoupledMambaBlock(d_dim=d_dim, l_dim=l_dim,
                                         hidden_d=coupled_hidden,
                                         hidden_l=coupled_hidden)

        # Per-stream classifiers (operate on coupled outputs, time-mean pooled)
        self.cls_d = nn.Linear(d_dim, num_classes)
        self.cls_l = nn.Linear(l_dim, num_classes)

        # Optional warm-start of lattice from standalone training
        if lattice_init_weights is not None:
            self._load_lattice_warmstart(lattice_init_weights)

    def _load_lattice_warmstart(self, ckpt_path):
        import os
        if not os.path.isfile(ckpt_path):
            print(f"[MotionCoupled] WARN: warm-start ckpt not found: {ckpt_path}")
            return
        ck = torch.load(ckpt_path, map_location='cpu')
        sd = ck.get('model_state_dict', ck)
        # Standalone keys: lift.weight, blocks.X.*, norms.X.*, final_norm.*, head.X.*
        side_sd = {k: v for k, v in sd.items() if not k.startswith('head.')}
        m, u = self.side.load_state_dict(side_sd, strict=False)
        print(f"[MotionCoupled] warm-start side: missing={len(m)} unexpected={len(u)}")

    def _split_input(self, x):
        if x.dim() == 4 and x.shape[-1] == 8:
            B, T, P_total, C = x.shape
        elif x.dim() == 4 and x.shape[1] == 8:
            x = x.permute(0, 2, 3, 1).contiguous()
            B, T, P_total, C = x.shape
        else:
            raise ValueError(f"Unexpected input shape {tuple(x.shape)}")
        P_d = P_total - self.lattice_n
        depth_pts = x[:, :, :P_d, :4].contiguous()
        lattice_q = x[:, :, P_d:, 4:8].contiguous()
        return depth_pts, lattice_q

    def forward(self, x):
        depth_pts, lattice_q = self._split_input(x)
        # Per-frame features
        x_d = per_frame_extract(self.main, depth_pts)            # (B, T, 1024)
        x_l = self.side(lattice_q)                                # (B, T, l_dim)
        # Coupled temporal
        y_d, y_l = self.coupled(x_d, x_l)                         # both (B, T, *)
        # Time-mean pool then classify each
        d_logits = self.cls_d(y_d.mean(dim=1))
        l_logits = self.cls_l(y_l.mean(dim=1))
        # Expose for framework aux loss
        self.temporal_logits = d_logits
        self.spatial_logits = l_logits
        return d_logits + l_logits
