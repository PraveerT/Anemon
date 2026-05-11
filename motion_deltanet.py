"""N2 with Gated DeltaNet (ICLR 2025, arXiv 2412.06464) replacing the temporal Mamba.

Self-contained implementation: T=32 is short, so a Python loop over time on GPU
is fast enough. No need for fla's Triton kernels.

DeltaNet recurrence (per head):
    S_t = alpha_t * (I - beta_t * k_t k_t^T) * S_{t-1} + beta_t * k_t v_t^T
    y_t = q_t^T S_t
where:
    S_t : (head_dim, value_dim) state matrix per head
    q_t, k_t : (head_dim,) query/key per head
    v_t : (value_dim,) value per head
    alpha_t : scalar forget gate per head per step (sigmoid)
    beta_t : scalar delta-rule write strength per head per step (sigmoid)

True state tracking (associative recall, parity) — strictly more expressive
than scalar SSM (Mamba). Drop QuaternionLinear (no equivariance value here).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class GatedDeltaNetBlock(nn.Module):
    """Single Gated DeltaNet block: takes (B, T, D) -> (B, T, D)."""
    def __init__(self, d_model, num_heads=4, head_dim=32, expand_v=2,
                 use_short_conv=True, conv_size=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = head_dim * int(expand_v)
        H, dk, dv = num_heads, head_dim, self.value_dim

        # QKV projections
        self.q_proj = nn.Linear(d_model, H * dk, bias=False)
        self.k_proj = nn.Linear(d_model, H * dk, bias=False)
        self.v_proj = nn.Linear(d_model, H * dv, bias=False)

        # Per-head per-step gates
        self.alpha_proj = nn.Linear(d_model, H)  # forget
        self.beta_proj = nn.Linear(d_model, H)   # write strength

        # Mamba-style short conv on Q, K, V (depthwise causal)
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        if use_short_conv:
            ch = H * dk * 2 + H * dv
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(H * dv, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        H, dk, dv = self.num_heads, self.head_dim, self.value_dim

        q = self.q_proj(x)              # (B, T, H*dk)
        k = self.k_proj(x)
        v = self.v_proj(x)              # (B, T, H*dv)
        if self.use_short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)  # (B, ch, T)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2) # causal: drop right pad
            q, k, v = qkv[..., :H*dk], qkv[..., H*dk:2*H*dk], qkv[..., 2*H*dk:]

        # Reshape to per-head: (B, H, T, dk_or_dv)
        q = q.view(B, T, H, dk).transpose(1, 2)
        k = k.view(B, T, H, dk).transpose(1, 2)
        v = v.view(B, T, H, dv).transpose(1, 2)
        # L2-normalize keys for stable Householder
        k = F.normalize(k, dim=-1)
        # Activation on q (silu, common in DeltaNet)
        q = F.silu(q)

        # Gates per head per step
        alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2)  # (B, H, T)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)

        # State (B, H, dk, dv); recurrent loop over T (T=32, fast)
        S = torch.zeros(B, H, dk, dv, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            qt = q[:, :, t]                # (B, H, dk)
            kt = k[:, :, t]
            vt = v[:, :, t]                # (B, H, dv)
            at = alpha[:, :, t].view(B, H, 1, 1)
            bt = beta[:, :, t].view(B, H, 1, 1)

            # k^T S: contract over dk. S[..., i, j] is dk x dv.
            # kt.unsqueeze(-1): (B, H, dk, 1); times S sum dim -2 -> (B, H, dv)
            kT_S = (kt.unsqueeze(-1) * S).sum(dim=-2)
            # Update: S = at * (I - bt k k^T) S + bt k v^T
            S = at * (S - bt * kt.unsqueeze(-1) * kT_S.unsqueeze(-2)) \
                + bt * (kt.unsqueeze(-1) * vt.unsqueeze(-2))
            # Output: q^T S -> (B, H, dv)
            out = (qt.unsqueeze(-1) * S).sum(dim=-2)
            outs.append(out)

        y = torch.stack(outs, dim=2).reshape(B, T, H * dv)
        y = self.dropout(y)
        return self.o_proj(y)


class GatedDeltaNetTemporalEncoder(nn.Module):
    """Drop-in replacement for MambaTemporalEncoder using Gated DeltaNet blocks."""
    def __init__(self, in_channels, hidden_dim=256, output_dim=None, num_layers=2,
                 num_heads=4, head_dim=64, expand_v=2,
                 use_short_conv=True, conv_size=4, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList([
            GatedDeltaNetBlock(hidden_dim, num_heads=num_heads, head_dim=head_dim,
                               expand_v=expand_v, use_short_conv=use_short_conv,
                               conv_size=conv_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        # x: (B, C, T, N)
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)
        for blk, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x) + residual
        x = self.final_norm(x)
        x = self.output_proj(x)
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        return x


class MotionDeltaNet(Motion):
    """N2 backbone with Gated DeltaNet replacing the temporal Mamba."""
    def __init__(self, *args, dn_hidden_dim=256, dn_num_layers=2, dn_num_heads=4,
                 dn_head_dim=64, dn_expand_v=2, dn_dropout=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = GatedDeltaNetTemporalEncoder(
            in_channels=in_c,
            hidden_dim=dn_hidden_dim,
            output_dim=out_d,
            num_layers=dn_num_layers,
            num_heads=dn_num_heads,
            head_dim=dn_head_dim,
            expand_v=dn_expand_v,
            dropout=dn_dropout,
        )
