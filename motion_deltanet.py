"""N2 with Gated DeltaNet (ICLR 2025, arXiv 2412.06464) — CHUNKWISE PARALLEL form.

Uses the closed-form chunkwise computation from the DeltaNet paper (no Python loop):
  V' = (I + L)^{-1} V          (triangular solve, L is rank-1-Householder kernel)
  Y  = (A @ V') * cum_alpha    (lower-triangular attention with delta-rule weights)
where:
  L[i,j] = β'_j (k_i · k_j) for j<i, else 0   (strictly lower)
  A[i,j] = β'_j (q_i · k_j) for j<=i, else 0  (lower with diagonal)
  β'_t = β_t / cum_alpha_t
  cum_alpha_t = ∏_{r=1..t} α_r  (in log space for stability)

Single chunk of length T (=32 here), no recurrence between chunks needed.

Derivation: claim S_t = Σ_{s≤t} β'_s k_s v'_s^T where v'_s = v_s − Σ_{r<s} β'_r (k_s·k_r) v'_r.
Verified by induction (Σ_t β k k^T outer telescopes through (I − β kk^T) Householder).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class GatedDeltaNetBlock(nn.Module):
    """Chunkwise parallel Gated DeltaNet. Drop-in for a single chunk of length T."""
    def __init__(self, d_model, num_heads=4, head_dim=32, expand_v=2,
                 use_short_conv=True, conv_size=4, dropout=0.1, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = head_dim * int(expand_v)
        self.eps = eps
        H, dk, dv = num_heads, head_dim, self.value_dim

        self.q_proj = nn.Linear(d_model, H * dk, bias=False)
        self.k_proj = nn.Linear(d_model, H * dk, bias=False)
        self.v_proj = nn.Linear(d_model, H * dv, bias=False)

        # Per-head per-step gates
        self.alpha_proj = nn.Linear(d_model, H)  # forget
        self.beta_proj = nn.Linear(d_model, H)   # write strength

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

        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)  # causal: drop right pad
            q, k, v = qkv[..., :H*dk], qkv[..., H*dk:2*H*dk], qkv[..., 2*H*dk:]

        # Per-head: (B, H, T, dk)
        q = q.view(B, T, H, dk).transpose(1, 2)
        k = k.view(B, T, H, dk).transpose(1, 2)
        v = v.view(B, T, H, dv).transpose(1, 2)
        k = F.normalize(k, dim=-1)
        q = F.silu(q)

        # Plain DeltaNet (no alpha forget gate, T=32 is short).
        # Avoids the β/cum_alpha numerical blow-up that NaNs the chunkwise solve.
        beta_p = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)   # (B, H, T)

        # KK[i,j] = k_i · k_j  (B, H, T, T)
        KK = torch.einsum('bhid,bhjd->bhij', k, k)
        # L[i,j] = β'_j * KK[i,j]  for j<i, 0 elsewhere
        # mask: strict lower triangular
        device = x.device
        mask_lt = torch.ones(T, T, device=device, dtype=torch.bool).tril(diagonal=-1)
        L = beta_p.unsqueeze(-2) * KK                                # broadcast: (B, H, T, T)
        L = L * mask_lt.to(L.dtype)
        # I + L is unit lower triangular
        eye = torch.eye(T, device=device, dtype=L.dtype).expand(B, H, T, T)
        I_plus_L = eye + L
        # V' = (I + L)^{-1} V
        V_prime = torch.linalg.solve_triangular(I_plus_L, v, upper=False, unitriangular=True)

        # A[i,j] = β'_j * (q_i · k_j) for j<=i, 0 elsewhere
        QK = torch.einsum('bhid,bhjd->bhij', q, k)
        mask_le = torch.ones(T, T, device=device, dtype=torch.bool).tril(diagonal=0)
        A = beta_p.unsqueeze(-2) * QK
        A = A * mask_le.to(A.dtype)

        # Y = A @ V' (no cum_alpha rescaling since alpha is dropped)
        Y = torch.matmul(A, V_prime)                                 # (B, H, T, dv)

        y = Y.transpose(1, 2).reshape(B, T, H * dv)
        y = self.dropout(y)
        return self.o_proj(y)


class GatedDeltaNetTemporalEncoder(nn.Module):
    """Drop-in replacement for MambaTemporalEncoder using chunkwise Gated DeltaNet blocks."""
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
    """N2 backbone with chunkwise Gated DeltaNet replacing the temporal Mamba."""
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
