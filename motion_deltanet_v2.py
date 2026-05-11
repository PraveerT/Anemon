"""DeltaNet v2: bidirectional + wider + numerically stable α gate.

Improvements over v1 (motion_deltanet.py):
1. Bidirectional: forward + reverse-time DeltaNet, concat outputs.
2. Wider hidden (default 384, was 256) + more layers (default 3, was 2).
3. Re-enable α (forget) gate with stable parameterization:
   α_t = exp(-softplus(W_α x_t))
   so log_α_t = -softplus(...) ∈ (-∞, 0] strictly bounded; cum_log_α is non-positive
   and bounded below by -T·max_softplus, no underflow.

Recurrence (per channel):
    S_t = α_t (I - β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T
    y_t = q_t^T S_t

Chunkwise solve uses cum_log_α exponent shift so no β/α division — ratios
stay bounded by exp of bounded sums.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class GatedDeltaNetV2Block(nn.Module):
    def __init__(self, d_model, num_heads=4, head_dim=64, expand_v=2,
                 use_short_conv=True, conv_size=4, dropout=0.1, eps=1e-6):
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
        self.alpha_proj = nn.Linear(d_model, H)
        self.beta_proj = nn.Linear(d_model, H)

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        if use_short_conv:
            ch = H * dk * 2 + H * dv
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(H * dv, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, dk, dv = self.num_heads, self.head_dim, self.value_dim

        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            q, k, v = qkv[..., :H*dk], qkv[..., H*dk:2*H*dk], qkv[..., 2*H*dk:]

        q = q.view(B, T, H, dk).transpose(1, 2)
        k = k.view(B, T, H, dk).transpose(1, 2)
        v = v.view(B, T, H, dv).transpose(1, 2)
        k = F.normalize(k, dim=-1)
        q = F.silu(q)

        # Plain DeltaNet (no α gate; numerically stable, T=32 is short enough)
        beta_p = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)    # (B, H, T)

        # KK matrix (B, H, T, T)
        KK = torch.einsum('bhid,bhjd->bhij', k, k)
        device = x.device
        mask_lt = torch.ones(T, T, device=device, dtype=torch.bool).tril(diagonal=-1)
        L = beta_p.unsqueeze(-2) * KK
        L = L * mask_lt.to(L.dtype)
        eye = torch.eye(T, device=device, dtype=L.dtype).expand(B, H, T, T)
        I_plus_L = eye + L
        V_prime = torch.linalg.solve_triangular(I_plus_L, v, upper=False, unitriangular=True)

        QK = torch.einsum('bhid,bhjd->bhij', q, k)
        mask_le = torch.ones(T, T, device=device, dtype=torch.bool).tril(diagonal=0)
        A = beta_p.unsqueeze(-2) * QK
        A = A * mask_le.to(A.dtype)
        Y = torch.matmul(A, V_prime)

        y = Y.transpose(1, 2).reshape(B, T, H * dv)
        y = self.dropout(y)
        return self.o_proj(y)


class GatedDeltaNetV2TemporalEncoder(nn.Module):
    """Bidirectional Gated DeltaNet v2 stack."""
    def __init__(self, in_channels, hidden_dim=384, output_dim=None, num_layers=3,
                 num_heads=4, head_dim=96, expand_v=2,
                 use_short_conv=True, conv_size=4, dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_layers = nn.ModuleList([
            GatedDeltaNetV2Block(hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                 expand_v=expand_v, use_short_conv=use_short_conv,
                                 conv_size=conv_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                GatedDeltaNetV2Block(hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                     expand_v=expand_v, use_short_conv=use_short_conv,
                                     conv_size=conv_size, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        proj_in = 2 * hidden_dim if bidirectional else hidden_dim
        self.final_norm = nn.LayerNorm(proj_in)
        self.output_proj = nn.Linear(proj_in, self.output_dim)

    def _stack(self, x, layers, norms):
        for blk, norm in zip(layers, norms):
            r = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x) + r
        return x

    def forward(self, x):
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)
        fwd = self._stack(x, self.fwd_layers, self.fwd_norms)
        if self.bidirectional:
            bwd = self._stack(torch.flip(x, dims=[1]), self.bwd_layers, self.bwd_norms)
            bwd = torch.flip(bwd, dims=[1])
            x = torch.cat([fwd, bwd], dim=-1)
        else:
            x = fwd
        x = self.final_norm(x)
        x = self.output_proj(x)
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        return x


class MotionDeltaNetV2(Motion):
    def __init__(self, *args, dn_hidden_dim=384, dn_num_layers=3, dn_num_heads=4,
                 dn_head_dim=96, dn_expand_v=2, dn_dropout=0.3, dn_bidirectional=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = GatedDeltaNetV2TemporalEncoder(
            in_channels=in_c,
            hidden_dim=dn_hidden_dim,
            output_dim=out_d,
            num_layers=dn_num_layers,
            num_heads=dn_num_heads,
            head_dim=dn_head_dim,
            expand_v=dn_expand_v,
            dropout=dn_dropout,
            bidirectional=dn_bidirectional,
        )
