"""BDN-Q vectorized (BDeltaQBlock with parallel windowed buffer-attention).

Exact reformulation of the naive per-timestep BDN-Q scan. Output at step t is
  y_t = buffer_attn_t + delta_read_t
where (verified equivalent to the naive loop):
  * buffer_attn_t : softmax attention of the rotated query q_rot[t] over the
    rotated keys k_rot[t-W+1 .. t] and values v[t-W+1 .. t]  (a length-W causal
    sliding window). This is computed for ALL t at once via a banded mask --
    the part that was the bulk of the Python loop's work.
  * delta_read_t  : q_silu[t] read from the DeltaNet state S that has absorbed
    every key EVICTED from the window so far, i.e. keys[0 .. t-W]. The eviction
    of key m happens at step m+W and uses beta[m+W] (the naive code's current-
    step beta), updating S with the ROTATED key k_rot[m]. Only this part stays
    a (short) sequential scan -- the genuine recurrence.

Same module/parameter layout as models.motion_bdn_q so naive weights load 1:1
(used by the equivalence test). Drop-in: BDeltaQTemporalEncoder is unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def emul(p, q):
    return p * q


class BDeltaQBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, n_q=4, n_v=8, buffer_size=4,
                 dropout=0.1, use_short_conv=True, conv_size=4,
                 max_seq_len=512, rope_base=10000.0):
        super().__init__()
        self.H = num_heads
        self.n_q = n_q
        self.n_v = n_v
        self.W = buffer_size

        d_qk = num_heads * n_q * 4
        d_v = num_heads * n_v * 4
        self.q_proj = nn.Linear(d_model, d_qk, bias=False)
        self.k_proj = nn.Linear(d_model, d_qk, bias=False)
        self.v_proj = nn.Linear(d_model, d_v, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads * n_q)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            ch = 2 * d_qk + d_v
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(d_v, d_model, bias=False)

        rope_dim = n_q * 4
        assert rope_dim % 2 == 0, "RoPE needs even rope_dim"
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', pos, inv_freq)
        self.register_buffer('rope_cos', freqs.cos(), persistent=False)
        self.register_buffer('rope_sin', freqs.sin(), persistent=False)
        self._rope_dim = rope_dim

    def _rope_flat(self, x_flat, pos):
        cos = self.rope_cos[pos]
        sin = self.rope_sin[pos]
        x1, x2 = x_flat[..., 0::2], x_flat[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.stack([rot1, rot2], dim=-1).flatten(-2)

    def _rope_all(self, xf):
        # xf: (B, T, H, rope_dim) -> RoPE each position t with cos/sin[t]
        B, T, H, RD = xf.shape
        cos = self.rope_cos[:T].view(1, T, 1, RD // 2)
        sin = self.rope_sin[:T].view(1, T, 1, RD // 2)
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
        return torch.stack([r1, r2], dim=-1).flatten(-2)

    def forward(self, x):
        B, T, D = x.shape
        H, nq, nv, W = self.H, self.n_q, self.n_v, self.W

        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        if self.use_short_conv:
            qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2)
            qkv = self.short_conv(qkv)[..., :T].transpose(1, 2)
            s1 = H * nq * 4
            s2 = s1 + H * nq * 4
            q, k, v = qkv[..., :s1], qkv[..., s1:s2], qkv[..., s2:]

        q = q.view(B, T, H, nq, 4)
        k = k.view(B, T, H, nq, 4)
        v = v.view(B, T, H, nv, 4)
        k = F.normalize(k.reshape(B, T, H, nq * 4), dim=-1).view(B, T, H, nq, 4)
        q = F.silu(q)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, nq)

        q_rot = self._rope_all(q.reshape(B, T, H, nq * 4))           # (B,T,H,nq*4)
        k_rot = self._rope_all(k.reshape(B, T, H, nq * 4))

        # --- windowed buffer attention (all t at once) ---
        scale = math.sqrt(nq * 4)
        scores = torch.einsum('bthd,bshd->bhts', q_rot, k_rot) / scale   # (B,H,T,T)
        idx = torch.arange(T, device=x.device)
        rel = idx.view(T, 1) - idx.view(1, T)                       # t - s
        band = (rel >= 0) & (rel <= W - 1)
        scores = scores.masked_fill(~band.view(1, 1, T, T), float('-inf'))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))         # (B,H,T,T)
        buf_out = torch.einsum('bhts,bshve->bhtve', attn, v)        # (B,H,T,nv,4)

        # --- delta read: sequential scan over evicted keys only ---
        k_rot_q = k_rot.view(B, T, H, nq, 4)
        S = torch.zeros(B, H, nv, nq, 4, device=x.device, dtype=x.dtype)
        delta = torch.zeros(B, H, T, nv, 4, device=x.device, dtype=x.dtype)
        for m in range(T - W):
            km = k_rot_q[:, m]                                       # (B,H,nq,4)
            vm = v[:, m]                                             # (B,H,nv,4)
            bm = beta[:, m + W]                                      # (B,H,nq)
            Skm = (S * km.unsqueeze(2)).sum(dim=3)                   # (B,H,nv,4)
            err = vm - Skm
            S = S + bm.view(B, H, 1, nq, 1) * (err.unsqueeze(3) * km.unsqueeze(2))
            t = m + W
            delta[:, :, t] = (S * q[:, t].unsqueeze(2)).sum(dim=3)   # (B,H,nv,4)

        y = (buf_out + delta).permute(0, 2, 1, 3, 4).reshape(B, T, H * nv * 4)
        return self.o_proj(self.dropout(y))


class BDeltaQTemporalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, output_dim=None, num_layers=2,
                 num_heads=4, n_q=4, n_v=8, buffer_size=4, dropout=0.3,
                 bidirectional=True, scan_axis='T'):
        super().__init__()
        assert scan_axis in ('T', 'N'), f'scan_axis must be T or N, got {scan_axis}'
        self.scan_axis = scan_axis
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_blocks = nn.ModuleList([
            BDeltaQBlock(hidden_dim, num_heads, n_q, n_v, buffer_size, dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_blocks = nn.ModuleList([
                BDeltaQBlock(hidden_dim, num_heads, n_q, n_v, buffer_size, dropout)
                for _ in range(num_layers)
            ])
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def _stack(self, x, layers, norms):
        for blk, norm in zip(layers, norms):
            residual = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x)
            x = x + residual
        return x

    def forward(self, x):
        Bz, C, T, N = x.shape
        if self.scan_axis == 'T':
            x = x.permute(0, 3, 2, 1).reshape(Bz * N, T, C)
        else:
            x = x.permute(0, 2, 3, 1).reshape(Bz * T, N, C)
        x = self.input_proj(x)
        fwd = self._stack(x, self.fwd_blocks, self.fwd_norms)
        out = fwd
        if self.bidirectional:
            bwd = self._stack(x.flip(1), self.bwd_blocks, self.bwd_norms).flip(1)
            out = out + bwd
        out = self.final_norm(out)
        out = self.output_proj(out)
        if self.scan_axis == 'T':
            out = out.reshape(Bz, N, T, self.output_dim).permute(0, 3, 2, 1)
        else:
            out = out.reshape(Bz, T, N, self.output_dim).permute(0, 3, 1, 2)
        return out
