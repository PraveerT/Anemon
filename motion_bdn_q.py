"""BDN with quaternion-shape substrate (BDN-Q).

Combines:
  - BDN hybrid:   short attention KV-buffer (FIFO, capacity W) + long-term delta
                  state. Buffer's oldest entry rolls into delta state on overflow.
  - RD substrate: 4-fold "quaternion" component split with elementwise mixing
                  (i.e. RD's real ablation -- no Hamilton products), n_q queries
                  and n_v values per head, all carrying a trailing 4-dim that is
                  combined elementwise.

Motivation: BDN (real, non-quaternion) underperforms RD on NVGesture even when
its hybrid is fully active (buf=1 -> 2 ejections/call at T=3). Closing the gap
likely requires the same 4-fold substrate RD/AttRD use.

Drop-in replacement for MotionBDelta -- same yaml plumbing, only the temporal
encoder swaps.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


def emul(p, q):
    """Elementwise product over last (4-component) dim. RD-real semantics."""
    return p * q


class BDeltaQBlock(nn.Module):
    """Buffered Delta block on RD's n_q/n_v/4 substrate.

    Shapes:
      q,k: (B, T, H, n_q, 4)
      v  : (B, T, H, n_v, 4)
      S  : (B, H,    n_v, n_q, 4)   -- long-term delta state, per 4-channel
      buf entries: each (kt, vt) with kt:(B,H,n_q,4), vt:(B,H,n_v,4)

    Read = buffer-attn over current buffer + delta-read from S.
    """
    def __init__(self, d_model, num_heads=4, n_q=4, n_v=8, buffer_size=4,
                 dropout=0.1, use_short_conv=True, conv_size=4,
                 max_seq_len=512, rope_base=10000.0):
        super().__init__()
        self.H = num_heads
        self.n_q = n_q
        self.n_v = n_v
        self.W = buffer_size

        d_qk = num_heads * n_q * 4
        d_v  = num_heads * n_v * 4
        self.q_proj    = nn.Linear(d_model, d_qk, bias=False)
        self.k_proj    = nn.Linear(d_model, d_qk, bias=False)
        self.v_proj    = nn.Linear(d_model, d_v,  bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads * n_q)

        self.use_short_conv = use_short_conv
        if use_short_conv:
            ch = 2 * d_qk + d_v
            self.short_conv = nn.Conv1d(ch, ch, kernel_size=conv_size,
                                        padding=conv_size - 1, groups=ch)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(d_v, d_model, bias=False)

        # RoPE over the flat (n_q*4) channel of q/k for buffer attention.
        rope_dim = n_q * 4
        assert rope_dim % 2 == 0, "RoPE needs even rope_dim"
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        pos = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', pos, inv_freq)        # (T_max, rope_dim/2)
        self.register_buffer('rope_cos', freqs.cos(), persistent=False)
        self.register_buffer('rope_sin', freqs.sin(), persistent=False)
        self._rope_dim = rope_dim

    def _rope_flat(self, x_flat, pos):
        """Apply RoPE on (..., rope_dim) at absolute position `pos`."""
        cos = self.rope_cos[pos]
        sin = self.rope_sin[pos]
        x1, x2 = x_flat[..., 0::2], x_flat[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.stack([rot1, rot2], dim=-1).flatten(-2)

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
        # Normalize key along the (n_q*4) flat axis -- mirrors BDN's F.normalize on D.
        k_flat = k.reshape(B, T, H, nq * 4)
        k_flat = F.normalize(k_flat, dim=-1)
        k = k_flat.view(B, T, H, nq, 4)
        q = F.silu(q)

        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, nq)   # per-component beta

        # Long-term delta state S: (B, H, n_v, n_q, 4)
        S = torch.zeros(B, H, nv, nq, 4, device=x.device, dtype=x.dtype)

        # Buffer holds rotated-key tensors (B,H,nq,4) and values (B,H,nv,4).
        K_buf, V_buf, P_buf = [], [], []
        ys = []

        for t in range(T):
            kt = k[:, t]; vt = v[:, t]                              # (B,H,nq,4), (B,H,nv,4)
            beta_t = beta[:, t]                                      # (B,H,nq)

            # Rotate key for buffer attention (in flat-axis RoPE space).
            kt_flat = kt.reshape(B, H, nq * 4)
            kt_rot_flat = self._rope_flat(kt_flat, t)
            kt_rot = kt_rot_flat.view(B, H, nq, 4)

            # Eject oldest into delta state on overflow.
            if len(K_buf) >= W:
                kt_old = K_buf.pop(0); vt_old = V_buf.pop(0); P_buf.pop(0)
                # Sk_old[b,h,nv,4] = sum_{nq} S[b,h,nv,nq,4] * k_old[b,h,nq,4]  (elementwise on 4)
                Sk_old = (S * kt_old.unsqueeze(2)).sum(dim=3)        # (B,H,nv,4)
                err = vt_old - Sk_old                                # (B,H,nv,4)
                # outer update on (nv,nq), elementwise on 4:
                # upd[b,h,nv,nq,c] = err[b,h,nv,c] * k_old[b,h,nq,c]
                upd = err.unsqueeze(3) * kt_old.unsqueeze(2)         # (B,H,nv,nq,4)
                beta_bc = beta_t.view(B, H, 1, nq, 1)                # broadcast over nv and 4
                S = S + beta_bc * upd

            K_buf.append(kt_rot); V_buf.append(vt); P_buf.append(t)

            # Buffer attention: flatten (nq,4) for scoring, then read v in (nv,4) shape.
            q_flat = q[:, t].reshape(B, H, nq * 4)                   # (B,H,nq*4)
            q_rot_flat = self._rope_flat(q_flat, t)                  # (B,H,nq*4)
            K_stack_flat = torch.stack([kb.reshape(B, H, nq * 4) for kb in K_buf], dim=2)  # (B,H,L,nq*4)
            V_stack = torch.stack(V_buf, dim=2)                      # (B,H,L,nv,4)
            scores = torch.einsum('bhd,bhld->bhl', q_rot_flat, K_stack_flat) / math.sqrt(nq * 4)
            attn = self.attn_dropout(F.softmax(scores, dim=-1))      # (B,H,L)
            buf_out = torch.einsum('bhl,bhlve->bhve', attn, V_stack) # (B,H,nv,4)

            # Delta read with un-rotated q: y_delta[b,h,nv,4] = sum_{nq} S[b,h,nv,nq,4] * q_t[b,h,nq,4]
            delta_out = (S * q[:, t].unsqueeze(2)).sum(dim=3)        # (B,H,nv,4)

            yt = buf_out + delta_out                                  # (B,H,nv,4)
            ys.append(yt)

        y = torch.stack(ys, dim=1).reshape(B, T, H * nv * 4)
        return self.o_proj(self.dropout(y))


class BDeltaQTemporalEncoder(nn.Module):
    """Bidirectional BDN-Q encoder. Mirrors RealDeltaNetTemporalEncoder.

    scan_axis: 'T' (default) scans length-T per-point trajectories;
               'N' scans length-N per-frame point traversals. With T<<N,
               'N' gives buf=1 ejections=N-1 (vs T-1 for 'T'), exercising
               the long-term delta state.
    """
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
            # per-point trajectories: each of N points gets a length-T sequence.
            x = x.permute(0, 3, 2, 1).reshape(Bz * N, T, C)            # (Bz*N, T, C)
        else:
            # per-frame point traversals: each of T frames gets a length-N sequence.
            x = x.permute(0, 2, 3, 1).reshape(Bz * T, N, C)            # (Bz*T, N, C)
        x = self.input_proj(x)
        fwd = self._stack(x, self.fwd_blocks, self.fwd_norms)
        out = fwd
        if self.bidirectional:
            bwd = self._stack(x.flip(1), self.bwd_blocks, self.bwd_norms).flip(1)
            out = out + bwd
        out = self.final_norm(out)
        out = self.output_proj(out)
        if self.scan_axis == 'T':
            out = out.reshape(Bz, N, T, self.output_dim).permute(0, 3, 2, 1)   # (Bz, hidden, T, N)
        else:
            out = out.reshape(Bz, T, N, self.output_dim).permute(0, 3, 1, 2)   # (Bz, hidden, T, N)
        return out


class MotionBDeltaQ(Motion):
    """PMamba with BDN-Q (buffered delta on quaternion-shape substrate)."""
    def __init__(self, *args, bdnq_hidden_dim=128, bdnq_num_layers=2, bdnq_num_heads=4,
                 bdnq_n_q=4, bdnq_n_v=8, bdnq_buffer_size=1, bdnq_dropout=0.3,
                 bdnq_bidirectional=True, bdnq_scan_axis='T', **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = BDeltaQTemporalEncoder(
            in_channels=in_c, hidden_dim=bdnq_hidden_dim, output_dim=out_d,
            num_layers=bdnq_num_layers, num_heads=bdnq_num_heads,
            n_q=bdnq_n_q, n_v=bdnq_n_v, buffer_size=bdnq_buffer_size,
            dropout=bdnq_dropout, bidirectional=bdnq_bidirectional,
            scan_axis=bdnq_scan_axis,
        )
