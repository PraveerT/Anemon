"""N2 with LinOSS (ICLR 2025, arXiv 2410.03943) replacing the temporal Mamba.

LinOSS = Linear Oscillatory State-Space. Each channel is a damped harmonic oscillator
with complex eigenvalue λ = exp(-γ·dt + i·ω·dt). Captures periodic dynamics (gesture
rhythm: waving, pinching). Drop-in for Mamba's recurrence kernel.

Recurrence (per channel, complex):
    z_t = λ · z_{t-1} + B·x_t
    y_t = Re(C·z_t)

Parallel form (chunk):
    M[t,s] = λ^(t-s) for s ≤ t else 0
    z = M @ (B·x)              # einsum over time, fully parallelized
    y = Re(z) projected via C

Two stacks: forward + reverse (bidirectional). Concat outputs.
No Python loop; O(T² N) compute via einsum but T=32 is small.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion


class LinOSSBlock(nn.Module):
    """Linear oscillatory SSM. (B, T, D) -> (B, T, D)."""
    def __init__(self, d_model, n_state=128, dropout=0.1, dt=1.0,
                 omega_init_max=3.0, gamma_init=0.05):
        super().__init__()
        self.d_model = d_model
        self.n_state = n_state
        self.dt = dt
        # Per-state oscillator params: omega (frequency), gamma (damping > 0)
        # Init: omega spread evenly in (0, omega_init_max] for diverse frequencies
        self.omega = nn.Parameter(torch.linspace(0.1, omega_init_max, n_state))
        # log_gamma so positivity guaranteed via exp
        self.log_gamma = nn.Parameter(torch.full((n_state,), float(torch.log(torch.tensor(gamma_init)))))
        # Real input projection D -> n_state
        self.B = nn.Linear(d_model, n_state)
        # Real output projection n_state -> D (uses Re(z))
        self.C = nn.Linear(n_state, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        u = self.B(x)                                      # (B, T, n_state) real
        gamma = self.log_gamma.exp()                        # (n_state,) > 0
        # Complex log-eigenvalue per channel
        log_lam = torch.complex(-gamma * self.dt, self.omega * self.dt)  # (n_state,)

        # Build parallel matrix M[t,s,k] = lam_k^(t-s) for s<=t, else 0
        idx = torch.arange(T, device=x.device, dtype=torch.float32)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)          # (T, T) real, t-s
        mask = (diff >= 0).to(log_lam.real.dtype)          # (T, T)
        # exponent[t,s,k] = (t-s) * log_lam[k]  (complex)
        diff_safe = diff.clamp(min=0)
        exponent = log_lam.view(1, 1, -1) * diff_safe.unsqueeze(-1)  # (T, T, n_state) complex
        M = exponent.exp() * mask.unsqueeze(-1)              # (T, T, n_state) complex

        # Cast u to complex for einsum
        u_c = torch.complex(u, torch.zeros_like(u))         # (B, T, n_state) complex
        # z[b, t, k] = sum_s M[t, s, k] * u_c[b, s, k]
        z = torch.einsum('tsk,bsk->btk', M, u_c)            # (B, T, n_state) complex
        y = z.real                                           # (B, T, n_state) real
        y = self.dropout(self.C(y))                          # (B, T, D)
        return y


class LinOSSTemporalEncoder(nn.Module):
    """Drop-in replacement for MambaTemporalEncoder using LinOSS blocks (bidirectional)."""
    def __init__(self, in_channels, hidden_dim=256, output_dim=None, num_layers=2,
                 n_state=128, dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.fwd_layers = nn.ModuleList([
            LinOSSBlock(hidden_dim, n_state=n_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                LinOSSBlock(hidden_dim, n_state=n_state, dropout=dropout)
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
        # x: (B, C, T, N)
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


class MotionLinOSS(Motion):
    """N2 backbone with LinOSS replacing the temporal Mamba."""
    def __init__(self, *args, lo_hidden_dim=256, lo_num_layers=2, lo_n_state=128,
                 lo_dropout=0.3, lo_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = LinOSSTemporalEncoder(
            in_channels=in_c,
            hidden_dim=lo_hidden_dim,
            output_dim=out_d,
            num_layers=lo_num_layers,
            n_state=lo_n_state,
            dropout=lo_dropout,
            bidirectional=lo_bidirectional,
        )
