"""PGCNetTCycle -- PGCNetPruned + a self-supervised temporal forward/backward
cycle on the per-frame descriptor sequence (s5 N-pooled, shape (B,1024,T)).

Motivation: the temporal mechanism on this net is decorative -- the classifier
ignores per-frame order (global-max does the work). This aux task FORCES the
encoder to model dynamics: learn an inverse latent transition pair
  F: z_t -> z_{t+1}   (forward dynamics)
  B: z_t -> z_{t-1}   (backward dynamics)
with three losses -- predict-next, predict-prev, and a CYCLE term B(F(z))=z --
so the cycle carries REAL temporal content (a t->t+1 shift), not same-frame
consistency (which is the R-Drop trap every prior cycle attempt fell into).

Gradient flows seq -> s5 -> stage5 -> encoder, so the spatial features are
shaped to be temporally predictable. Aux loss only; at inference self.training
is False so aux is skipped and the forward output is byte-identical to
PGCNetPruned (the cycle module contributes nothing at test time).

CONTROL: cycle_shuffle=True permutes the time axis before the cycle, destroying
temporal order. If the shuffled control lifts as much as the ordered version,
the lift is regularization (decorative), not temporal structure.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pgcnet_pruned import PGCNetPruned


class _Transition(nn.Module):
    """Residual MLP latent transition operator."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, z):
        return z + self.net(z)


class PGCNetTCycle(PGCNetPruned):
    def __init__(self, *args, cycle_latent=128, cycle_hidden=256,
                 cycle_weight=0.2, cycle_warm_epochs=3, cycle_decay_epochs=90,
                 cycle_iters_per_epoch=132, cycle_var_weight=0.1,
                 cycle_shuffle=False, **kwargs):
        super().__init__(*args, **kwargs)
        L = cycle_latent
        self.cyc_proj = nn.Sequential(nn.Linear(1024, L), nn.LayerNorm(L))
        self.cyc_fwd = _Transition(L, cycle_hidden)
        self.cyc_bwd = _Transition(L, cycle_hidden)
        self.cycle_weight = cycle_weight
        self.cycle_warm = cycle_warm_epochs
        self.cycle_decay = cycle_decay_epochs
        self.cycle_ipe = cycle_iters_per_epoch
        self.cycle_var_weight = cycle_var_weight
        self.cycle_shuffle = cycle_shuffle
        self.register_buffer('_cyc_step', torch.zeros((), dtype=torch.long))
        self.aux_loss = None

    def _cyc_w(self):
        """Warmup -> cosine-decay-to-zero weight: a transient early nudge."""
        ep = float(self._cyc_step.item()) / max(1, self.cycle_ipe)
        if ep < self.cycle_warm:
            return self.cycle_weight * ep / max(1e-6, self.cycle_warm)
        if ep >= self.cycle_decay:
            return 0.0
        p = (ep - self.cycle_warm) / max(1e-6, self.cycle_decay - self.cycle_warm)
        return self.cycle_weight * 0.5 * (1.0 + math.cos(math.pi * p))

    def _cycle_loss(self, seq):                          # seq: (B, 1024, T)
        z = self.cyc_proj(seq.transpose(1, 2))           # (B, T, L), per-frame LN
        if self.cycle_shuffle:                           # CONTROL: break time order
            idx = torch.randperm(z.shape[1], device=z.device)
            z = z[:, idx]
        zt, znext = z[:, :-1], z[:, 1:]
        l_f = F.mse_loss(self.cyc_fwd(zt), znext)        # predict t+1
        l_b = F.mse_loss(self.cyc_bwd(znext), zt)        # predict t-1
        l_cyc = F.mse_loss(self.cyc_bwd(self.cyc_fwd(zt)), zt)   # F then B = id
        std_t = z.std(dim=1)                             # (B, L) variance over time
        l_var = torch.relu(1.0 - std_t).mean()           # anti-collapse
        return l_f + l_b + l_cyc + self.cycle_var_weight * l_var

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)
        fea3 = self._encode_sampled_points(coords)
        s5 = self.stage5(fea3)
        g = self.global_bn(self.pool5(s5)).flatten(1)
        seq = s5.max(dim=3).values                       # (B, 1024, T)
        t = self.temporal_head(seq)
        logits = self.dual_fc(self.dual_drop(torch.cat([g, t], dim=1)))
        if self.training:
            w = self._cyc_w()
            self.aux_loss = w * self._cycle_loss(seq) if w > 0 else None
            self._cyc_step += 1
        else:
            self.aux_loss = None
        return logits

    def extract_features(self, inputs):
        return self.forward(inputs)
