"""PGCNetCycleZoo -- PGCNetPruned + augmented-view consistency ("cycle").

A chain of cycle-consistency variants under one model. For each training batch
we classify the clean clip AND a geometrically-transformed view, and add a
symmetric-KL consistency loss between the two logit distributions. cycle_mode
selects the transform T (the "cycle"):

  rot       in-plane uv rotation about the per-sample mean (+/- rot_deg)
  refl      left-right mirror (u -> 2*mean-u)              [reflection axis]
  treverse  time reversal of motion (flip frames on u,v,d) [antisymmetry axis]
  jitter    additive gaussian coord noise
  scale     uv zoom about the per-sample mean (+/- 20%)
  pdrop     point dropout via resample-with-replacement (~37% unique dropped)
  combo     rot + jitter + pdrop stacked

This is the ONE cycle formulation history says can actually help -- a cycle
loss WITH a real transformed view during training (ZRCC-3D style), not
same-input consistency (which collapses to R-Drop). Aux loss only; inference
forward is byte-identical to PGCNetPruned (no aug, aux_loss=None at eval).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pgcnet_pruned import PGCNetPruned


def _sym_kl(p_logits, q_logits, T):
    p = F.log_softmax(p_logits / T, dim=1)
    q = F.log_softmax(q_logits / T, dim=1)
    kl_pq = F.kl_div(p, q.exp(), reduction='batchmean')
    kl_qp = F.kl_div(q, p.exp(), reduction='batchmean')
    return 0.5 * (kl_pq + kl_qp) * (T * T)


class PGCNetCycleZoo(PGCNetPruned):
    def __init__(self, *args, cycle_mode='rot', cycle_weight=1.0,
                 cycle_warm_epochs=5, cycle_decay_epochs=10000,
                 cycle_iters_per_epoch=132, cycle_temp=4.0,
                 rot_deg=15.0, aug_strength=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle_mode = cycle_mode
        self.cycle_weight = cycle_weight
        self.cycle_warm = cycle_warm_epochs
        self.cycle_decay = cycle_decay_epochs
        self.cycle_ipe = cycle_iters_per_epoch
        self.cycle_temp = cycle_temp
        self.rot_deg = rot_deg
        self.aug_strength = aug_strength
        self.register_buffer('_cyc_step', torch.zeros((), dtype=torch.long))
        self.aux_loss = None

    def _cyc_w(self):
        ep = float(self._cyc_step.item()) / max(1, self.cycle_ipe)
        if ep < self.cycle_warm:
            return self.cycle_weight * ep / max(1e-6, self.cycle_warm)
        if ep >= self.cycle_decay:
            return 0.0
        return self.cycle_weight

    def _augment(self, x):                               # x: (B, T, P, C>=4)
        B, T, P, C = x.shape
        x = x.clone()
        mode, a = self.cycle_mode, self.aug_strength
        if mode in ('rot', 'combo'):
            u, v = x[..., 0], x[..., 1]
            cu = u.mean(dim=(1, 2), keepdim=True)
            cv = v.mean(dim=(1, 2), keepdim=True)
            ang = (torch.rand(B, device=x.device) * 2 - 1) * (self.rot_deg * math.pi / 180) * a
            ca, sa = torch.cos(ang).view(B, 1, 1), torch.sin(ang).view(B, 1, 1)
            uu, vv = u - cu, v - cv
            x[..., 0] = ca * uu - sa * vv + cu
            x[..., 1] = sa * uu + ca * vv + cv
        if mode == 'refl':
            u = x[..., 0]
            cu = u.mean(dim=(1, 2), keepdim=True)
            x[..., 0] = 2 * cu - u
        if mode == 'scale':
            u, v = x[..., 0], x[..., 1]
            cu = u.mean(dim=(1, 2), keepdim=True)
            cv = v.mean(dim=(1, 2), keepdim=True)
            s = (1 + (torch.rand(B, device=x.device) * 2 - 1) * 0.2 * a).view(B, 1, 1)
            x[..., 0] = (u - cu) * s + cu
            x[..., 1] = (v - cv) * s + cv
        if mode == 'treverse':
            x[..., :3] = torch.flip(x[..., :3], dims=[1])
        if mode in ('jitter', 'combo'):
            x[..., :3] = x[..., :3] + torch.randn_like(x[..., :3]) * (0.05 * a)
        if mode in ('pdrop', 'combo'):
            idx = torch.randint(0, P, (B, P), device=x.device)
            x = torch.gather(x, 2, idx.view(B, 1, P, 1).expand(B, T, P, C))
        return x

    def _classify(self, x):
        coords = self._sample_points(x)
        fea3 = self._encode_sampled_points(coords)
        s5 = self.stage5(fea3)
        g = self.global_bn(self.pool5(s5)).flatten(1)
        seq = s5.max(dim=3).values
        t = self.temporal_head(seq)
        return self.dual_fc(self.dual_drop(torch.cat([g, t], dim=1)))

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        logits = self._classify(inputs)
        if self.training:
            w = self._cyc_w()
            if w > 0:
                xa = self._augment(inputs)
                la = self._classify(xa)
                self.aux_loss = w * _sym_kl(logits, la, self.cycle_temp)
                self._cyc_step += 1
            else:
                self.aux_loss = None
                self._cyc_step += 1
        else:
            self.aux_loss = None
        return logits

    def extract_features(self, inputs):
        return self.forward(inputs)
