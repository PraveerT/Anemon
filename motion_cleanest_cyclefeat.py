"""Direction A': Global Frame-Feature Cycle Consistency (no quaternions, no
correspondence).

Per ST-QNet's original cycle-consistency thesis, but with:
  - feature-space displacements instead of quaternion rotations
  - 3-anchor stratified sampling (same as old design)
  - global per-frame aggregation (no point correspondence needed)
  - vector cycle: Δ_ab + Δ_bc ≈ Δ_ac instead of quaternion product = identity

Architecture additions over CN-XXL:
  1. AttnPool head: aggregate fea3 (B, 256, T, N) -> per-frame globals (B, T, C)
  2. Displacement MLP: predicts Δ ∈ R^C between any two frames
  3. Auxiliary loss computed inside forward, exposed via get_auxiliary_loss()

main.py already supports model.get_auxiliary_loss() / get_auxiliary_metrics().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.motion_cleanest import CleanestLinXLEncoder


class GlobalFrameCycleHead(nn.Module):
    """Computes per-frame global vectors from (B, C, T, N) features, picks
    3 stratified anchors, and computes the feature cycle loss."""
    def __init__(self, in_channels=256, embed_dim=128, dropout=0.3):
        super().__init__()
        # Attention-pool over N points per frame: a learned weighted sum
        self.attn_score = nn.Linear(in_channels, 1)
        # Project to a smaller embedding for the cycle MLP
        self.embed = nn.Linear(in_channels, embed_dim)
        # Displacement MLP: takes (f_a, f_b) -> Δ ∈ R^embed_dim
        self.delta_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def aggregate(self, x):
        """(B, C, T, N) -> (B, T, C) per-frame attention-pooled global."""
        B, C, T, N = x.shape
        x_perm = x.permute(0, 2, 3, 1)                       # (B, T, N, C)
        scores = self.attn_score(x_perm).squeeze(-1)          # (B, T, N)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)        # (B, T, N, 1)
        return (attn * x_perm).sum(dim=2)                     # (B, T, C)

    def cycle_loss(self, fea):
        """fea: (B, C, T, N). Returns scalar loss + metrics dict."""
        B, C, T, N = fea.shape
        f = self.aggregate(fea)                               # (B, T, C)
        e = self.embed(f)                                     # (B, T, embed)

        # Sample anchors per batch: a from [0, T/3), b from [T/3, 2T/3), c from [2T/3, T)
        t_a = torch.randint(0, max(1, T // 3), (B,), device=fea.device)
        t_b = torch.randint(T // 3, max(T // 3 + 1, 2 * T // 3), (B,), device=fea.device)
        t_c = torch.randint(2 * T // 3, T, (B,), device=fea.device)
        idx_b = torch.arange(B, device=fea.device)
        e_a = e[idx_b, t_a]                                   # (B, embed)
        e_b = e[idx_b, t_b]
        e_c = e[idx_b, t_c]

        d_ab = self.delta_mlp(torch.cat([e_a, e_b], dim=-1))
        d_bc = self.delta_mlp(torch.cat([e_b, e_c], dim=-1))
        d_ac = self.delta_mlp(torch.cat([e_a, e_c], dim=-1))

        residual = d_ab + d_bc - d_ac                          # (B, embed)
        loss = (residual ** 2).sum(dim=-1).mean()              # scalar

        # Metrics: report the L2 of cycle residual + norms of the displacements
        metrics = {
            'cycle_residual_l2': residual.detach().pow(2).sum(-1).mean().sqrt(),
            'delta_norm_mean': (d_ab.detach().pow(2).sum(-1).sqrt().mean()
                                + d_bc.detach().pow(2).sum(-1).sqrt().mean()
                                + d_ac.detach().pow(2).sum(-1).sqrt().mean()) / 3.0,
        }
        return loss, metrics


class _MambaWithCycleHook(nn.Module):
    """Wraps the temporal encoder. On forward, runs encoder + stashes output;
    if in training mode, computes the cycle aux loss on the encoder output."""
    def __init__(self, encoder, cycle_head, cycle_weight):
        super().__init__()
        self.encoder = encoder
        self.cycle_head = cycle_head
        self.cycle_weight = float(cycle_weight)
        self.last_aux_loss = None
        self.last_aux_metrics = {}
        # Expose for any introspecting code
        self.in_channels = getattr(encoder, 'in_channels', 256)
        self.output_dim = getattr(encoder, 'output_dim', 256)

    def forward(self, x):
        y = self.encoder(x)
        if self.training:
            loss, metrics = self.cycle_head.cycle_loss(y)
            self.last_aux_loss = self.cycle_weight * loss
            self.last_aux_metrics = {f'cycle_{k}': v for k, v in metrics.items()}
        else:
            self.last_aux_loss = None
            self.last_aux_metrics = {}
        return y


class MotionCleanestLinXLCycleFeat(Motion):
    """CN-XXL + global frame-feature cycle auxiliary loss.

    Identical architecture to MotionCleanestLinXL; wraps self.mamba so that
    each forward pass also computes the cycle loss on the post-encoder
    features. The wrapper exposes the aux loss to main.py via
    get_auxiliary_loss().
    """
    def __init__(self, *args, lxl_hidden_dim=256, lxl_mlp_dim=512,
                 lxl_num_layers=4, lxl_dropout=0.3, lxl_bidirectional=True,
                 lxl_residual_scale=0.7, cycle_embed_dim=128, cycle_weight=0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        encoder = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale,
        )
        cycle_head = GlobalFrameCycleHead(in_channels=256, embed_dim=cycle_embed_dim)
        # Wrap encoder so its forward also computes the aux loss
        self.mamba = _MambaWithCycleHook(encoder, cycle_head, cycle_weight)

    def get_auxiliary_loss(self):
        return self.mamba.last_aux_loss

    def get_auxiliary_metrics(self):
        return self.mamba.last_aux_metrics

    def load_state_dict(self, state_dict, strict=True):
        """Remap CN-XXL keys (mamba.X) to wrapped keys (mamba.encoder.X) so
        we can load the CN-XXL checkpoint cleanly."""
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('mamba.') and not k.startswith('mamba.encoder.') and not k.startswith('mamba.cycle_head.'):
                remapped['mamba.encoder.' + k[len('mamba.'):]] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)
