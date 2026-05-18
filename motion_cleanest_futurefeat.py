"""Direction A'': Future Feature Prediction.

Auxiliary loss that asks the network to predict the per-frame global feature
at t+k from the feature at t. No quaternions, no correspondence, no trivial
collapse (the target f_{t+k} is detached, so the predictor must match a real
non-zero target).

Architecture additions over CN-XXL:
  1. AttnPool head over (B, C, T, N) -> per-frame global (B, T, C)
  2. Per-step transition MLP: predict f_{t+1} from f_t
  3. Apply k times for k-step lookahead
  4. Auxiliary loss = mean ||f_pred(t+k) - f_target(t+k).detach()||^2

Inside training only; main.py picks up the loss via get_auxiliary_loss().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.motion_cleanest import CleanestLinXLEncoder


class FutureFeaturePredictor(nn.Module):
    """Predicts f_{t+k} from f_t. No trivial-zero collapse because target is
    detached and non-zero."""
    def __init__(self, in_channels=256, embed_dim=128, dropout=0.3, horizons=(1, 2, 4)):
        super().__init__()
        self.embed_dim = embed_dim
        self.horizons = tuple(int(h) for h in horizons)
        # Attention-pool over N points -> per-frame global vector
        self.attn_score = nn.Linear(in_channels, 1)
        # Project to embedding
        self.embed = nn.Linear(in_channels, embed_dim)
        # Per-step transition MLP
        self.transition = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def aggregate(self, x):
        """(B, C, T, N) -> (B, T, C) per-frame attention-pooled global."""
        x_perm = x.permute(0, 2, 3, 1)                    # (B, T, N, C)
        scores = self.attn_score(x_perm).squeeze(-1)      # (B, T, N)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)     # (B, T, N, 1)
        return (attn * x_perm).sum(dim=2)                  # (B, T, C)

    def loss(self, fea):
        """fea: (B, C, T, N). Returns scalar loss + metrics dict."""
        B, C, T, N = fea.shape
        f = self.aggregate(fea)                            # (B, T, C)
        e = self.embed(f)                                   # (B, T, embed)

        total_loss = 0.0
        n_terms = 0
        max_h = max(self.horizons)
        if T <= max_h:
            # Sequence too short for the chosen horizons; return zero loss safely
            zero = e.sum() * 0.0
            return zero, {'future_zero_loss': zero.detach()}

        # Predict step-by-step, then accumulate predictions for each requested horizon
        # We roll forward by applying the transition repeatedly.
        max_h_idx = max(self.horizons)
        # Source frames: t in [0, T - max_h)
        src = e[:, :T - max_h_idx]                          # (B, T-max_h, embed)
        pred = src
        h_preds = {}
        for step in range(1, max_h_idx + 1):
            pred = self.transition(pred) + pred             # residual step (encourages non-trivial)
            if step in self.horizons:
                h_preds[step] = pred

        metrics = {}
        for h, p in h_preds.items():
            target = e[:, h:h + p.shape[1]].detach()        # detached target — no collapse
            diff = (p - target).pow(2).sum(dim=-1)           # (B, T-max_h)
            term = diff.mean()
            total_loss = total_loss + term
            n_terms += 1
            metrics[f'future_h{h}_mse'] = term.detach()
            metrics[f'future_h{h}_tgt_norm'] = target.pow(2).sum(-1).sqrt().mean().detach()

        total_loss = total_loss / max(1, n_terms)
        return total_loss, metrics


class _MambaWithFutureHook(nn.Module):
    """Wraps the temporal encoder. On forward, runs encoder + (if training)
    computes the future-feature auxiliary loss on its output."""
    def __init__(self, encoder, future_head, future_weight):
        super().__init__()
        self.encoder = encoder
        self.future_head = future_head
        self.future_weight = float(future_weight)
        self.last_aux_loss = None
        self.last_aux_metrics = {}
        self.in_channels = getattr(encoder, 'in_channels', 256)
        self.output_dim = getattr(encoder, 'output_dim', 256)

    def forward(self, x):
        y = self.encoder(x)
        if self.training:
            loss, metrics = self.future_head.loss(y)
            self.last_aux_loss = self.future_weight * loss
            self.last_aux_metrics = metrics
        else:
            self.last_aux_loss = None
            self.last_aux_metrics = {}
        return y


class MotionCleanestLinXLFutureFeat(Motion):
    """CN-XXL + Future Feature Prediction auxiliary loss."""
    def __init__(self, *args, lxl_hidden_dim=256, lxl_mlp_dim=512,
                 lxl_num_layers=4, lxl_dropout=0.3, lxl_bidirectional=True,
                 lxl_residual_scale=0.7, future_embed_dim=128, future_weight=0.1,
                 future_horizons=(1, 2, 4), **kwargs):
        super().__init__(*args, **kwargs)
        encoder = CleanestLinXLEncoder(
            in_channels=256, hidden_dim=lxl_hidden_dim, mlp_dim=lxl_mlp_dim,
            output_dim=256, num_layers=lxl_num_layers, dropout=lxl_dropout,
            bidirectional=lxl_bidirectional, residual_scale=lxl_residual_scale,
        )
        future_head = FutureFeaturePredictor(in_channels=256, embed_dim=future_embed_dim,
                                              horizons=tuple(future_horizons))
        self.mamba = _MambaWithFutureHook(encoder, future_head, future_weight)

    def get_auxiliary_loss(self):
        return self.mamba.last_aux_loss

    def get_auxiliary_metrics(self):
        return self.mamba.last_aux_metrics

    def load_state_dict(self, state_dict, strict=True):
        """Remap CN-XXL keys (mamba.X) -> wrapped keys (mamba.encoder.X)."""
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('mamba.') and not k.startswith('mamba.encoder.') and not k.startswith('mamba.future_head.'):
                remapped['mamba.encoder.' + k[len('mamba.'):]] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)
