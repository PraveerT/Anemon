"""Residual-only model: classify using ONLY rigid-fit per-point residuals.

Hypothesis test (2026-04-20): if the residual of the best-fit single
rotation per frame-pair IS the articulation signal that distinguishes
gestures, a classifier built from nothing but those residuals should
score well above 4% (random over 25 classes). No EdgeConv, no quaternion
mixer, no rigidity modulation, no raw XYZ features feeding the classifier
— XYZ is used only to compute the residual and is then thrown away.

Adds class `ResidualOnlyMotion` to models/reqnn_motion.py:
  - Correspondence-pair points using corr_full_target_idx (Hungarian cache)
  - Centered Procrustes per frame-pair (differentiable SVD, R detached)
  - Per-point residual = ||R @ src_centered - tgt_centered||^2
  - log1p-normalize, conv1d over points, max-pool
  - conv1d over frame-pairs, max-pool
  - Linear -> num_classes

Config references:
  model: models.reqnn_motion.ResidualOnlyMotion
  model_args:
    num_classes: 25
    pts_size: 96
    hidden: 256
Loader: NvidiaQuaternionQCCParityLoader with Hungarian correspondence.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

model_code = '''

class ResidualOnlyMotion(nn.Module):
    """Pure residual-only classifier. No XYZ features reach the classifier;
    XYZ is consumed only to compute per-point rigid-fit residuals."""

    def __init__(
        self,
        num_classes=25,
        pts_size=96,
        hidden=256,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.pts_size = pts_size
        c1, c2, c3 = 32, 64, hidden
        self.point_conv = nn.Sequential(
            nn.Conv1d(1, c1, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(c1, c2, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(c2, c3, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(c3, c3, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c3, num_classes)
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    @staticmethod
    def _procrustes_residuals(src, tgt, mask):
        """src, tgt: (B, P, 3); mask: (B, P) float (1 where valid).
        Returns: (B, P) per-point squared residual of best-fit rigid rotation."""
        B, P, _ = src.shape
        device = src.device
        w = mask.clamp(min=0.0)
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        src_mean = (src * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum.unsqueeze(-1)
        tgt_mean = (tgt * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum.unsqueeze(-1)
        src_c = src - src_mean
        tgt_c = tgt - tgt_mean
        H = torch.einsum("bp,bpi,bpj->bij", w, src_c, tgt_c)
        H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
        try:
            U, S, Vh = torch.linalg.svd(H)
        except Exception:
            R = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)
            pred = torch.einsum("bij,bpj->bpi", R, src_c)
            return ((pred - tgt_c) ** 2).sum(dim=-1) * mask
        V = Vh.transpose(-1, -2)
        det = torch.det(torch.matmul(V, U.transpose(-1, -2)))
        D_diag = torch.ones(B, 3, device=device)
        D_diag[..., -1] = det
        D_mat = torch.diag_embed(D_diag)
        R = torch.matmul(V, torch.matmul(D_mat, U.transpose(-1, -2)))
        # Detach R to avoid differentiable SVD gradient issues -- residual magnitude
        # is still meaningful (it's a geometric quantity).
        R = R.detach()
        bad = ~torch.isfinite(R).all(dim=-1).all(dim=-1)
        if bad.any():
            R = torch.where(
                bad.unsqueeze(-1).unsqueeze(-1),
                torch.eye(3, device=device).unsqueeze(0).expand_as(R),
                R,
            )
        pred = torch.einsum("bij,bpj->bpi", R, src_c)
        res = ((pred - tgt_c) ** 2).sum(dim=-1)
        return res * mask

    def forward(self, sample):
        # sample: dict with points (B,F,P,C), corr_full_target_idx, corr_full_weight
        pts = sample["points"]
        if pts.dim() == 3:
            # already-flattened (B, F*P, C) — reshape to (B, F, P, C)
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape
        device = pts.device

        xyz = pts[..., :3].float()
        tgt_idx_full = sample["corr_full_target_idx"].view(B, F_, P).long()
        weight_full = sample["corr_full_weight"].view(B, F_, P).float()

        residuals_list = []
        arange_P = torch.arange(P, device=device)
        for t in range(F_ - 1):
            src = xyz[:, t]                          # (B, P, 3)
            tgt = xyz[:, t + 1]                      # (B, P, 3)
            idx = tgt_idx_full[:, t]                 # (B, P) flat target index
            valid = (idx >= 0) & (weight_full[:, t] > 0)  # (B, P)
            idx_within = torch.where(valid, idx % P, torch.zeros_like(idx))
            tgt_paired = torch.gather(
                tgt, 1, idx_within.unsqueeze(-1).expand(-1, -1, 3)
            )
            res = self._procrustes_residuals(src, tgt_paired, valid.float())
            residuals_list.append(res)

        residuals = torch.stack(residuals_list, dim=1)  # (B, F-1, P)
        residuals = torch.log1p(residuals)

        # Per-frame-pair conv over points
        r = residuals.view(B * (F_ - 1), 1, P)
        f = self.point_conv(r).squeeze(-1)              # (B*(F-1), c3)
        f = f.view(B, F_ - 1, -1).transpose(1, 2)        # (B, c3, F-1)
        f = self.temporal_conv(f).squeeze(-1)            # (B, c3)
        f = self.dropout(f)
        logits = self.classifier(f)
        return logits

'''

# Append to end of file
src = src.rstrip() + "\n" + model_code
PATH.write_text(src, encoding="utf-8")
print("added ResidualOnlyMotion to models/reqnn_motion.py")
