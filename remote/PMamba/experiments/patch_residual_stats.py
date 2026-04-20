"""Residual-only v2: use per-frame-pair sorted top-K + stats instead of raw.

Rewrites ResidualOnlyMotion.forward to extract a compact articulation
signature per frame-pair, so the classifier sees actual discriminative
structure even though residuals are tiny.

Signature per frame-pair (F_k):
  - Sort residuals descending
  - Take top-K = 32 values (captures articulation magnitude)
  - Append stats: [mean, std, p50, p75, p90, p95, max]  (7 values)
  - Total: 39 features per frame-pair

Classifier:
  - Conv1d over frame-pairs on the 39-channel signal
  - Global pool + linear
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

# Replace the class body end-to-end by rewriting the forward and constructor.
# First, strip the existing ResidualOnlyMotion class.
import re
pat = re.compile(r"\nclass ResidualOnlyMotion\(nn\.Module\):.*?(?=\nclass |\Z)", re.DOTALL)
src = pat.sub("\n", src)

new_class = '''
class ResidualOnlyMotion(nn.Module):
    """Pure residual-only classifier using sorted top-K + stat features."""

    def __init__(
        self,
        num_classes=25,
        pts_size=96,
        hidden=256,
        dropout=0.1,
        top_k=32,
        **kwargs,
    ):
        super().__init__()
        self.pts_size = pts_size
        self.top_k = top_k
        self.num_stats = 7                       # mean, std, p50, p75, p90, p95, max
        self.per_pair_dim = top_k + self.num_stats

        c1, c2 = 128, hidden
        self.pair_mlp = nn.Sequential(
            nn.Linear(self.per_pair_dim, c1),
            nn.GELU(),
            nn.Linear(c1, c2),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c2, num_classes)
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    @staticmethod
    def _procrustes_residuals(src, tgt, mask):
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
        pts = sample["points"]
        if pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape
        device = pts.device

        xyz = pts[..., :3].float()
        tgt_idx_fp = sample["corr_full_target_idx"].view(B, F_, P).long()
        w_fp = sample["corr_full_weight"].view(B, F_, P).float()

        per_pair_feats = []
        K = min(self.top_k, P)
        for t in range(F_ - 1):
            src = xyz[:, t]
            tgt = xyz[:, t + 1]
            idx = tgt_idx_fp[:, t]
            valid = (idx >= 0) & (w_fp[:, t] > 0)
            idx_within = torch.where(valid, idx % P, torch.zeros_like(idx))
            tgt_paired = torch.gather(tgt, 1, idx_within.unsqueeze(-1).expand(-1, -1, 3))
            res = self._procrustes_residuals(src, tgt_paired, valid.float())  # (B, P)

            # sorted descending top-K (retains largest articulation signal)
            top_vals, _ = torch.topk(res, K, dim=-1, largest=True)

            # stats over all valid residuals (non-zero after masking)
            valid_f = valid.float().clamp(min=1e-6)
            w_sum = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mean = (res * valid_f).sum(dim=-1, keepdim=True) / w_sum
            var = ((res - mean) ** 2 * valid_f).sum(dim=-1, keepdim=True) / w_sum
            std = var.sqrt()
            # quantiles: sort ascending, take at k=int(qN) after removing invalid
            sorted_res, _ = torch.sort(res, dim=-1)
            p50 = sorted_res[:, P // 2 : P // 2 + 1]
            p75 = sorted_res[:, int(0.75 * P) : int(0.75 * P) + 1]
            p90 = sorted_res[:, int(0.90 * P) : int(0.90 * P) + 1]
            p95 = sorted_res[:, int(0.95 * P) : int(0.95 * P) + 1]
            mx = sorted_res[:, -1:]
            stats = torch.cat([mean, std, p50, p75, p90, p95, mx], dim=-1)  # (B, 7)

            pair_feat = torch.cat([top_vals, stats], dim=-1)  # (B, K+7)
            per_pair_feats.append(pair_feat)

        pair_stack = torch.stack(per_pair_feats, dim=1)                    # (B, F-1, K+7)

        # Per-sample z-normalize the full feature tensor so scale is consistent
        flat = pair_stack.reshape(B, -1)
        mean_n = flat.mean(dim=-1, keepdim=True)
        std_n = flat.std(dim=-1, keepdim=True).clamp(min=1e-6)
        pair_stack = (pair_stack - mean_n.unsqueeze(-1)) / std_n.unsqueeze(-1)

        # Per-pair MLP
        pp = self.pair_mlp(pair_stack)                                      # (B, F-1, c2)
        pp = pp.transpose(1, 2)                                             # (B, c2, F-1)
        fvec = self.temporal(pp).squeeze(-1)                                # (B, c2)
        fvec = self.dropout(fvec)
        return self.classifier(fvec)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("rewrote ResidualOnlyMotion with top-K + stats features")
