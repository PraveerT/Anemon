"""Per-point shortest-rotation quaternion as the ONLY feature.

Hypothesis (2026-04-20): the field of per-point shortest-rotation quats
(mapping each centered point in frame t to its Hungarian-corresponded
centered partner in frame t+1) is a structured signal that encodes both
rigid motion AND each point's position relative to the rotation axis.
Richer than scalar residual.

New class: ShortestRotOnlyMotion.
  - Correspondence-pair points (Hungarian cache)
  - Center each frame at its weighted centroid
  - Per-point: axis = (src_c × tgt_c) / |...|,
               angle = acos(src_c · tgt_c / (|src_c| |tgt_c|))
               quat = (cos(angle/2), sin(angle/2) * axis)
  - Invalid points get identity quat (1,0,0,0)
  - Input to classifier: (B, F-1, P, 4) — nothing else reaches the classifier
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class ShortestRotOnlyMotion(nn.Module):
    """Classify gestures using ONLY per-point shortest-rotation quaternions."""

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
        c1, c2, c3 = 64, 128, hidden
        # Per frame-pair: input is (B, 4, P) — 4 quaternion channels, P points.
        self.point_conv = nn.Sequential(
            nn.Conv1d(4, c1, 5, padding=2),
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
    def _shortest_rot_quats(src, tgt, mask):
        """src, tgt: (B, P, 3) centered; mask: (B, P) float.
        Returns: (B, P, 4) quat (w, x, y, z). Invalid -> identity (1,0,0,0)."""
        B, P, _ = src.shape
        device = src.device
        dot = (src * tgt).sum(dim=-1)
        sn = src.norm(dim=-1).clamp(min=1e-6)
        tn = tgt.norm(dim=-1).clamp(min=1e-6)
        cos_a = (dot / (sn * tn)).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        angle = torch.acos(cos_a)                                    # (B, P)
        cross = torch.cross(src, tgt, dim=-1)
        cross_norm = cross.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        axis = cross / cross_norm
        half = angle * 0.5
        w = torch.cos(half).unsqueeze(-1)
        xyz = torch.sin(half).unsqueeze(-1) * axis
        quat = torch.cat([w, xyz], dim=-1)
        identity = torch.zeros_like(quat)
        identity[..., 0] = 1.0
        mb = mask.bool().unsqueeze(-1)
        return torch.where(mb, quat, identity)

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

        quat_list = []
        for t in range(F_ - 1):
            src = xyz[:, t]
            tgt = xyz[:, t + 1]
            idx = tgt_idx_fp[:, t]
            valid = (idx >= 0) & (w_fp[:, t] > 0)
            idx_within = torch.where(valid, idx % P, torch.zeros_like(idx))
            tgt_paired = torch.gather(tgt, 1, idx_within.unsqueeze(-1).expand(-1, -1, 3))

            # Center each frame at its weighted centroid (masked).
            mask_f = valid.float()
            w_sum = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            src_mean = (src * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum.unsqueeze(-1)
            tgt_mean = (tgt_paired * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum.unsqueeze(-1)
            src_c = src - src_mean
            tgt_c = tgt_paired - tgt_mean

            q = self._shortest_rot_quats(src_c, tgt_c, mask_f)      # (B, P, 4)
            quat_list.append(q)

        quats = torch.stack(quat_list, dim=1)                       # (B, F-1, P, 4)

        # Per-frame-pair conv over points
        qf = quats.permute(0, 1, 3, 2).reshape(B * (F_ - 1), 4, P)  # (B*(F-1), 4, P)
        f = self.point_conv(qf).squeeze(-1)                         # (B*(F-1), c3)
        f = f.view(B, F_ - 1, -1).transpose(1, 2)                    # (B, c3, F-1)
        f = self.temporal_conv(f).squeeze(-1)                        # (B, c3)
        f = self.dropout(f)
        return self.classifier(f)

'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added ShortestRotOnlyMotion")
