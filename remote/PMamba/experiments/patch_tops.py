"""Space-filling-tops model: per-point per-frame orientation quaternion.

Each point in each frame becomes a quaternion: the rotation that takes
canonical "north" (0,1,0) into the unit direction from the frame's weighted
centroid to that point. As the hand deforms through time, each point's
quaternion evolves — the hand "disturbing" a field of tops initially all
pointing north.

Classifier sees ONLY this field:
  input shape (B, F, P, 4) -- no XYZ, no correspondence, no residuals.

The tiny v19a/v20a tests ruled out scalar residual and pair-wise shortest-
rot quat. This one gives the classifier a full per-point orientation field
across all 32 frames, so the articulation pattern has a chance to show up
as temporal dynamics on the quaternion manifold.

Class: TopsOnlyMotion. Config v21a.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class TopsOnlyMotion(nn.Module):
    """Pure "tops field" classifier: per-point per-frame orientation quats."""

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
        # Per frame: input is (B, 4, P). Convs along points.
        c1, c2, c3 = 64, 128, hidden
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
    def _quat_from_north(direction):
        """direction: (B, P, 3) unit vector. Returns (B, P, 4) quat taking (0,1,0) to direction."""
        B, P, _ = direction.shape
        device = direction.device
        north = torch.zeros_like(direction)
        north[..., 1] = 1.0
        dot = (north * direction).sum(dim=-1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        angle = torch.acos(dot)                              # (B, P)
        cross = torch.cross(north, direction, dim=-1)        # (B, P, 3)
        cross_n = cross.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        axis = cross / cross_n
        half = angle * 0.5
        w = torch.cos(half).unsqueeze(-1)
        xyz = torch.sin(half).unsqueeze(-1) * axis
        return torch.cat([w, xyz], dim=-1)                    # (B, P, 4)

    def forward(self, sample):
        pts = sample["points"]
        if pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape
        device = pts.device

        xyz = pts[..., :3].float()                            # (B, F, P, 3)

        # Weighted centroid per frame (uniform weights since we have all points).
        centroid = xyz.mean(dim=2, keepdim=True)              # (B, F, 1, 3)
        rel = xyz - centroid                                  # (B, F, P, 3)
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = rel / rel_norm                            # (B, F, P, 3)

        # Per-point per-frame quaternion taking north -> direction
        q = self._quat_from_north(direction.view(B * F_, P, 3)).view(B, F_, P, 4)

        # Per-frame conv over points
        qf = q.permute(0, 1, 3, 2).reshape(B * F_, 4, P)      # (B*F, 4, P)
        f = self.point_conv(qf).squeeze(-1)                    # (B*F, c3)
        f = f.view(B, F_, -1).transpose(1, 2)                  # (B, c3, F)
        f = self.temporal_conv(f).squeeze(-1)                  # (B, c3)
        f = self.dropout(f)
        return self.classifier(f)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added TopsOnlyMotion")
