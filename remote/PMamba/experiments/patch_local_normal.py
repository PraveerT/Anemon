"""Local-surface-normal quaternion field ONLY classifier.

Per point per frame:
  - Find k=10 nearest neighbors (same frame)
  - PCA on neighborhood (smallest eigenvector = normal direction)
  - Flip sign so normal points toward frame centroid (consistent orientation)
  - Quaternion from (0,1,0) to normal direction
Classifier sees ONLY this (B, F, P, 4) field. No XYZ.

Class: LocalNormalOnlyMotion. Config v22a.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class LocalNormalOnlyMotion(nn.Module):
    """Classify from local-normal quaternion field only. No XYZ input."""

    def __init__(
        self,
        num_classes=25,
        pts_size=96,
        hidden=256,
        dropout=0.1,
        knn_k=10,
        **kwargs,
    ):
        super().__init__()
        self.pts_size = pts_size
        self.knn_k = knn_k
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
    def _compute_local_normals(xyz, k):
        """xyz: (B, F, P, 3); returns (B, F, P, 3) unit normals, sign-flipped toward centroid."""
        B, F_, P, _ = xyz.shape
        pts = xyz.reshape(B * F_, P, 3)
        dist = torch.cdist(pts, pts)                                 # (B*F, P, P)
        _, idx = torch.topk(dist, k=min(k + 1, P), largest=False, dim=-1)
        idx = idx[:, :, 1:]                                          # drop self
        k_eff = idx.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        pts_exp = pts.unsqueeze(1).expand(-1, P, -1, -1)             # (B*F, P, P, 3)
        neighbors = torch.gather(pts_exp, 2, idx_exp)                # (B*F, P, k, 3)
        nmean = neighbors.mean(dim=2, keepdim=True)
        nc = neighbors - nmean                                        # (B*F, P, k, 3)
        try:
            _, _, Vh = torch.linalg.svd(nc, full_matrices=False)     # Vh: (B*F, P, 3, 3)
        except Exception:
            return torch.zeros_like(pts).reshape(B, F_, P, 3)
        normals = Vh[..., -1, :]                                      # (B*F, P, 3)
        # Sign flip so normal . (centroid - p) > 0 (i.e. points TOWARD centroid).
        centroid = pts.mean(dim=1, keepdim=True)
        to_centroid = centroid - pts
        sign = torch.sign((normals * to_centroid).sum(dim=-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        normals = normals * sign
        # Normalize again just in case.
        nn2 = normals.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        normals = normals / nn2
        return normals.reshape(B, F_, P, 3)

    @staticmethod
    def _quat_from_north(direction):
        B, P, _ = direction.shape
        device = direction.device
        north = torch.zeros_like(direction)
        north[..., 1] = 1.0
        dot = (north * direction).sum(dim=-1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        angle = torch.acos(dot)
        cross = torch.cross(north, direction, dim=-1)
        cross_n = cross.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        axis = cross / cross_n
        half = angle * 0.5
        w = torch.cos(half).unsqueeze(-1)
        xyz = torch.sin(half).unsqueeze(-1) * axis
        return torch.cat([w, xyz], dim=-1)

    def forward(self, sample):
        pts = sample["points"]
        if pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape

        xyz = pts[..., :3].float()
        normals = self._compute_local_normals(xyz, self.knn_k)       # (B, F, P, 3)

        # Detach normals to avoid SVD gradient instability.
        normals = normals.detach()

        q = self._quat_from_north(normals.view(B * F_, P, 3)).view(B, F_, P, 4)

        qf = q.permute(0, 1, 3, 2).reshape(B * F_, 4, P)
        f = self.point_conv(qf).squeeze(-1)
        f = f.view(B, F_, -1).transpose(1, 2)
        f = self.temporal_conv(f).squeeze(-1)
        f = self.dropout(f)
        return self.classifier(f)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added LocalNormalOnlyMotion")
