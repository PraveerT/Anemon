"""Local normal QUATERNION + XYZ coords as classifier input.

Same local-normal derivation as v22a, but classifier sees 7 channels
per point per frame: [x, y, z, qw, qx, qy, qz]. Tests whether the
derived field becomes useful once the classifier also has raw position.

Class: LocalNormalXYZMotion. Config v23a.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class LocalNormalXYZMotion(nn.Module):
    """XYZ coords + local-normal quaternion per point per frame (7 channels)."""

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
            nn.Conv1d(7, c1, 5, padding=2),
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

    def forward(self, sample):
        pts = sample["points"]
        if pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            pts = pts.view(B, F_, self.pts_size, C)
        B, F_, P, C = pts.shape

        xyz = pts[..., :3].float()

        # Local-normal via PCA (same as v22a)
        normals = LocalNormalOnlyMotion._compute_local_normals(xyz, self.knn_k).detach()
        q = LocalNormalOnlyMotion._quat_from_north(
            normals.view(B * F_, P, 3)
        ).view(B, F_, P, 4)

        # Concat xyz + quat -> 7 channels per point per frame
        feat = torch.cat([xyz, q], dim=-1)                           # (B, F, P, 7)

        # Per-frame conv over points
        fp = feat.permute(0, 1, 3, 2).reshape(B * F_, 7, P)
        f = self.point_conv(fp).squeeze(-1)
        f = f.view(B, F_, -1).transpose(1, 2)
        f = self.temporal_conv(f).squeeze(-1)
        f = self.dropout(f)
        return self.classifier(f)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added LocalNormalXYZMotion")
