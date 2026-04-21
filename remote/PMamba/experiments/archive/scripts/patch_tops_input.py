"""Feed centroid-radial direction ('tops' field) through the FULL architecture.

Same approach as NormalInputMotion but the spatial input is the unit
direction from frame centroid to point, instead of local surface normal.

  normal (v24a): local surface orientation from k-NN PCA (true geometry)
  tops   (v25a): (p - centroid) / |p - centroid|        (radial approximation)

For a roughly convex hand they agree on palm-facing surfaces and differ
at articulated parts (curled fingers). Lets us compare a true normal
field vs the simple centroid-radial approximation under identical
architecture and training.

Class: TopsInputMotion. Config v25a.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class TopsInputMotion(BearingQCCFeatureMotion):
    """BearingQCCFeatureMotion with xyz replaced by unit direction from frame centroid."""

    def extract_features(self, inputs, aux_input=None):
        if isinstance(inputs, dict):
            pts = inputs["points"]
        else:
            pts = inputs
        if pts.dim() == 4:
            B, F_, P, C = pts.shape
            xyz_full = pts[..., :3].float()
        elif pts.dim() == 3:
            B, N, C = pts.shape
            F_ = N // self.pts_size
            P = self.pts_size
            xyz_full = pts[..., :3].float().view(B, F_, P, 3)
        else:
            return super().extract_features(inputs, aux_input=aux_input)

        centroid = xyz_full.mean(dim=2, keepdim=True)                 # (B, F, 1, 3)
        rel = xyz_full - centroid                                     # (B, F, P, 3)
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = (rel / rel_norm).detach()                         # (B, F, P, 3)

        pts_new = pts.clone()
        if pts.dim() == 4:
            pts_new[..., :3] = direction
        else:
            pts_new[..., :3] = direction.view(B, N, 3)

        if isinstance(inputs, dict):
            new_inputs = dict(inputs)
            new_inputs["points"] = pts_new
        else:
            new_inputs = pts_new

        return super().extract_features(new_inputs, aux_input=aux_input)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added TopsInputMotion")
