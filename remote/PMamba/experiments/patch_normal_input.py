"""Feed local-normal directions through the FULL BearingQCCFeatureMotion architecture.

Replaces the xyz spatial input channels with per-point local surface normals
(PCA on k-NN neighborhood, sign-flipped toward centroid). Keeps channel 3+
(time, aux) intact. Everything else in the architecture — EdgeConv, quaternion
mixer, rigidity modulation, attention-pooled readout — is unchanged.

This isolates: can the FULL architecture classify using the normal field?
If yes, normals carry signal and the earlier "normal-only" tests were
architecture-limited.
If no, normals genuinely don't carry discriminative signal.

Class: NormalInputMotion. Config v24a.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class NormalInputMotion(BearingQCCFeatureMotion):
    """BearingQCCFeatureMotion with xyz replaced by per-point local normals."""

    def __init__(self, *args, knn_k_normal=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.knn_k_normal = knn_k_normal

    def extract_features(self, inputs, aux_input=None):
        if isinstance(inputs, dict):
            pts = inputs["points"]
        else:
            pts = inputs
        orig_shape = pts.shape
        if pts.dim() == 4:
            B, F_, P, C = orig_shape
            xyz_full = pts[..., :3].float()
        elif pts.dim() == 3:
            B, N, C = orig_shape
            F_ = N // self.pts_size
            P = self.pts_size
            xyz_full = pts[..., :3].float().view(B, F_, P, 3)
        else:
            return super().extract_features(inputs, aux_input=aux_input)

        # Compute local normals per frame.
        normals = LocalNormalOnlyMotion._compute_local_normals(
            xyz_full, self.knn_k_normal
        ).detach()                                                    # (B, F, P, 3)

        pts_new = pts.clone()
        if pts.dim() == 4:
            pts_new[..., :3] = normals
        else:
            pts_new[..., :3] = normals.view(B, N, 3)

        if isinstance(inputs, dict):
            new_inputs = dict(inputs)
            new_inputs["points"] = pts_new
        else:
            new_inputs = pts_new

        return super().extract_features(new_inputs, aux_input=aux_input)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added NormalInputMotion")
