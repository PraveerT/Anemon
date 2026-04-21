"""Tighter fix: detach R in the already-injected _PartsFeatureProcrustes.

The parts_feature module is already present in reqnn_motion.py from the
earlier apply of patch_parts_feature.py, but it didn't detach R after SVD
and so gradient flow through differentiable SVD exploded at the first
optimizer step (NaN in the real training run). This patch replaces the
post-SVD block with the detached version.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

# Original post-SVD block inside _PartsFeatureProcrustes.forward
old = (
    '            V = Vh.transpose(-1, -2)\n'
    '            det = torch.det(torch.matmul(V, U.transpose(-1, -2)))\n'
    '            D_diag = torch.ones(B, K, 3, device=device)\n'
    '            D_diag[..., -1] = det\n'
    '            D_mat = torch.diag_embed(D_diag)\n'
    '            R = torch.matmul(V, torch.matmul(D_mat, U.transpose(-1, -2)))\n'
    '\n'
    '            bad = ~torch.isfinite(R).all(dim=-1).all(dim=-1)\n'
    '            if bad.any():\n'
    '                R = torch.where(bad.unsqueeze(-1).unsqueeze(-1), I3.expand_as(R), R)\n'
    '\n'
    '            # Rigidity residual per part (how well the single rotation fits).\n'
    '            pred = torch.einsum("bkij,bkpj->bkpi", R, src_c)\n'
    '            residual_per_point = ((pred - tgt_c) ** 2).sum(dim=-1)       # (B, K, P)\n'
    '            residual_per_part = (w_k * residual_per_point).sum(dim=-1) / w_sum.squeeze(-1)  # (B, K)\n'
    '\n'
    '            quats = self._rot_to_quat(R)                                 # (B, K, 4)\n'
)

new = (
    '            V = Vh.transpose(-1, -2)\n'
    '            det = torch.det(torch.matmul(V, U.transpose(-1, -2)))\n'
    '            D_diag = torch.ones(B, K, 3, device=device)\n'
    '            D_diag[..., -1] = det\n'
    '            D_mat = torch.diag_embed(D_diag)\n'
    '            R = torch.matmul(V, torch.matmul(D_mat, U.transpose(-1, -2)))\n'
    '\n'
    '            # Detach R: feature-only path, no SVD gradient needed.\n'
    '            R_used = R.detach()\n'
    '            bad = ~torch.isfinite(R_used).all(dim=-1).all(dim=-1)\n'
    '            if bad.any():\n'
    '                R_used = torch.where(bad.unsqueeze(-1).unsqueeze(-1), I3.expand_as(R_used), R_used)\n'
    '\n'
    '            pred = torch.einsum("bkij,bkpj->bkpi", R_used, src_c)\n'
    '            residual_per_point = ((pred - tgt_c) ** 2).sum(dim=-1)\n'
    '            residual_per_part = (w_k * residual_per_point).sum(dim=-1) / w_sum.squeeze(-1)\n'
    '\n'
    '            quats = self._rot_to_quat(R_used)\n'
)

assert old in src, "old block not found -- file shape has drifted"
src = src.replace(old, new, 1)
PATH.write_text(src, encoding="utf-8")
print("detached R in _PartsFeatureProcrustes (v18a fix)")
