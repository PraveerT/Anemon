"""Residual-only fix: normalize residuals per-sample so conv input has real signal.

First launch got stuck at random (4.56%) because residuals are tiny (median
0.008, p90 0.027 in normalized-point units^2). log1p stays near 0, convs
produce constant features, classifier outputs uniform. Fix: per-sample
z-normalize the residuals before feeding into the point conv.
"""
from pathlib import Path

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

old = (
    "        residuals = torch.stack(residuals_list, dim=1)  # (B, F-1, P)\n"
    "        residuals = torch.log1p(residuals)\n"
    "\n"
    "        # Per-frame-pair conv over points\n"
    "        r = residuals.view(B * (F_ - 1), 1, P)\n"
)

new = (
    "        residuals = torch.stack(residuals_list, dim=1)  # (B, F-1, P)\n"
    "\n"
    "        # Per-sample z-normalize so classifier sees a scale-free signal.\n"
    "        res_flat = residuals.reshape(B, -1)\n"
    "        res_mean = res_flat.mean(dim=-1, keepdim=True)\n"
    "        res_std = res_flat.std(dim=-1, keepdim=True).clamp(min=1e-6)\n"
    "        residuals = (residuals - res_mean.unsqueeze(-1)) / res_std.unsqueeze(-1)\n"
    "\n"
    "        # Per-frame-pair conv over points\n"
    "        r = residuals.view(B * (F_ - 1), 1, P)\n"
)

assert old in src, "old block not found"
src = src.replace(old, new, 1)
PATH.write_text(src, encoding="utf-8")
print("residual z-normalization applied")
