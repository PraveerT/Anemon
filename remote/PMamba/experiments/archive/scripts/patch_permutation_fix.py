"""Fix permutation-equivariance in all XxxOnly/XYZ classifiers.

Replaces kernel_size=5 point-axis convs (which treated unordered points as
an ordered sequence) with kernel_size=1 shared MLPs (classic PointNet
encoder: Linear-per-point + global max-pool = permutation-invariant).
"""
from pathlib import Path
import re

PATH = Path("models/reqnn_motion.py")
src = PATH.read_text(encoding="utf-8")

# Target all point_conv definitions that use kernel_size=5 padding=2 along the
# point axis. Three instances: ResidualOnlyMotion, TopsOnlyMotion,
# ShortestRotOnlyMotion, LocalNormalOnlyMotion, LocalNormalXYZMotion.
# Each has the same structure: Conv1d(in, c1, 5, padding=2) -> Conv1d(c1, c2, 5, padding=2) -> Conv1d(c2, c3, 5, padding=2) -> AdaptiveMaxPool1d(1)
# Replace 5, padding=2 with 1 (removes windowing -> per-point shared MLP).

old_pattern = re.compile(r"(nn\.Conv1d\(\w+, \w+, )5, padding=2(\))")
src2, n = old_pattern.subn(lambda m: m.group(1) + "1" + m.group(2), src)
print(f"replaced {n} kernel_size=5 convs with kernel_size=1 (PointNet-style)")
PATH.write_text(src2, encoding="utf-8")
