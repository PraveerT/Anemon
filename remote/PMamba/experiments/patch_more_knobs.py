"""Make mamba_output_dim, ms_num_scales, ms_feature_dim configurable on Motion.

stage5 input dim must follow mamba_output_dim (concat coords=4 + mamba_output_dim).
"""
from pathlib import Path
import re

m = Path("models/motion.py")
src = m.read_text(encoding="utf-8")

# Add args to Motion.__init__ if not present
old_sig = "    def __init__(self, num_classes, pts_size, topk=16, downsample=(2, 2, 2), mamba_hidden_dim=128, mamba_num_layers=2,"
new_sig = "    def __init__(self, num_classes, pts_size, topk=16, downsample=(2, 2, 2), mamba_hidden_dim=128, mamba_num_layers=2, mamba_output_dim=256, ms_num_scales=4, ms_feature_dim=32,"
if "mamba_output_dim" not in src:
    assert old_sig in src, "Motion init signature anchor missing (run patch_mamba_dim.py first)"
    src = src.replace(old_sig, new_sig, 1)
    print("added mamba_output_dim/ms_num_scales/ms_feature_dim args")
else:
    print("extra knob args already present")

# Reroute the MambaTemporalEncoder construction
old_mamba = "        self.mamba = MambaTemporalEncoder(in_channels=256, hidden_dim=mamba_hidden_dim, output_dim=256, num_layers=mamba_num_layers)"
new_mamba = "        self.mamba = MambaTemporalEncoder(in_channels=256, hidden_dim=mamba_hidden_dim, output_dim=mamba_output_dim, num_layers=mamba_num_layers)"
if old_mamba in src:
    src = src.replace(old_mamba, new_mamba, 1)
    print("rerouted Mamba output_dim")

# stage5 must follow mamba_output_dim: stage5 input = 4 (coords) + mamba_output_dim
old_stage5 = "        self.stage5 = MLPBlock([260, 1024], 2)  # Updated from 512 to 260 (fea3 channels)"
new_stage5 = "        self.stage5 = MLPBlock([4 + mamba_output_dim, 1024], 2)"
if old_stage5 in src:
    src = src.replace(old_stage5, new_stage5, 1)
    print("updated stage5 to follow mamba_output_dim")

# multi_scale
old_ms = "        self.multi_scale = MultiScaleFeatureProcessor(in_channels=132, num_scales=4, feature_dim=32)"
new_ms = "        self.multi_scale = MultiScaleFeatureProcessor(in_channels=132, num_scales=ms_num_scales, feature_dim=ms_feature_dim)"
if old_ms in src:
    src = src.replace(old_ms, new_ms, 1)
    print("rerouted MultiScale num_scales/feature_dim")

m.write_text(src, encoding="utf-8")
print("patched models/motion.py")
