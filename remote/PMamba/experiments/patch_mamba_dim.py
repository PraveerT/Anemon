"""Patch models/motion.py: make Mamba hidden_dim/num_layers configurable via Motion.__init__.

After patch, Motion.__init__ accepts mamba_hidden_dim (default 128) and
mamba_num_layers (default 2). The MambaTemporalEncoder line is updated to use them.
"""
from pathlib import Path
import re

m = Path("models/motion.py")
src = m.read_text(encoding="utf-8")

old_init = "    def __init__(self, num_classes, pts_size, topk=16, downsample=(2, 2, 2),"
new_init = "    def __init__(self, num_classes, pts_size, topk=16, downsample=(2, 2, 2), mamba_hidden_dim=128, mamba_num_layers=2,"
if "mamba_hidden_dim" not in src:
    assert old_init in src, "init signature anchor not found"
    src = src.replace(old_init, new_init, 1)
    print("added mamba_hidden_dim/mamba_num_layers to Motion.__init__")
else:
    print("mamba args already present in Motion.__init__")

old_mamba = "        self.mamba = MambaTemporalEncoder(in_channels=256, hidden_dim=128, output_dim=256, num_layers=2)"
new_mamba = "        self.mamba = MambaTemporalEncoder(in_channels=256, hidden_dim=mamba_hidden_dim, output_dim=256, num_layers=mamba_num_layers)"
if old_mamba in src:
    src = src.replace(old_mamba, new_mamba, 1)
    print("rerouted Mamba hidden_dim/num_layers to args")

m.write_text(src, encoding="utf-8")
print("patched models/motion.py")
