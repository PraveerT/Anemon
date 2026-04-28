"""Patch main.py + utils/parameters.py to wire ST-QNet quaternion cycle loss.

Adds:
- argparse args: --quaternion-cycle-weight, --quaternion-lambda-unit
- import: from models.motion import quaternion_cycle_consistency_loss
- Processor.__init__: read quaternion_cycle_weight and quaternion_lambda_unit
- in train loop after classification_loss: add cycle loss term if model exposes
  `geometric_quaternions` and `actual_quaternions`
"""
from pathlib import Path
import re

# 1) parser args
pp = Path("utils/parameters.py")
psrc = pp.read_text(encoding="utf-8")
if "'--quaternion-cycle-weight'" not in psrc:
    inject = """    parser.add_argument(
        '--quaternion-cycle-weight',
        type=float,
        default=0.0,
        help='Weight for ST-QNet quaternion cycle consistency loss (0 disables)')
    parser.add_argument(
        '--quaternion-lambda-unit',
        type=float,
        default=0.1,
        help='Weight for unit-quaternion regularization inside cycle loss')
"""
    psrc = psrc.replace("    return parser\n", inject + "    return parser\n", 1)
    pp.write_text(psrc, encoding="utf-8")
    print("added quaternion args to utils/parameters.py")
else:
    print("quaternion args already present")

# 2) main.py
m = Path("main.py")
src = m.read_text(encoding="utf-8")

# import at top
import_line = "from utils import get_parser, import_class, GpuDataParallel, Optimizer, Recorder, Stat, RandomState"
new_import = "from utils import get_parser, import_class, GpuDataParallel, Optimizer, Recorder, Stat, RandomState\nfrom models.motion import quaternion_cycle_consistency_loss"
if "quaternion_cycle_consistency_loss" not in src:
    src = src.replace(import_line, new_import, 1)
    print("added cycle-loss import")

# __init__ config read — find a stable anchor
init_anchor = "        self.best_accuracy = 0.0  # Track best accuracy within current run"
init_inject = (
    "        self.quaternion_cycle_weight = getattr(arg, 'quaternion_cycle_weight', 0.0)\n"
    "        self.quaternion_lambda_unit = getattr(arg, 'quaternion_lambda_unit', 0.1)\n"
    "        if self.quaternion_cycle_weight > 0:\n"
    "            self.recoder.print_log(f'ST-QNet quaternion cycle loss enabled: weight={self.quaternion_cycle_weight}, lambda_unit={self.quaternion_lambda_unit}')\n"
    "        self.best_accuracy = 0.0  # Track best accuracy within current run"
)
if "self.quaternion_cycle_weight" not in src:
    assert init_anchor in src, "init anchor not found"
    src = src.replace(init_anchor, init_inject, 1)
    print("added cycle-loss config to __init__")

# train loop — inject after classification_loss assignment
train_anchor = "            loss = classification_loss\n\n            aux_loss = None"
train_inject = (
    "            loss = classification_loss\n"
    "\n"
    "            # ST-QNet quaternion cycle loss\n"
    "            if self.quaternion_cycle_weight > 0 and \\\n"
    "               hasattr(model_ref, 'geometric_quaternions') and \\\n"
    "               hasattr(model_ref, 'actual_quaternions'):\n"
    "                qcc_loss, _ = quaternion_cycle_consistency_loss(\n"
    "                    model_ref.geometric_quaternions,\n"
    "                    model_ref.actual_quaternions,\n"
    "                    lambda_unit=self.quaternion_lambda_unit,\n"
    "                )\n"
    "                loss = loss + self.quaternion_cycle_weight * qcc_loss\n"
    "\n"
    "            aux_loss = None"
)
if "qcc_loss = quaternion_cycle_consistency_loss" not in src:
    assert train_anchor in src, "train anchor not found"
    src = src.replace(train_anchor, train_inject, 1)
    print("added cycle-loss term to train loop")

m.write_text(src, encoding="utf-8")
print("patched main.py")
