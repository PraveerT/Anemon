"""Revert ST-QNet patch from main.py + utils/parameters.py."""
from pathlib import Path
import re

pp = Path("utils/parameters.py")
psrc = pp.read_text(encoding="utf-8")
psrc = re.sub(
    r"    parser\.add_argument\(\s*\n\s*'--quaternion-cycle-weight',[\s\S]*?help='Weight for unit-quaternion regularization inside cycle loss'\)\n",
    "",
    psrc,
)
pp.write_text(psrc, encoding="utf-8")
print("removed quaternion args from utils/parameters.py")

m = Path("main.py")
src = m.read_text(encoding="utf-8")

src = src.replace(
    "from utils import get_parser, import_class, GpuDataParallel, Optimizer, Recorder, Stat, RandomState\nfrom models.motion import quaternion_cycle_consistency_loss",
    "from utils import get_parser, import_class, GpuDataParallel, Optimizer, Recorder, Stat, RandomState",
)

src = re.sub(
    r"        self\.quaternion_cycle_weight = getattr\(arg, 'quaternion_cycle_weight', 0\.0\)\n"
    r"        self\.quaternion_lambda_unit = getattr\(arg, 'quaternion_lambda_unit', 0\.1\)\n"
    r"        if self\.quaternion_cycle_weight > 0:\n"
    r"            self\.recoder\.print_log\(.*?\)\n",
    "",
    src,
)

src = re.sub(
    r"\n            # ST-QNet quaternion cycle loss\n"
    r"            if self\.quaternion_cycle_weight > 0 and \\\n[\s\S]*?"
    r"                loss = loss \+ self\.quaternion_cycle_weight \* qcc_loss\n",
    "",
    src,
)

m.write_text(src, encoding="utf-8")
print("removed cycle loss from main.py")
