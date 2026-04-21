"""Make main.py respect config's eval_interval instead of hardcoded 10-until-100.
"""
from pathlib import Path

PATH = Path("main.py")
src = PATH.read_text(encoding="utf-8")

old = "                eval_interval = 10 if (epoch + 1) < 100 else 1"
new = "                eval_interval = getattr(self.arg, 'eval_interval', 10)"

assert old in src, "anchor not found"
src = src.replace(old, new, 1)
PATH.write_text(src, encoding="utf-8")
print("patched main.py: eval_interval now uses config value")
