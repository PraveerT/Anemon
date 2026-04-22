"""Move the oracle compute + best_metric definition BEFORE the is_new_best
check so the variable exists when it's used. Uses re for whitespace-tolerance.
"""
import re
from pathlib import Path
PATH = Path("main.py")
src = PATH.read_text(encoding="utf-8")

pattern = re.compile(
    r"# Send Telegram message with evaluation results\s*\n"
    r"        try:\s*\n"
    r"            # Check if this is a new best\s*\n"
    r"            # Track best by oracle when available, else by prec1\.\s*\n"
    r"            is_new_best = best_metric > self\.best_accuracy\s*\n"
    r"            if is_new_best:\s*\n"
    r"                self\.best_accuracy = best_metric\s*\n"
    r"                try:\s*\n"
    r"                    best_path = f\"\{self\.arg\.work_dir\}/best_model\.pt\"\s*\n"
    r"                    self\.save_model\(epoch, self\.model, self\.optimizer, best_path\)\s*\n"
    r"                    self\.recoder\.print_log\(f\"  Saved new best to \{best_path\} at \{best_label\}=\{self\.best_accuracy:\.2f\}% \(prec1=\{prec1:\.2f\}%\)\"\)\s*\n"
    r"                except Exception as _e:\s*\n"
    r"                    self\.recoder\.print_log\(f\"Failed to save best_model\.pt: \{_e\}\"\)\s*\n"
    r"\s*\n"
    r"            self\._maybe_compute_oracle\(epoch, mode\)\s*\n"
    r"            # Pick best-metric: oracle if cached PMamba available, else prec1\.\s*\n"
    r"            best_metric = self\._latest_oracle if self\._latest_oracle is not None else prec1\s*\n"
    r"            best_label = \"oracle\" if self\._latest_oracle is not None else \"prec1\""
)

replacement = '''# Send Telegram message with evaluation results
        try:
            # Compute oracle first so best_metric selection can use it.
            self._maybe_compute_oracle(epoch, mode)
            # Pick best-metric: oracle if cached PMamba available, else prec1.
            best_metric = self._latest_oracle if self._latest_oracle is not None else prec1
            best_label = "oracle" if self._latest_oracle is not None else "prec1"
            # Check if this is a new best (by the chosen metric).
            is_new_best = best_metric > self.best_accuracy
            if is_new_best:
                self.best_accuracy = best_metric
                try:
                    best_path = f"{self.arg.work_dir}/best_model.pt"
                    self.save_model(epoch, self.model, self.optimizer, best_path)
                    self.recoder.print_log(f"  Saved new best to {best_path} at {best_label}={self.best_accuracy:.2f}% (prec1={prec1:.2f}%)")
                except Exception as _e:
                    self.recoder.print_log(f"Failed to save best_model.pt: {_e}")'''

match = pattern.search(src)
assert match, "pattern not found — maybe already patched"
src = src[:match.start()] + replacement + src[match.end():]
PATH.write_text(src, encoding="utf-8")
print("fixed: oracle compute + best_metric now defined before is_new_best check")
