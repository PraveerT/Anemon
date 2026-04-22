"""Track best checkpoint by oracle (vs cached PMamba) when available, else by
solo prec1. Updates best_model.pt and the "New Best" telegram indicator to
reflect whichever metric is active.
"""
from pathlib import Path

PATH = Path("main.py")
src = PATH.read_text(encoding="utf-8")

# Replace the best-tracking block so it uses _latest_oracle when present.
old = '''            self._maybe_compute_oracle(epoch, mode)
            # Format message as: Train: train acc train loss Test: test acc test loss
            if train_acc is not None and train_loss is not None:'''
new = '''            self._maybe_compute_oracle(epoch, mode)
            # Pick best-metric: oracle if cached PMamba available, else prec1.
            best_metric = self._latest_oracle if self._latest_oracle is not None else prec1
            best_label = "oracle" if self._latest_oracle is not None else "prec1"
            # Format message as: Train: train acc train loss Test: test acc test loss
            if train_acc is not None and train_loss is not None:'''
assert old in src, "anchor 1 missing"
src = src.replace(old, new, 1)

old2 = '''            is_new_best = prec1 > self.best_accuracy
            if is_new_best:
                self.best_accuracy = prec1'''
new2 = '''            # Track best by oracle when available, else by prec1.
            is_new_best = best_metric > self.best_accuracy
            if is_new_best:
                self.best_accuracy = best_metric'''
assert old2 in src, "anchor 2 missing"
src = src.replace(old2, new2, 1)

# Tag the telegram "New Best" to indicate which metric. Replace both branches.
old3 = '''                if is_new_best:
                    message += f" ? New Best: {self.best_accuracy:.1f}%"'''
new3 = '''                if is_new_best:
                    message += f" \u2728 New Best [{best_label}]: {self.best_accuracy:.2f}%"'''
if old3 in src:
    src = src.replace(old3, new3, 1)

old4 = '''                if is_new_best:
                    message += f"? New Best: {self.best_accuracy:.1f}%\\n"'''
new4 = '''                if is_new_best:
                    message += f"\u2728 New Best [{best_label}]: {self.best_accuracy:.2f}%\\n"'''
if old4 in src:
    src = src.replace(old4, new4, 1)

# Also update the saved-path log line
old5 = '''                    self.recoder.print_log(f"  Saved new best to {best_path} at {prec1:.2f}%")'''
new5 = '''                    self.recoder.print_log(f"  Saved new best to {best_path} at {best_label}={self.best_accuracy:.2f}% (prec1={prec1:.2f}%)")'''
if old5 in src:
    src = src.replace(old5, new5, 1)

PATH.write_text(src, encoding="utf-8")
print("patched main.py: best_model.pt now saved on best oracle (falls back to prec1)")
