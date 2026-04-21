"""Make main.py save best_model.pt whenever eval accuracy exceeds prior best."""
from pathlib import Path

PATH = Path("main.py")
src = PATH.read_text(encoding="utf-8")

old = """            is_new_best = prec1 > self.best_accuracy
            if is_new_best:
                self.best_accuracy = prec1"""
new = """            is_new_best = prec1 > self.best_accuracy
            if is_new_best:
                self.best_accuracy = prec1
                try:
                    best_path = f"{self.arg.work_dir}/best_model.pt"
                    self.save_model(epoch, self.model, self.optimizer, best_path)
                    self.recoder.print_log(f"  Saved new best to {best_path} at {prec1:.2f}%")
                except Exception as _e:
                    self.recoder.print_log(f"Failed to save best_model.pt: {_e}")"""

assert old in src, "anchor missing"
src = src.replace(old, new, 1)
PATH.write_text(src, encoding="utf-8")
print("patched main.py: saves best_model.pt on every new best eval")
