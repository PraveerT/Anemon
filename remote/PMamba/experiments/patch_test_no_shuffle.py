"""Disable shuffle on the test DataLoader so oracle hook aligns with cache."""
from pathlib import Path
PATH = Path("main.py")
src = PATH.read_text(encoding="utf-8")

old = '''            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.arg.num_worker,
            )'''
new = '''            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.arg.test_batch_size,
                shuffle=False,                 # deterministic for oracle alignment
                drop_last=False,
                num_workers=self.arg.num_worker,
            )'''
assert old in src, "anchor missing"
src = src.replace(old, new, 1)
PATH.write_text(src, encoding="utf-8")
print("patched test loader: shuffle=False")
