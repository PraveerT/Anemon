"""Add per-sample classification-loss reweighting hook to main.py.

After computing the raw logits, if the active model exposes a scalar
`latest_sample_weights` tensor of shape (B,) from its last forward, we replace
the mean CE with a weighted mean using those weights. Otherwise main.py
behaves unchanged.
"""
from pathlib import Path

PATH = Path("main.py")
src = PATH.read_text(encoding="utf-8")

old = "            classification_loss = torch.mean(self.loss(output, label))\n            loss = classification_loss"
new = (
    "            sample_w = getattr(model_ref, 'latest_sample_weights', None)\n"
    "            if sample_w is not None:\n"
    "                per_sample = torch.nn.functional.cross_entropy(output, label, reduction='none')\n"
    "                classification_loss = (per_sample * sample_w).sum() / (sample_w.sum() + 1e-8)\n"
    "            else:\n"
    "                classification_loss = torch.mean(self.loss(output, label))\n"
    "            loss = classification_loss"
)
assert old in src, "anchor missing — already patched?"
src = src.replace(old, new, 1)
PATH.write_text(src, encoding="utf-8")
print("patched main.py: per-sample CE reweighting via model.latest_sample_weights")
