"""Patch main.py to apply ShuffleMix+ on point cloud inputs.

Inserts:
1. shufflemix_pc helper function at top of main.py (after imports)
2. Augmentation call right after image+label are on device
3. Mixed label-loss computation replacing single classification_loss

Idempotent: detects existing patch by sentinel comment.
"""
import re
import io
from pathlib import Path

MAIN = Path("/notebooks/PMamba/experiments/main.py")
SENTINEL = "# === ShuffleMix+ patch ==="

text = MAIN.read_text(encoding="utf-8")

if SENTINEL in text:
    print("ShuffleMix+ patch already applied; skipping.")
    raise SystemExit(0)

# 1. Add shufflemix_pc helper before the first 'class ' definition
helper = f'''

{SENTINEL}
def shufflemix_pc(image, label, smprob):
    """ShuffleMix+ for point cloud (B, T, P, C).

    For ``smprob`` fraction of frames per clip, replace those frames with the
    corresponding frames from a flipped-batch sample. Returns the mixed input
    and the two label sets plus a mixing weight ``lam`` for label-side mixing.
    """
    if smprob <= 0.0 or image.dim() < 3:
        return image, label, label, 1.0
    import random as _random
    B = image.size(0)
    T = image.size(1)
    label_b = label.flip(0)
    if (label_b == label).all():
        return image, label, label, 1.0
    n_replace = max(1, int(round(smprob * T)))
    idx = _random.sample(range(T), n_replace)
    image_b = image.flip(0).clone()
    image[:, idx] = image_b[:, idx]
    lam = 1.0 - n_replace / T
    return image, label, label_b, lam
# === end ShuffleMix+ patch ===
'''

# Insert after imports / before first class definition
class_match = re.search(r"\nclass\s+\w+", text)
insert_pos = class_match.start() if class_match else 0
text = text[:insert_pos] + helper + text[insert_pos:]

# 2. Replace the classification_loss computation block with shufflemix-aware one
old = (
    "            image = self.device.data_to_device(data[0])\n"
    "            label = self.device.data_to_device(data[1])\n"
)
if old not in text:
    raise RuntimeError("Could not find data_to_device block to patch")
new = (
    "            image = self.device.data_to_device(data[0])\n"
    "            label = self.device.data_to_device(data[1])\n"
    "            sm_smprob = float(getattr(self.arg, 'shufflemix_smprob', 0.0))\n"
    "            image, _label_a, _label_b, _sm_lam = shufflemix_pc(image, label, sm_smprob)\n"
)
text = text.replace(old, new, 1)

# 3. Replace classification_loss line to mix labels via _sm_lam
old_loss = (
    "                classification_loss = torch.mean(self.loss(output, label))\n"
)
if old_loss not in text:
    raise RuntimeError("Could not find classification_loss line to patch")
new_loss = (
    "                if _sm_lam < 1.0:\n"
    "                    loss_a = torch.mean(self.loss(output, _label_a))\n"
    "                    loss_b = torch.mean(self.loss(output, _label_b))\n"
    "                    classification_loss = _sm_lam * loss_a + (1.0 - _sm_lam) * loss_b\n"
    "                else:\n"
    "                    classification_loss = torch.mean(self.loss(output, label))\n"
)
text = text.replace(old_loss, new_loss, 1)

MAIN.write_text(text, encoding="utf-8")
print("Applied ShuffleMix+ patch to main.py")
