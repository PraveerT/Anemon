"""Prepare a pmamba-warm-start PMambaTopsMotion init checkpoint.

  1. Build PMambaTopsMotion (fresh random init; stage1 has 7-ch input).
  2. Load pmamba_branch/epoch110_model.pt (all layers except stage1 match).
  3. For stage1's first Conv (shape [32, 7, 1, 1]):
     - copy pmamba's 4-ch weights into the first 4 input columns
     - zero the last 3 columns (tops)
  4. Save the resulting state dict to
     work_dir/pmamba_tops_v27b/pmamba_tops_warmstart_init.pt

Result: at init, stage1(xyz + tops) == pmamba_stage1(xyz) exactly. Everything
else is pmamba's trained weights. Training from this init guarantees we
inherit pmamba's knowledge and can only improve by learning tops signal.
"""
import sys, os
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
from models.motion import PMambaTopsMotion

PMAMBA_CKPT = 'work_dir/pmamba_branch/epoch110_model.pt'
OUT_DIR = 'work_dir/pmamba_tops_v27b'
OUT_CKPT = f'{OUT_DIR}/pmamba_tops_warmstart_init.pt'

os.makedirs(OUT_DIR, exist_ok=True)

# Build fresh model (random init).
model = PMambaTopsMotion(num_classes=25, pts_size=96, knn=[32, 24, 48, 24], topk=8)
print(f"PMambaTopsMotion built. stage1 first Conv shape: {model.stage1.layer_list[0][0].weight.shape}")

# Load pmamba_branch weights.
ckpt = torch.load(PMAMBA_CKPT, map_location='cpu')
pmamba_state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))

# Identify stage1 first-conv weights in pmamba (shape [32, 4, ...]).
pmamba_stage1_key = None
for k in pmamba_state:
    if 'stage1.layer_list.0.0.weight' in k:   # nn.Sequential wrapper's first child = Conv2d
        pmamba_stage1_key = k
        break
if pmamba_stage1_key is None:
    # Fallback: find any stage1.layer_list.0 weight with 4 input channels
    for k in pmamba_state:
        if k.startswith('stage1.layer_list.0.') and 'weight' in k:
            if pmamba_state[k].dim() == 4 and pmamba_state[k].shape[1] == 4:
                pmamba_stage1_key = k
                break

assert pmamba_stage1_key is not None, "Could not find pmamba stage1 first-conv weight"
print(f"pmamba stage1 key: {pmamba_stage1_key}, shape: {pmamba_state[pmamba_stage1_key].shape}")

# Custom handling: expand 4-ch weight to 7-ch with zero padding on tops columns.
pmamba_w4 = pmamba_state[pmamba_stage1_key]   # (32, 4, 1, 1)
target_key = pmamba_stage1_key
target_w7 = model.state_dict()[target_key].clone()   # (32, 7, 1, 1) random
target_w7[:, :4] = pmamba_w4
target_w7[:, 4:] = 0.0
pmamba_state[target_key] = target_w7

# Also handle any bias (shouldn't depend on input channels, should match).
print(f"Remapped stage1 weight: first 4 cols = pmamba, last 3 cols = 0")

# Load into model (now state dict has our remapped stage1).
missing, unexpected = model.load_state_dict(pmamba_state, strict=False)
print(f"Load result: missing={len(missing)}, unexpected={len(unexpected)}")
if missing:
    print(f"  missing (first 5): {missing[:5]}")
if unexpected:
    print(f"  unexpected (first 5): {unexpected[:5]}")

# Sanity check: verify that tops columns in stage1 first conv are exactly zero
w = model.stage1.layer_list[0][0].weight.detach()
tops_cols_norm = w[:, 4:].abs().sum().item()
xyz_cols_norm = w[:, :4].abs().sum().item()
print(f"Stage1 weight: tops cols |w|_1 = {tops_cols_norm:.6f} (should be 0)")
print(f"Stage1 weight: xyz+t cols |w|_1 = {xyz_cols_norm:.3f} (copied from pmamba)")

torch.save({'model_state_dict': model.state_dict(), 'epoch': 0}, OUT_CKPT)
print(f"Saved init checkpoint: {OUT_CKPT}")
