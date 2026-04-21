"""Patch main.py _prepare_state_dict to partial-copy when only dim 1 differs.

Current: shape mismatches are skipped entirely (skipped params reinit from
scratch). We want: when source is (D, K_old, ...) and target is (D, K_new, ...),
copy source into target[:, :K_old, ...] and leave the rest at init (zero).

Applies to rigidity_proj which grows channels across variants:
  v8a: rigidity_proj[0].weight shape (256, 1, 1)
  v13: rigidity_proj[0].weight shape (256, 2, 1)  (cycle_rigidity)
  v11b: rigidity_proj[0].weight shape (256, 3, 1) (multiscale)
"""
from pathlib import Path

PATH = Path('main.py')
src = PATH.read_text(encoding='utf-8')

old_block = """            if target_state[actual_target_key].shape != tensor.shape:
                shape_mismatches.append(
                    (source_key, actual_target_key, tuple(tensor.shape), tuple(target_state[actual_target_key].shape))
                )
                continue

            prepared_state[actual_target_key] = tensor"""

new_block = """            if target_state[actual_target_key].shape != tensor.shape:
                src_shape = tuple(tensor.shape)
                tgt_shape = tuple(target_state[actual_target_key].shape)
                # Partial-copy if only dim 1 differs and target is wider
                if (len(src_shape) == len(tgt_shape)
                        and len(src_shape) >= 2
                        and src_shape[0] == tgt_shape[0]
                        and src_shape[1] < tgt_shape[1]
                        and src_shape[2:] == tgt_shape[2:]):
                    merged = target_state[actual_target_key].clone()
                    merged[:, :src_shape[1]] = tensor
                    prepared_state[actual_target_key] = merged
                    shape_mismatches.append(
                        (source_key, actual_target_key, src_shape, tgt_shape)
                    )
                    continue
                shape_mismatches.append(
                    (source_key, actual_target_key, src_shape, tgt_shape)
                )
                continue

            prepared_state[actual_target_key] = tensor"""

if old_block not in src:
    raise SystemExit('ERR: target block not found')
src = src.replace(old_block, new_block, 1)

# Also update the shape_mismatches in strict-load failure check: we should not
# fail when partial-copy succeeded. Filter out mismatches that landed in
# prepared_state.
# For now, main.py's strict_load=False path accepts mismatches fine — they're
# just logged. So no further change needed.

PATH.write_text(src, encoding='utf-8')
print('OK: patched _prepare_state_dict to partial-copy channel-expanded params')
