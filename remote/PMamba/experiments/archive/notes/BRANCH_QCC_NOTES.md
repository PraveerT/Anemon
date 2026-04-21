# Quaternion Branch QCC Notes

- Control rerun in `work_dir/quaternion_branch` peaked at `77.18%` on epoch `112`.
- The nearest saved checkpoint before that peak is `epoch110_model.pt`; use that for QCC initialization.
- Do not use `resume: true` when switching from the winner model to the QCC model. The optimizer state does not match the expanded parameter set.
- Keep the QCC rerun in a separate work dir so the clean control remains reproducible.
