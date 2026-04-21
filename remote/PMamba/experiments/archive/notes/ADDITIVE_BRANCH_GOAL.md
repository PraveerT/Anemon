# Additive Branch 2 Goal

Keep a single standalone branch-2 reference alongside branch 1 (`Motion`).

## Reference Branch-2 Run

- model: `models.reqnn_motion.EdgeConvQuaternionStackedDualMergeWeightedRMSAttentionReadoutMotion`
- config: `linear_branch_stacked_quat_dualmerge_weighted_attreadout_rmsmerge_drop005.yaml`
- work dir: `work_dir/linear_branch_edgeconv_quatstack_dualmerge_weighted_attreadout_rms_drop005_h256_e120/`
- best observed test accuracy so far: `77.1784%` at epoch `112`

## Kept Design

1. Keep the DGCNN-style EdgeConv neighborhood block.
2. Keep the quaternion point mixer and the extra quaternion refinement stage before collapse.
3. Collapse each quaternion with both weighted RMS magnitude and a real-part summary before projection.
4. Use attention-pooled readout on top of the stacked winner.
5. Keep the branch standalone and evaluate it on its own before any fusion work.

## Run Command

```bash
cd /notebooks/PMamba/experiments
python main.py \
  --config linear_branch_stacked_quat_dualmerge_weighted_attreadout_rmsmerge_drop005.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_dualmerge_weighted_attreadout_rms_drop005_h256_e120 \
  --num-epoch 120 \
  --device 0
```
