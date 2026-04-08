# Training Commands

Run all commands from:

```bash
cd /notebooks/PMamba/experiments
```

## Train Branch 1 (Motion temporal branch)

```bash
python main.py \
  --config pointlstm.yaml \
  --work-dir ./work_dir/motion_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Standalone baseline, 71.7842)

```bash
python main.py \
  --config linear_branch_stacked_quat_weighted_rmsmerge.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_weighted_rms_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Previous best, low-dropout attention readout, 74.8963)

```bash
python main.py \
  --config linear_branch_stacked_quat_weighted_attreadout_rmsmerge_drop005.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_weighted_attreadout_rms_drop005_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train Branch 2 (Current best, dual quaternion merge, 77.1784)

```bash
python main.py \
  --config linear_branch_stacked_quat_dualmerge_weighted_attreadout_rmsmerge_drop005.yaml \
  --work-dir ./work_dir/linear_branch_edgeconv_quatstack_dualmerge_weighted_attreadout_rms_drop005_h256_e120 \
  --num-epoch 120 \
  --device 0
```

## Train QCC In Isolation (from branch winner, parity path)

```bash
python main.py \
  --config quaternion_corr_cycle_parity_from110.yaml \
  --work-dir ./work_dir/quaternion_qcc_isolated_parity_from110 \
  --num-epoch 140 \
  --device 0
```

## Train QCC On Quaternion Branch (from fresh control checkpoint)

```bash
python main.py \
  --config quaternion_branch_qcc_from110.yaml \
  --work-dir ./work_dir/quaternion_branch_qcc_from110 \
  --device 0
```

## Notes

- `epoch120_model.pt` is saved because the configs use `save_interval: 5`.
- Evaluation now runs every 10 epochs before epoch 100, and every epoch from epoch 100 onward.
- Keep QCC experiments isolated from `quaternion_branch` until they beat the standalone winner.
