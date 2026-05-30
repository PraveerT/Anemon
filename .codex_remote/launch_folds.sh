#!/bin/bash
cd /notebooks/Anemon/experiments
# FG83 fold (depth r2plus1d, fg-crop) -> dumps valid(=218 OOF) logits
nohup python -u /notebooks/Anemon/train_depth_small.py \
  --split-root /notebooks/cvpr_data/dataset_splits_fold \
  --cache-dir /notebooks/Anemon/dataset/Nvidia/Processed/fg83_fold_cache \
  --workdir /notebooks/Anemon/experiments/work_dir/fg83_fold \
  --arch r2plus1d_18 --input-mode raw3_kinetics --pretrained --fg-crop \
  --epochs 70 --batch-size 4 --frames 32 --lr 3e-4 --backbone-lr 2e-5 --stop-at 95 \
  > work_dir/fg83_fold.log 2>&1 &
echo "fg83_fold pid=$!"
# RGB fold (color fg-crop r2plus1d) -> dumps valid(=218 OOF) logits
export SPLITS=/notebooks/cvpr_data/dataset_splits_fold
export RGB_CACHE=/notebooks/Anemon/dataset/Nvidia/Processed/rgb_fold_cache
export WD=/notebooks/Anemon/experiments/work_dir/rgb_fold
nohup python -u train_rgb_color.py --arch r2plus1d_18 --epochs 60 --bs 6 --frames 40 \
  > work_dir/rgb_fold.log 2>&1 &
echo "rgb_fold pid=$!"
