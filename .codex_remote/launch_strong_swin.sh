#!/bin/bash
cd /notebooks/Anemon/experiments
export WD=/notebooks/Anemon/experiments/work_dir/rgb_swin224up
nohup python -u train_rgb_color.py --arch swin3d_t --epochs 40 --bs 2 --frames 32 \
  --cache-size 128 --crop 112 --resize 224 --lr 2e-4 --blr 1e-5 \
  > work_dir/rgb_swin224up.log 2>&1 &
echo "swin224up pid=$!"
