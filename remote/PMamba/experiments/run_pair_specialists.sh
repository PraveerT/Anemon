#!/bin/bash
# Run Cfbq specialists for the 4 worst confusion pairs.
# For each pair: filter train+test lists, train PMamba+Cfbq from ep110, save preds.

set -e
cd /notebooks/PMamba/experiments

PAIRS=(
    "3-16:03|16"
    "18-9:18|09"
    "5-4:05|04"
    "1-0:01|00"
)

# Backup current lists if not already
[ -f /notebooks/PMamba/dataset/Nvidia/Processed/train_depth_list_full2x.txt ] || \
    cp /notebooks/PMamba/dataset/Nvidia/Processed/train_depth_list.txt /notebooks/PMamba/dataset/Nvidia/Processed/train_depth_list_full2x.txt
[ -f /notebooks/PMamba/dataset/Nvidia/Processed/test_depth_list_full.txt ] || \
    cp /notebooks/PMamba/dataset/Nvidia/Processed/test_depth_list.txt /notebooks/PMamba/dataset/Nvidia/Processed/test_depth_list_full.txt

for entry in "${PAIRS[@]}"; do
    PAIR="${entry%%:*}"
    REGEX="${entry##*:}"
    NAME="pmamba_cfbq_pair${PAIR//-/_}"
    WORKDIR="./work_dir/${NAME}"

    echo "================================"
    echo "PAIR ${PAIR}  regex _label_(${REGEX})"
    echo "================================"

    # filter
    grep -E "_label_(${REGEX})" /notebooks/PMamba/dataset/Nvidia/Processed/train_depth_list_full2x.txt \
        > /notebooks/PMamba/dataset/Nvidia/Processed/train_depth_list.txt
    grep -E "_label_(${REGEX})" /notebooks/PMamba/dataset/Nvidia/Processed/test_depth_list_full.txt \
        > /notebooks/PMamba/dataset/Nvidia/Processed/test_depth_list.txt
    echo "  train clips: $(wc -l < /notebooks/PMamba/dataset/Nvidia/Processed/train_depth_list.txt)"
    echo "  test clips : $(wc -l < /notebooks/PMamba/dataset/Nvidia/Processed/test_depth_list.txt)"

    # write yaml with unique workdir
    cat > /tmp/${NAME}.yaml <<EOF
dataloader: nvidia_dataloader.NvidiaLoader
phase: train
num_epoch: 160
work_dir: ${WORKDIR}/
batch_size: 8
test_batch_size: 1
num_worker: 8
device: 0
log_interval: 50
eval_interval: 1
save_interval: 5
framesize: 32
pts_size: 256
dynamic_pts_size: true
pts_random_range: [48, 256]
strict_load: false

weights: ./work_dir/pmamba_branch/epoch110_model.pt

train_loader_args: {phase: train, framerate: 32}
test_loader_args:  {phase: test,  framerate: 32}

optimizer_args:
  optimizer: Adam
  base_lr: 0.00012
  step: [10]
  weight_decay: 0.03
  start_epoch: 110
  nesterov: false

model: models.motion.MotionCfbq
model_args:
  pts_size: 256
  num_classes: 25
  knn: [32, 24, 48, 24]
  topk: 8
EOF

    rm -rf "${WORKDIR}"
    python -u main.py --config /tmp/${NAME}.yaml 2>&1 | tee /tmp/${NAME}.log | grep -E 'Test, Evaluation|Mean training' | tail -5
    echo "  done -> ${WORKDIR}"
done

# restore
cp /notebooks/PMamba/dataset/Nvidia/Processed/train_depth_list_full2x.txt /notebooks/PMamba/dataset/Nvidia/Processed/train_depth_list.txt
cp /notebooks/PMamba/dataset/Nvidia/Processed/test_depth_list_full.txt /notebooks/PMamba/dataset/Nvidia/Processed/test_depth_list.txt
echo "all 4 specialists done; lists restored"
