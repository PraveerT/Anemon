#!/bin/bash
# Q-rot-cycle mechanism battery: controls + amplification scouts.
# All runs share the best-run hyperparams; only the rot config / seed varies.
cd /notebooks/Anemon
ROOT=experiments/work_dir/qexp
mkdir -p $ROOT

COMMON="--frames 16 --points 128 --epochs 260 --batch-size 64 --workers 0 \
--lr 0.00075 --min-lr 0.000015 --warmup-epochs 10 --wd 0.04 --ema-decay 0.995 \
--label-smoothing 0.08 --qcc-weight 0.0 --cycle-weight 0.0 --rot-aug-ce-weight 0.0 \
--rot-cycle-prob 1.0 --dropout 0.30 --point-hidden 160 --temporal-hidden 256 \
--layers 2 --jitter 0.006 --point-drop 0.08 --no-quat-inject --no-publish-active"

run () {
  local name="$1"; shift
  local wd="$ROOT/$name"
  mkdir -p "$wd"
  nohup python -u train_corr_qcc_fusion.py --workdir "$wd" --seed "$SEED" $COMMON "$@" \
    > "$wd/run.log" 2>&1 &
  echo "launched $name (pid $!)"
}

# ---- Battery 1: controls across the 5 validation seeds ----
for SEED in 29 31 37 43 47; do
  run "clean_s${SEED}"  --rot-cycle-weight 0.0
  run "qrot_s${SEED}"   --rot-cycle-weight 0.02 --rot-mode uniform
  run "ident_s${SEED}"  --rot-cycle-weight 0.02 --rot-mode z --rot-max-angle-deg 0
done

# ---- Battery 2: amplification scouts (seed 29 only) ----
SEED=29
run "amp_w004"    --rot-cycle-weight 0.04 --rot-mode uniform
run "amp_w008"    --rot-cycle-weight 0.08 --rot-mode uniform
run "amp_w016"    --rot-cycle-weight 0.16 --rot-mode uniform
run "amp_so3_45"  --rot-cycle-weight 0.02 --rot-mode small-so3 --rot-max-angle-deg 45
run "amp_z60"     --rot-cycle-weight 0.02 --rot-mode z --rot-max-angle-deg 60

echo "ALL LAUNCHED"
wait
echo "ALL DONE"
