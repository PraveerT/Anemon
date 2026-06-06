#!/bin/bash
# Autonomous cycle-consistency chain: 7 augmented-view consistency variants,
# trained sequentially on the lean 1.06M base (seed 0). Non-deterministic for
# speed (~30 min/run). Goal: beat 90.
cd /notebooks/Manta/experiments || exit 1
export DENSE=0 GPU_RESIDENT=1 DETERMINISTIC=0

for md in rot refl treverse jitter scale pdrop combo; do
  cfg="cyc_${md}"
  wd="./work_dir/cyc_${md}_s0"
  mkdir -p "$wd"
  n=$(grep -c 'Overall Accuracy' "$wd/log.txt" 2>/dev/null)
  n=${n:-0}
  if [ "$n" -ge 150 ]; then
    echo "[$md] already complete ($n evals)"; continue
  fi
  echo "[$md] launching -> $wd"
  python -u main.py --config "${cfg}.yaml" > "$wd/stdout.log" 2>&1
  echo "[$md] finished rc=$?"
done
echo ALL_DONE
