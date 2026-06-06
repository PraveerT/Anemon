#!/bin/bash
# Temporal-cycle SSL experiment: ordered cycle vs time-shuffled control (seed 0).
cd /notebooks/Manta/experiments || exit 1
export DENSE=0 GPU_RESIDENT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8

run () {
  local cfg=$1 wd=$2
  mkdir -p "$wd"
  if [ "$(grep -c 'Overall Accuracy' "$wd/log.txt" 2>/dev/null)" -ge 150 ]; then
    echo "[$cfg] already complete (150 evals)"; return
  fi
  echo "[$cfg] launching -> $wd"
  python -u main.py --config "$cfg.yaml" > "$wd/stdout.log" 2>&1
  echo "[$cfg] finished rc=$?"
}

run pgcnet_tcycle      ./work_dir/pgcnet_tcycle_s0
run pgcnet_tcycle_shuf ./work_dir/pgcnet_tcycle_shuf_s0
echo ALL_DONE
