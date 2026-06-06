#!/bin/bash
# LogitNorm tau sweep on lean base: t02/t04/t10, sequential, non-deterministic.
cd /notebooks/Manta/experiments || exit 1
export DENSE=0 GPU_RESIDENT=1 DETERMINISTIC=0

for tag in t02 t04 t10; do
  wd="./work_dir/logitnorm_${tag}"
  mkdir -p "$wd"
  n=$(grep -c 'Overall Accuracy' "$wd/log.txt" 2>/dev/null); n=${n:-0}
  if [ "$n" -ge 145 ]; then echo "[$tag] done ($n)"; continue; fi
  echo "[$tag] launching"
  python -u main.py --config "logitnorm_${tag}.yaml" > "$wd/stdout.log" 2>&1
  echo "[$tag] rc=$?"
done
echo ALL_DONE
