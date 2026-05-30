#!/bin/bash
# Auto-resuming trainer for the quaternion-bottleneck runs. Survives container
# 503 restarts: on each (re)launch it points `weights` at the latest checkpoint
# (fresh start if none), so progress is never lost.
cd /notebooks/Anemon/experiments

mk_cfg(){ python - "$1" "$2" <<'PY'
import sys, glob, re, yaml
base, wd = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(base))
cks = glob.glob(wd + '/epoch*_model.pt')
if cks:
    ck = max(cks, key=lambda p: int(re.search(r'epoch(\d+)_', p).group(1)))
    cfg['weights'] = ck
    cfg['resume'] = True
else:
    cfg['weights'] = None
    cfg['resume'] = False
yaml.dump(cfg, open(wd + '/_cfg.yaml', 'w'))
PY
}

loop(){ base="$1"; wd="$2";
  for a in $(seq 1 10); do
    grep -q 'Training epoch: 150' "$wd/run.log" 2>/dev/null && { echo "[$base] DONE"; break; }
    mk_cfg "$base" "$wd"
    echo "[$base] launch attempt $a"
    python -u main.py --config "$wd/_cfg.yaml" >> "$wd/run.log" 2>&1
    sleep 3
  done
}

loop quat_bottleneck.yaml work_dir/quat_bottleneck &
loop quat_bottleneck_realctrl.yaml work_dir/quat_bottleneck_realctrl &
wait
echo "ALL_WATCHDOG_DONE"
