#!/bin/bash
cd /notebooks/Anemon/experiments
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
run_one(){
  local mode=$1
  local WD=work_dir/cn_skewtcc_$mode
  mkdir -p $WD
  for a in $(seq 1 15); do
    grep -q 'Training epoch: 150' $WD/run.log 2>/dev/null && { echo "$mode ALLDONE"; return; }
    python - "$mode" "$WD" <<'PY'
import sys, glob, re, yaml
mode, wd = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open('cn_skewtcc_%s.yaml' % mode))
cks = glob.glob(wd + '/epoch*_model.pt')
if cks:
    ck = max(cks, key=lambda p: int(re.search(r'epoch(\d+)_', p).group(1)))
    cfg['weights'] = ck; cfg['resume'] = True
yaml.dump(cfg, open(wd + '/_cfg.yaml', 'w'))
PY
    python -u main.py --config $WD/_cfg.yaml >> $WD/run.log 2>&1
    sleep 3
  done
}
run_one skew     # sequential: full GPU each, no OOM contention
run_one sym
run_one random
echo ALL_SKEWTCC_DONE
