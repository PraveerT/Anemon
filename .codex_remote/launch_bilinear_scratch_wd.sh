#!/bin/bash
cd /notebooks/Anemon/experiments
WD=work_dir/cn_xxl_bilinear_scratch
mkdir -p $WD
mk_cfg(){ python - <<'PY'
import glob, re, yaml
wd = 'work_dir/cn_xxl_bilinear_scratch'
cfg = yaml.safe_load(open('cn_bilinear_scratch.yaml'))
cks = glob.glob(wd + '/epoch*_model.pt')
if cks:
    ck = max(cks, key=lambda p: int(re.search(r'epoch(\d+)_', p).group(1)))
    cfg['weights'] = ck; cfg['resume'] = True   # crash-resume from own ckpt
# else fresh from scratch (weights null, resume False) — identical to original
yaml.dump(cfg, open(wd + '/_cfg.yaml', 'w'))
print('launch weights=', cfg['weights'], 'resume=', cfg['resume'], flush=True)
PY
}
for a in $(seq 1 20); do
  grep -q 'Training epoch: 150' $WD/run.log 2>/dev/null && { echo ALLDONE; break; }
  mk_cfg
  python -u main.py --config $WD/_cfg.yaml >> $WD/run.log 2>&1
  sleep 3
done
echo BILINEAR_SCRATCH_DONE
