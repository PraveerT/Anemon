#!/bin/bash
# Sweep β for v9_clean. Re-uses config_v9_clean.yaml, overriding clip_reweight_beta.
cd /notebooks/PMamba/experiments
set -e
for BETA in 0.5 1.5 2.0; do
  TAG="v9_clean_b${BETA//./_}"
  WD="./work_dir/depth_branch/${TAG}"
  CFG="depth_branch/config_${TAG}.yaml"
  # Generate config from base, swap beta
  python -c "
import yaml, sys
with open('depth_branch/config_v9_clean.yaml') as f:
    c = yaml.safe_load(f)
c['model_args']['clip_reweight_beta'] = ${BETA}
c['work_dir'] = '${WD}/'
with open('${CFG}', 'w') as f:
    yaml.safe_dump(c, f, sort_keys=False)
print('wrote', '${CFG}', 'beta=${BETA}')
"
  rm -rf "${WD}"
  python main.py --config "${CFG}" --work-dir "${WD}" --num-epoch 140 --device 0 > "work_dir/${TAG}_launch.log" 2>&1
  echo "=== ${TAG} done ==="
done
echo "sweep done"
