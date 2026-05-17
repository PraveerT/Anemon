#!/bin/bash
# Wait for BDN buf=1 to finish, then launch BDN-Q.
cd /notebooks/PMamba/experiments
echo "[queue] waiting for BDN buf=1 to finish..."
while ps -ef | grep -v grep | grep -q 'main.py --config pmamba_baseline_bdn_buf1.yaml'; do
  sleep 60
done
echo "[queue] buf=1 done at $(date), launching BDN-Q..."
nohup python main.py --config pmamba_baseline_bdnq.yaml > work_dir/bdnq_train.log 2>&1 &
echo "[queue] launched pid=$!"
