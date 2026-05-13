#!/bin/bash
# Autonomous post-AttRD-v2 pipeline.
set -e
PID=$1
cd /notebooks/PMamba/experiments

echo "Waiting for pid $PID..."
until ! ps -p "$PID" > /dev/null 2>&1; do sleep 60; done
echo "PID $PID done."

# Find train-best epoch
python - <<'PY'
import re, os
log = 'work_dir/pmamba_baseline_attrd_v2/log.txt'
if not os.path.exists(log):
    print('NO_LOG'); raise SystemExit
ep_acc = []; cur = None
for line in open(log):
    m = re.search(r'Training epoch:\s+(\d+)', line)
    if m: cur = int(m.group(1)); continue
    m = re.search(r'Mean training acc:\s+([\d.]+)%', line)
    if m and cur: ep_acc.append((cur, float(m.group(1))))
ep_acc.sort(key=lambda x:-x[1])
print('top5_train:', ep_acc[:5])
with open('/tmp/train_best_ep.txt','w') as f: f.write(str(ep_acc[0][0]))
PY

BEST_EP=$(cat /tmp/train_best_ep.txt)
echo "Train-best epoch: $BEST_EP"

# Dump both train-best and ep120
python - <<PY
import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_attrd_v2 import MotionAttRDv2

def dump(ep):
    ckpt = f'work_dir/pmamba_baseline_attrd_v2/epoch{ep}_model.pt'
    if not os.path.exists(ckpt):
        # fallback to nearest multiple of 5
        for delta in range(1, 5):
            for sign in [-1, 1]:
                e = ep + sign*delta
                cp = f'work_dir/pmamba_baseline_attrd_v2/epoch{e}_model.pt'
                if os.path.exists(cp):
                    ckpt = cp; ep = e
                    print(f'fallback to ep{e}')
                    break
            if os.path.exists(ckpt): break
    ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
    loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    m = MotionAttRDv2(num_classes=25, pts_size=256, knn=[32,24,48,24],
                     topk=8, multi_scale_num_scales=5,
                     av2_hidden_dim=128, av2_num_layers=2, av2_num_heads=4,
                     av2_n_q=4, av2_n_v=8, av2_d_read=64, av2_dropout=0.3,
                     av2_bidirectional=True).cuda()
    state = torch.load(ckpt, map_location='cpu')['model_state_dict']
    m.load_state_dict(state, strict=False); m.eval()
    pl, ll = [], []
    with torch.no_grad():
        for b in loader:
            x, y = b[0].cuda().float(), b[1]
            pl.append(torch.softmax(m(x), -1).cpu().numpy())
            ll.append(y.numpy() if hasattr(y, 'numpy') else np.array(y))
    P = np.concatenate(pl); L = np.concatenate(ll)
    out = f'dump_probs_runs/attrd_v2_ep{ep}.npz'
    np.savez(out, probs=P, labels=L)
    acc = (P.argmax(1)==L).mean()*100
    print(f'  AttRD-v2 ep{ep} solo = {acc:.2f}% -> {out}')
    return out, acc

dump_tb = dump($BEST_EP)
dump_120 = dump(120)
PY

# Fusion analysis
python - <<'PY'
import os, itertools, numpy as np
os.chdir('/notebooks/PMamba/experiments')
def load_dsn(p, t=9.5):
    z = np.load(p); L = z['logits']*t; L=L-L.max(1,keepdims=True); e=np.exp(L); return e/e.sum(1,keepdims=True), z['labels']
def load_p(p):
    z = np.load(p); return z['probs'], z['labels']

import glob
M = {}
M['DSN'], L = load_dsn('dump_probs_runs/cvpr_dsn_K_depth.npz')
M['RD'], _ = load_p('dump_probs_runs/realdeltanet_ep118.npz')
M['BRD'], _ = load_p('dump_probs_runs/brd_ep112.npz')
M['AttRD'], _ = load_p('dump_probs_runs/attrd_ep120.npz')

# Find AttRD-v2 dumps
v2_files = sorted(glob.glob('dump_probs_runs/attrd_v2_ep*.npz'))
for fp in v2_files:
    name = 'AttRDv2_' + fp.split('attrd_v2_')[-1].replace('.npz','')
    M[name], _ = load_p(fp)

names = list(M.keys())
print(f'=== Solos ===')
for n in names:
    a = (M[n].argmax(1) == L).mean()*100
    print(f'  {n:18s} {a:.2f}%')

res = []
print('\n=== All combos including DSN ===')
non_dsn = [n for n in names if n != 'DSN']
for k in range(1, len(non_dsn)+1):
    for c in itertools.combinations(non_dsn, k):
        ns = ['DSN'] + list(c)
        avg = np.mean([M[n] for n in ns], 0)
        acc = (avg.argmax(1) == L).mean()*100
        res.append((acc, ' + '.join(ns)))
res.sort(reverse=True)
for a, n in res[:30]:
    print(f'  {a:.2f}%   {n}')
PY
