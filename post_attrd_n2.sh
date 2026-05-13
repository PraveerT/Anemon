#!/bin/bash
set -e
PID=$1
cd /notebooks/PMamba/experiments

echo "Waiting for pid $PID..."
until ! ps -p "$PID" > /dev/null 2>&1; do sleep 60; done
echo "PID $PID done."

# Find train-best epoch
python - <<'PY'
import re, os
log = 'work_dir/pmamba_dtw_attrd/log.txt'
ep_acc = []; cur = None
for line in open(log):
    m = re.search(r'Training epoch:\s+(\d+)', line)
    if m: cur = int(m.group(1)); continue
    m = re.search(r'Mean training acc:\s+([\d.]+)%', line)
    if m and cur: ep_acc.append((cur, float(m.group(1))))
ep_acc.sort(key=lambda x:-x[1])
print('top5_train:', ep_acc[:5])
with open('/tmp/n2_attrd_train_best.txt','w') as f: f.write(str(ep_acc[0][0]))
PY

BEST_EP=$(cat /tmp/n2_attrd_train_best.txt)
echo "AttRD(N2) train-best: $BEST_EP"

# Dump train-best + ep120
python - <<PY
import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_attrd import MotionAttRD

def dump(ep):
    ckpt = f'work_dir/pmamba_dtw_attrd/epoch{ep}_model.pt'
    if not os.path.exists(ckpt):
        for delta in range(1, 6):
            for sign in [-1, 1]:
                cp = f'work_dir/pmamba_dtw_attrd/epoch{ep+sign*delta}_model.pt'
                if os.path.exists(cp):
                    ckpt = cp; ep = ep+sign*delta
                    print(f'fallback ep{ep}')
                    break
            if os.path.exists(ckpt): break
    ds = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='test')
    loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    m = MotionAttRD(num_classes=25, pts_size=256, knn=[32,24,48,24],
                    topk=8, multi_scale_num_scales=5,
                    ar_hidden_dim=128, ar_num_layers=2, ar_num_heads=4,
                    ar_n_q=4, ar_n_v=8, ar_d_read=32, ar_dropout=0.3,
                    ar_bidirectional=True).cuda()
    state = torch.load(ckpt, map_location='cpu')['model_state_dict']
    m.load_state_dict(state, strict=False); m.eval()
    pl, ll = [], []
    with torch.no_grad():
        for b in loader:
            x, y = b[0].cuda().float(), b[1]
            pl.append(torch.softmax(m(x), -1).cpu().numpy())
            ll.append(y.numpy() if hasattr(y, 'numpy') else np.array(y))
    P = np.concatenate(pl); L = np.concatenate(ll)
    out = f'dump_probs_runs/attrd_n2_ep{ep}.npz'
    np.savez(out, probs=P, labels=L)
    print(f'  AttRD(N2) ep{ep} solo = {(P.argmax(1)==L).mean()*100:.2f}% -> {out}')

dump($BEST_EP)
dump(120)
PY

# Fusion analysis
python - <<'PY'
import os, itertools, glob, numpy as np
os.chdir('/notebooks/PMamba/experiments')
def load_dsn(p, t=9.5):
    z = np.load(p); L = z['logits']*t; L=L-L.max(1,keepdims=True); e=np.exp(L); return e/e.sum(1,keepdims=True), z['labels']
def load_p(p):
    z = np.load(p); return z['probs'], z['labels']

M = {}
M['DSN'], L = load_dsn('dump_probs_runs/cvpr_dsn_K_depth.npz')
M['RD(N1)'], _  = load_p('dump_probs_runs/realdeltanet_ep118.npz')
M['RD(N2)'], _  = load_p('dump_probs_runs/realdeltanet_n2_ep118.npz')
M['BRD'], _     = load_p('dump_probs_runs/brd_ep112.npz')
M['AttRD(N1)'], _ = load_p('dump_probs_runs/attrd_ep120.npz')

for fp in sorted(glob.glob('dump_probs_runs/attrd_n2_ep*.npz')):
    name = 'AttRD(N2)_' + fp.split('attrd_n2_')[-1].replace('.npz','')
    M[name], _ = load_p(fp)

print('=== Solos ===')
for n in M:
    print(f'  {n:20s} {(M[n].argmax(1)==L).mean()*100:.2f}%')

non_dsn = [n for n in M if n != 'DSN']
res = []
for k in range(1, len(non_dsn)+1):
    for c in itertools.combinations(non_dsn, k):
        ns = ['DSN'] + list(c)
        avg = np.mean([M[n] for n in ns], 0)
        res.append(((avg.argmax(1)==L).mean()*100, ' + '.join(ns)))
res.sort(reverse=True)
print('\n=== Top 30 combos with DSN ===')
for a, n in res[:30]:
    print(f'  {a:.2f}%   {n}')
PY
