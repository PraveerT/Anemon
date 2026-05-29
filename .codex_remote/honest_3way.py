"""Definitive honest non-DSN ensemble: CN-XXL + RGB + FG83(depth).
Weights/temps chosen on a held-out 30% of TRAIN (seed 0); test untouched.
Also reports the test-tuned ORACLE as the hard ceiling.
"""
import numpy as np, re, os, itertools

W = '/notebooks/Anemon/experiments/work_dir/'
P = {
    'cnxxl': (W+'cn_xxl_quat_head/train_logits.npz', W+'cn_xxl_quat_head/test_logits.npz'),
    'rgb':   (W+'rgb_color_r2p1d/train_logits.npz',  W+'rgb_color_r2p1d/best_logits.npz'),
    'fg83':  (W+'depth_small_r2_fg83_restored_20260528_033028/train_logits.npz',
              W+'depth_small_r2_fg83_restored_20260528_033028/test_logits.npz'),
}


def sig_of(s):
    s=str(s); m=re.search(r'class_(\d+)/subject(\d+)_r(\d+)',s)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}' if m else s


def load(p, ref=None):
    d=np.load(p, allow_pickle=True)
    lg = d['logits'] if 'logits' in d.files else d['pred_logits']
    y=d['labels']
    sg=np.array([sig_of(s) for s in d['sigs']]) if 'sigs' in d.files else None
    if ref is not None and sg is not None:
        by={s:i for i,s in enumerate(sg)}; idx=np.array([by[s] for s in ref]); lg,y=lg[idx],y[idx]
    return lg.astype(float), y, sg


def lsm(z,T=1.0):
    z=z/T; z=z-z.max(1,keepdims=True); return z-np.log(np.exp(z).sum(1,keepdims=True))


# ---- TEST (align to cnxxl test sigs) ----
cnte,yte,ref = load(P['cnxxl'][1])
rgte,_,_ = load(P['rgb'][1], ref)
fgte,_,_ = load(P['fg83'][1], ref)
# ---- TRAIN (align to cnxxl train sigs) ----
cntr,ytr,reftr = load(P['cnxxl'][0])
rgtr,_,_ = load(P['rgb'][0], reftr)
fgtr,_,_ = load(P['fg83'][0], reftr)

def acc(z,y): return (z.argmax(1)==y).mean()*100
print(f'solo test: cnxxl {acc(cnte,yte):.2f}  rgb {acc(rgte,yte):.2f}  fg83 {acc(fgte,yte):.2f}')

# held-out calibration slice of TRAIN
rng=np.random.RandomState(0); perm=rng.permutation(len(ytr)); cal=perm[:int(0.3*len(ytr))]
Ws=[0.0,0.1,0.2,0.3,0.5,0.75,1.0]
Ts=[1.0,1.5,2.0,3.0]
cn_c=lsm(cntr[cal])

best=None
for tr_ in Ts:
  rg_c=lsm(rgtr[cal],tr_)
  for tf in Ts:
    fg_c=lsm(fgtr[cal],tf)
    for w1 in Ws:
      for w2 in Ws:
        a=acc(cn_c + w1*rg_c + w2*fg_c, ytr[cal])
        if best is None or a>best[0]: best=(a,tr_,tf,w1,w2)
_,tr_,tf,w1,w2=best
fused = lsm(cnte) + w1*lsm(rgte,tr_) + w2*lsm(fgte,tf)
print(f'\nHONEST (cal on train-holdout): rgbT={tr_} fgT={tf} w_rgb={w1} w_fg={w2}')
print(f'  -> TEST {acc(fused,yte):.3f}  ({(fused.argmax(1)==yte).sum()}/{len(yte)})   [cnxxl alone 91.29]')

# ORACLE ceiling (test-tuned, NOT honest)
best=0;arg=None
for tr_ in Ts:
  rg=lsm(rgte,tr_)
  for tf in Ts:
    fg=lsm(fgte,tf)
    for w1 in np.arange(0,1.01,0.05):
      for w2 in np.arange(0,1.01,0.05):
        a=acc(lsm(cnte)+w1*rg+w2*fg,yte)
        if a>best: best,arg=a,(round(w1,2),round(w2,2),tr_,tf)
print(f'\nORACLE (test-tuned ceiling): {best:.3f}  @ w_rgb={arg[0]} w_fg={arg[1]} rgbT={arg[2]} fgT={arg[3]}')
