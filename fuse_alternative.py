"""Alternative fusion aggregations on existing train-best dumps.

Tries:
- arithmetic mean (baseline, prob avg)
- geometric mean (log-prob avg)
- temperature-scaled prob avg (each model's softmax sharpened/smoothed)
- argmax voting
- max over models (per class)
- power-mean (generalized)
"""
import os, itertools, numpy as np

os.chdir('/notebooks/PMamba/experiments')
D = 'dump_probs_runs'

def load_dsn(p, t=9.5):
    z = np.load(p); L = z['logits']*t; L=L-L.max(1,keepdims=True); e=np.exp(L); return e/e.sum(1,keepdims=True), z['labels']
def load_p(p):
    z = np.load(p); return z['probs'], z['labels']

M = {}
M['DSN'], L = load_dsn(f'{D}/cvpr_dsn_K_depth.npz')
M['RD'],    _ = load_p(f'{D}/realdeltanet_ep118.npz')
M['BRD'],   _ = load_p(f'{D}/brd_ep112.npz')
M['AttRD'], _ = load_p(f'{D}/attrd_ep120.npz')

def acc(p): return (p.argmax(1) == L).mean() * 100

def fuse_arith(ps): return np.mean(ps, axis=0)
def fuse_geom(ps, eps=1e-9):
    log_p = np.mean([np.log(p + eps) for p in ps], axis=0)
    e = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
def fuse_max(ps): return np.max(ps, axis=0)
def fuse_vote(ps):
    preds = np.stack([p.argmax(1) for p in ps], axis=1)  # (N, K)
    res = np.zeros((preds.shape[0], 25))
    for i in range(preds.shape[0]):
        for c in range(25):
            res[i, c] = (preds[i] == c).sum()
    return res
def fuse_tempered(ps, T=2.0):
    """Smooth each prob by T before averaging."""
    sharpened = []
    for p in ps:
        log_p = np.log(p + 1e-9) / T
        e = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        sharpened.append(e / e.sum(axis=1, keepdims=True))
    return np.mean(sharpened, axis=0)
def fuse_power(ps, p=0.5):
    """Power mean (M_p = (mean(x^p))^(1/p)). p=1 arith, p->0 geom, p=2 quad."""
    if p == 0: return fuse_geom(ps)
    avg = np.mean([x**p for x in ps], axis=0) ** (1/p)
    return avg / avg.sum(axis=1, keepdims=True)

combos = [
    ['DSN','BRD','AttRD'],          # known 92.32 trio
    ['DSN','RD','AttRD'],
    ['DSN','RD','BRD'],
    ['DSN','RD','BRD','AttRD'],
    ['DSN','AttRD'],
    ['DSN','BRD'],
]

print(f"{'Combo':35s}  arith  geom   max    vote   T=2.0  pow0.5 pow2.0")
print('-' * 90)
for combo in combos:
    ps = [M[n] for n in combo]
    a_arith = acc(fuse_arith(ps))
    a_geom  = acc(fuse_geom(ps))
    a_max   = acc(fuse_max(ps))
    a_vote  = acc(fuse_vote(ps))
    a_temp  = acc(fuse_tempered(ps, T=2.0))
    a_pow05 = acc(fuse_power(ps, p=0.5))
    a_pow2  = acc(fuse_power(ps, p=2.0))
    label = ' + '.join(combo)
    print(f'{label:35s}  {a_arith:5.2f}  {a_geom:5.2f}  {a_max:5.2f}  {a_vote:5.2f}  {a_temp:5.2f}  {a_pow05:5.2f}  {a_pow2:5.2f}')

# Best across all aggregators for all combos
print('\n=== Top 15 (any aggregator, train-best dumps) ===')
results = []
for combo in combos:
    ps = [M[n] for n in combo]
    label = ' + '.join(combo)
    for name, fn in [('arith', fuse_arith), ('geom', fuse_geom),
                      ('max', fuse_max), ('vote', fuse_vote),
                      ('T=2', lambda x: fuse_tempered(x, T=2.0)),
                      ('T=0.5', lambda x: fuse_tempered(x, T=0.5)),
                      ('p=0.5', lambda x: fuse_power(x, p=0.5)),
                      ('p=2', lambda x: fuse_power(x, p=2.0))]:
        results.append((acc(fn(ps)), f'{label}  [{name}]'))
results.sort(reverse=True)
for a, n in results[:15]:
    print(f'  {a:.2f}%   {n}')
