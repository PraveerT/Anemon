"""Wide fusion search: every available dump + every train-best, looking for combos > 92.32."""
import os, glob, itertools, numpy as np

os.chdir('/notebooks/PMamba/experiments')
D = 'dump_probs_runs'

def load_dsn(p, t=9.5):
    z = np.load(p); L = z['logits']*t; L=L-L.max(1,keepdims=True); e=np.exp(L); return e/e.sum(1,keepdims=True), z['labels']
def load_p(p):
    z = np.load(p); return z['probs'], z['labels']

# Build pool: prefer train-best epochs when known, else use ep120
pool = {
    'DSN':          ('cvpr_dsn_K_depth.npz', True),  # fixed
    # Train-best epochs (verified)
    'RD':           ('realdeltanet_ep118.npz', False),
    'BRD':          ('brd_ep112.npz', False),
    'AttRD':        ('attrd_ep120.npz', False),
    'RD(N2)':       ('realdeltanet_n2_ep118.npz', False),
    'RD(N14)':      ('realdeltanet_n14_ep112.npz', False),
    # Other existing dumps (may not be train-best, but worth trying)
    'AttRDv2':      ('attrd_v2_ep110.npz', False),
    'AttRD(N2)':    ('attrd_n2_ep112.npz', False),
    'DN2(N1)':      ('deltanet_v2_n1_ep111.npz', False),
    'DN2(N2)':      ('deltanet_v2_dtw_ep113.npz', False),
    'DN2(N14)':     ('deltanet_v2_n14_ep104.npz', False),
    'DN(N2)':       ('deltanet_dtw_ep112.npz', False),
    'DeltaProduct': ('deltaproduct_n1_ep108.npz', False),
    'DeltaOSS':     ('deltaoss_n1_ep107.npz', False),
    'QDN(N1)':      ('qdeltanet_n1_ep110.npz', False),
    'QDN(N2)':      ('qdeltanet_n2_ep112.npz', False),
    'LinOSS(N2)':   ('linoss_dtw_ep117.npz', False),
}

M = {}; L = None
for name, (path, is_dsn) in pool.items():
    fp = os.path.join(D, path)
    if not os.path.exists(fp): continue
    if is_dsn: p, l = load_dsn(fp)
    else:      p, l = load_p(fp)
    if L is None: L = l
    if not np.array_equal(l, L): continue
    M[name] = p

print(f'Loaded {len(M)} models: {list(M.keys())}')
print()

def acc_combo(combo):
    avg = np.mean([M[n] for n in combo], axis=0)
    return (avg.argmax(1) == L).mean() * 100

# Look at all 3- and 4-way combos with DSN (since DSN is the strongest solo)
names = list(M.keys())
non_dsn = [n for n in names if n != 'DSN']
print('=== Top 25 (DSN + N partners, N <= 4) ===')
res = []
for k in range(1, 5):
    for c in itertools.combinations(non_dsn, k):
        res.append((acc_combo(['DSN'] + list(c)), 'DSN + ' + ' + '.join(c)))
res.sort(reverse=True)
for a, n in res[:25]: print(f'  {a:.2f}%  {n}')

# Also 5-way
print('\n=== Top 10 (5-way with DSN) ===')
res5 = []
for c in itertools.combinations(non_dsn, 5):
    res5.append((acc_combo(['DSN'] + list(c)), 'DSN + ' + ' + '.join(c)))
res5.sort(reverse=True)
for a, n in res5[:10]: print(f'  {a:.2f}%  {n}')
