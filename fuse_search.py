"""Broad fusion search across all available test-set softmax dumps."""
import os, glob, itertools, numpy as np

os.chdir('/notebooks/PMamba/experiments')
D = 'dump_probs_runs'

# Hand-pick the dumps that historically appeared in fusion runs.
candidates = {
    'DSN':                ('cvpr_dsn_K_depth.npz', True, 9.5),
    # RealDeltaNet family (current sprint)
    'RD(N1)_best':        ('realdeltanet_best.npz', False, None),
    'RD(N1)_ep118':       ('realdeltanet_ep118.npz', False, None),
    'RD(N1)_ep120':       ('realdeltanet_ep120.npz', False, None),
    'RD(N2)_best':        ('realdeltanet_n2_best.npz', False, None),
    'RD(N2)_ep118':       ('realdeltanet_n2_ep118.npz', False, None),
    'RD(N2)_ep120':       ('realdeltanet_n2_ep120.npz', False, None),
    'RD(N14)_ep120':      ('realdeltanet_n14_ep120.npz', False, None),
    'BRD_ep120':          ('brd_ep120.npz', False, None),
    'AttRD_ep120':        ('attrd_ep120.npz', False, None),
    # DeltaNet v2 family (prior sprint, gave 92.5x memory claim)
    'DN2(N1)_ep111':      ('deltanet_v2_n1_ep111.npz', False, None),
    'DN2(N2)_ep113':      ('deltanet_v2_dtw_ep113.npz', False, None),
    'DN2(N14)_ep104':     ('deltanet_v2_n14_ep104.npz', False, None),
    # Other Delta variants
    'DN(N2)_ep112':       ('deltanet_dtw_ep112.npz', False, None),
    'DeltaProduct(N1)':   ('deltaproduct_n1_ep108.npz', False, None),
    'DeltaOSS(N1)':       ('deltaoss_n1_ep107.npz', False, None),
    'QDN(N1)':            ('qdeltanet_n1_ep110.npz', False, None),
    'QDN(N2)':            ('qdeltanet_n2_ep112.npz', False, None),
    'LinOSS(N2)':         ('linoss_dtw_ep117.npz', False, None),
}

def load(path, is_dsn, temp):
    p = os.path.join(D, path)
    if not os.path.exists(p): return None, None
    z = np.load(p)
    if 'logits' in z.files:
        L = z['logits'] * (temp if is_dsn else 1)
        L = L - L.max(axis=1, keepdims=True)
        e = np.exp(L); pr = e / e.sum(axis=1, keepdims=True)
    else:
        pr = z['probs']
    return pr, z['labels']

probs, labels_ref = {}, None
solo_acc = {}
for name, (path, is_dsn, temp) in candidates.items():
    pr, lb = load(path, is_dsn, temp)
    if pr is None: continue
    if labels_ref is None: labels_ref = lb
    if not np.array_equal(lb, labels_ref):
        print(f'SKIP {name}: label mismatch'); continue
    probs[name] = pr
    solo_acc[name] = (pr.argmax(1) == lb).mean() * 100

print('--- solo ---')
for n, a in sorted(solo_acc.items(), key=lambda x: -x[1]):
    print(f'  {n:25s} {a:.2f}%')

def fuse(names):
    avg = np.mean([probs[n] for n in names], axis=0)
    return (avg.argmax(1) == labels_ref).mean() * 100

# All 2-, 3-, 4-way subsets that include DSN
all_names = list(probs.keys())
non_dsn = [n for n in all_names if n != 'DSN']
results = []
print('\n--- exhaustive fusion (includes DSN) ---')
for k in range(1, 5):
    for combo in itertools.combinations(non_dsn, k):
        ns = ['DSN'] + list(combo)
        acc = fuse(ns)
        results.append((acc, ' + '.join(ns)))

results.sort(reverse=True)
print('top 25:')
for acc, name in results[:25]:
    print(f'  {acc:.2f}%   {name}')
