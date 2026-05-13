"""Honest fusion using train-best epochs (selected via training acc, not test)."""
import os, itertools, numpy as np

os.chdir('/notebooks/PMamba/experiments')
D = 'dump_probs_runs'

# Train-best epochs per model (selected from training-accuracy peak)
candidates = {
    'DSN':        ('cvpr_dsn_K_depth.npz', True, 9.5),
    'RD(N1)':     ('realdeltanet_ep118.npz', False, None),     # train-best ep118
    'RD(N2)':     ('realdeltanet_n2_ep118.npz', False, None),  # train-best ep118
    'RD(N14)':    ('realdeltanet_n14_ep112.npz', False, None), # train-best ep112
    'BRD':        ('brd_ep112.npz', False, None),              # train-best ep112
    'AttRD':      ('attrd_ep120.npz', False, None),            # train-best ep120
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

probs, labels_ref, solo = {}, None, {}
for name, (path, is_dsn, temp) in candidates.items():
    pr, lb = load(path, is_dsn, temp)
    if pr is None:
        print(f'MISSING {name}: {path}'); continue
    if labels_ref is None: labels_ref = lb
    assert np.array_equal(lb, labels_ref), f'label mismatch for {name}'
    probs[name] = pr
    solo[name] = (pr.argmax(1) == lb).mean() * 100

print('=== Solo (train-best epoch) ===')
for n, a in sorted(solo.items(), key=lambda x: -x[1]):
    print(f'  {n:10s} {a:.2f}%')

def fuse(names):
    avg = np.mean([probs[n] for n in names], axis=0)
    return (avg.argmax(1) == labels_ref).mean() * 100

all_names = list(probs.keys())
non_dsn = [n for n in all_names if n != 'DSN']
results = []
print('\n=== All fusion combos (1/K uniform, DSN included) ===')
for k in range(1, len(non_dsn) + 1):
    for combo in itertools.combinations(non_dsn, k):
        ns = ['DSN'] + list(combo)
        acc = fuse(ns)
        results.append((acc, ' + '.join(ns)))

results.sort(reverse=True)
for acc, name in results[:25]:
    print(f'  {acc:.2f}%   {name}')
