"""Honest-fusion analysis.

Loads softmax dumps and computes:
- per-model solo acc
- uniform 1/K honest fusion combinations
"""
import os, glob, numpy as np

os.chdir('/notebooks/PMamba/experiments')
D = 'dump_probs_runs'

# DSN logits need temperature 9.5 (per memory). All others already softmaxed.
def load_npz(path, is_dsn=False, dsn_temp=9.5):
    z = np.load(path)
    if 'logits' in z.files:
        L = z['logits']
        # softmax with temperature
        x = L * dsn_temp if is_dsn else L
        x = x - x.max(axis=1, keepdims=True)
        p = np.exp(x); p /= p.sum(axis=1, keepdims=True)
    else:
        p = z['probs']
    labels = z['labels']
    return p, labels

models = {
    'DSN':       (f'{D}/cvpr_dsn_K_depth.npz', True),
    'RD(N1)':    (f'{D}/realdeltanet_ep120.npz', False),
    'RD(N2)':    (f'{D}/realdeltanet_n2_ep120.npz', False),
    'RD(N14)':   (f'{D}/realdeltanet_n14_ep120.npz', False),
    'BRD':       (f'{D}/brd_ep120.npz', False),
    'AttRD':     (f'{D}/attrd_ep120.npz', False),
}

probs = {}
labels_ref = None
for name, (path, is_dsn) in models.items():
    if not os.path.exists(path):
        print(f'SKIP {name}: missing {path}'); continue
    p, l = load_npz(path, is_dsn=is_dsn)
    probs[name] = p
    if labels_ref is None: labels_ref = l
    else:
        assert np.array_equal(labels_ref, l), f'label mismatch for {name}'
    acc = (p.argmax(1) == l).mean() * 100
    print(f'  solo {name:10s} = {acc:.2f}%  shape={p.shape}')

print()
def fuse(names):
    ps = [probs[n] for n in names]
    avg = np.mean(ps, axis=0)
    return (avg.argmax(1) == labels_ref).mean() * 100

def report(combo):
    if not all(n in probs for n in combo):
        return None
    return fuse(list(combo))

combos = [
    ['DSN', 'RD(N1)'],
    ['DSN', 'RD(N2)'],
    ['DSN', 'RD(N14)'],
    ['DSN', 'BRD'],
    ['DSN', 'AttRD'],
    ['DSN', 'RD(N1)', 'RD(N2)'],
    ['DSN', 'RD(N1)', 'RD(N14)'],
    ['DSN', 'RD(N2)', 'RD(N14)'],
    ['DSN', 'RD(N1)', 'BRD'],
    ['DSN', 'RD(N1)', 'AttRD'],
    ['DSN', 'RD(N1)', 'RD(N2)', 'RD(N14)'],
    ['DSN', 'RD(N1)', 'RD(N2)', 'BRD'],
    ['DSN', 'RD(N1)', 'RD(N2)', 'AttRD'],
    ['DSN', 'RD(N1)', 'BRD', 'AttRD'],
    ['DSN', 'RD(N1)', 'RD(N2)', 'BRD', 'AttRD'],
    ['DSN', 'RD(N1)', 'RD(N2)', 'RD(N14)', 'BRD'],
    ['DSN', 'RD(N1)', 'RD(N2)', 'RD(N14)', 'AttRD'],
    ['DSN', 'RD(N1)', 'RD(N2)', 'RD(N14)', 'BRD', 'AttRD'],
]

print('=== Uniform 1/K honest fusion ===')
results = []
for combo in combos:
    acc = report(combo)
    if acc is not None:
        results.append((acc, ' + '.join(combo)))

results.sort(reverse=True)
for acc, name in results:
    print(f'  {acc:.2f}%   {name}')
