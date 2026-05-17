"""Fuse BDN-Q (train-best) into the current 5-way 92.53% ceiling.

Tests:
  1) reproduce 5-way ceiling (DSN + RD + BRD(N2) + AttRD + DN2(N1))
  2) BDN-Q solo
  3) 6-way (5 + BDN-Q)
  4) BDN-Q substituted for each of the 5
  5) other combos that include BDN-Q
"""
import os, itertools, numpy as np
os.chdir('/notebooks/PMamba/experiments')
D = 'dump_probs_runs'

# (path, is_dsn, temp)
specs = {
    'DSN':     ('cvpr_dsn_K_depth.npz',     True,  9.5),
    'RD':      ('realdeltanet_ep118.npz',   False, None),
    'BRD(N2)': ('brd_n2_ep109.npz',         False, None),
    'AttRD':   ('attrd_ep120.npz',          False, None),
    'DN2(N1)': ('deltanet_v2_n1_ep109.npz', False, None),
    'BDN-Q':   ('bdnq_train_best.npz',      False, None),
}

def load(path, is_dsn, temp):
    z = np.load(os.path.join(D, path))
    if 'logits' in z.files:
        L = z['logits'] * (temp if is_dsn and temp else 1)
        L = L - L.max(axis=1, keepdims=True)
        e = np.exp(L); return e / e.sum(axis=1, keepdims=True), z['labels']
    return z['probs'], z['labels']

probs, lref = {}, None
for n, (p, dsn, t) in specs.items():
    pr, lb = load(p, dsn, t)
    if lref is None: lref = lb
    assert np.array_equal(lb, lref), f'label mismatch {n}'
    probs[n] = pr
    acc = (pr.argmax(1) == lb).mean() * 100
    print(f'  solo {n:9s} = {acc:5.2f}%  shape={pr.shape}')

def fuse(names):
    ps = np.mean([probs[n] for n in names], axis=0)
    return (ps.argmax(1) == lref).mean() * 100

ceiling5 = ['DSN', 'RD', 'BRD(N2)', 'AttRD', 'DN2(N1)']
print()
print(f'5-way ceiling  ({", ".join(ceiling5)}) = {fuse(ceiling5):.2f}%')
print(f'6-way + BDN-Q                            = {fuse(ceiling5 + ["BDN-Q"]):.2f}%')

print()
print('BDN-Q substituted for each of the 5:')
for swap in ceiling5:
    if swap == 'DSN': continue   # DSN is the non-PMamba anchor; keep it
    new = [n if n != swap else 'BDN-Q' for n in ceiling5]
    acc = fuse(new)
    print(f'  swap {swap:9s} -> BDN-Q : {acc:.2f}%  ({", ".join(new)})')

print()
print('Smaller combos with BDN-Q:')
others = ['RD', 'BRD(N2)', 'AttRD', 'DN2(N1)']
print(f'  DSN + BDN-Q                              = {fuse(["DSN", "BDN-Q"]):.2f}%')
for k in (2, 3, 4):
    best = (-1, None)
    for combo in itertools.combinations(others, k):
        names = ['DSN', 'BDN-Q', *combo]
        a = fuse(names)
        if a > best[0]: best = (a, names)
    print(f'  best {k+2}-way w/BDN-Q                       = {best[0]:.2f}%  ({", ".join(best[1])})')
