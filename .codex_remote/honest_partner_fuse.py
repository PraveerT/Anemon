"""Honest fusion eval: does the quaternion-bottleneck CN-XXL help CN-XXL reach
honest >92 without DSN, and is the quaternion non-decorative vs the real-head
control?

Fusion rule = FIXED equal-weight log-prob (1/K) -> no test-tuning. Also reports
an oracle weight sweep (informational) and error-overlap diagnostics.
"""
import numpy as np, os, re

WD = '/notebooks/Anemon/experiments/work_dir'
PATHS = {
    'cnxxl':  f'{WD}/cn_xxl_quat_head/test_logits.npz',          # 91.29 test-best
    'cnxxl_tb': f'{WD}/cn_xxl_quat_head/test_logits_train_best.npz',  # 90.66 honest
    'quat':   f'{WD}/quat_bottleneck/test_logits.npz',
    'real':   f'{WD}/quat_bottleneck_realctrl/test_logits.npz',
    'fg83':   f'{WD}/depth_small_r2_fg83_restored_20260528_033028/test_logits.npz',
}


def sig_of(s):
    s = str(s); m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', s)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}' if m else s


def load(p, ref=None):
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    lg = d['logits'] if 'logits' in d.files else d['base_logits']
    y = d['labels']
    sg = np.array([sig_of(s) for s in (d['sigs'] if 'sigs' in d.files else d['paths'])]) if ('sigs' in d.files or 'paths' in d.files) else None
    if ref is not None and sg is not None:
        by = {s: i for i, s in enumerate(sg)}
        idx = np.array([by[s] for s in ref])
        lg, y = lg[idx], y[idx]
    return {'lg': lg.astype(np.float64), 'y': y, 'sg': sg}


def lsm(z):
    z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


def acc(lg, y):
    return (lg.argmax(1) == y).mean() * 100


cn = load(PATHS['cnxxl'])
ref = cn['sg']; y = cn['y']
D = {k: load(v, ref) for k, v in PATHS.items()}
print('=== solo ===')
for k in D:
    if D[k] is not None:
        print(f'  {k:8s} {acc(D[k]["lg"], y):.3f}  ({(D[k]["lg"].argmax(1)==y).sum()}/{len(y)})')

cnp = lsm(cn['lg']); cn_err = cn['lg'].argmax(1) != y
N = len(y)


def overlap(name):
    if D[name] is None: return
    pe = D[name]['lg'].argmax(1) != y
    shared = (cn_err & pe).sum()
    cn_only = (cn_err & ~pe).sum()      # partner fixes these
    p_only = (~cn_err & pe).sum()       # partner breaks these
    print(f'  {name}: cnxxl_errs={cn_err.sum()} partner_errs={pe.sum()} shared={shared} '
          f'partner_fixes={cn_only} partner_breaks={p_only}')


print('\n=== error overlap vs cnxxl (test-best 91.29) ===')
for k in ('quat', 'real', 'fg83'):
    overlap(k)


def fuse(parts, ws):
    s = sum(w * lsm(D[p]['lg']) for p, w in zip(parts, ws))
    return acc(s, y)


print('\n=== honest fixed fusion (equal-weight 1/K, no tuning) ===')
for base in ('cnxxl', 'cnxxl_tb'):
    if D['quat'] is not None:
        print(f'  {base}+quat (1:1): {fuse([base,"quat"],[1,1]):.3f}')
    if D['real'] is not None:
        print(f'  {base}+real (1:1): {fuse([base,"real"],[1,1]):.3f}')
    if D['quat'] is not None and D['fg83'] is not None:
        print(f'  {base}+quat+fg83 (1:1:0.3): {fuse([base,"quat","fg83"],[1,1,0.3]):.3f}')

print('\n=== oracle weight sweep cnxxl+quat (informational, NOT honest) ===')
if D['quat'] is not None:
    best = 0; bw = 0
    for w in np.arange(0, 2.01, 0.05):
        a = acc(cnp + w * lsm(D['quat']['lg']), y)
        if a > best: best, bw = a, w
    print(f'  cnxxl+quat oracle: {best:.3f} @ w={bw:.2f}')
