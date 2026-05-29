"""Honest fixed-fusion re-eval of the q-rot-cycle battery.

For every qexp run we fuse the SAME FG83 base with that run's correspondence
branch logits using a FIXED, a-priori fusion rule (uniform log-prob, temp=1.0),
NOT a grid-search tuned on the 482-sample eval set.

Two honest signals per run:
  - branch_solo : branch argmax acc (no fusion, no tuning at all)
  - fixed@w     : acc of log_softmax(base) + w*log_softmax(branch_final), temp 1

For reference / contamination size we also report:
  - tuned       : the trainer's test-tuned grid fusion (max over 2989 configs)

Branch logits = FINAL epoch (branch_logits.npz), selected by the same early-stop
rule for every config -> fair A/B. best_branch_logits.npz (test-tuned epoch) is
only used for the 'tuned' reference column.
"""
import glob, os, re, json
import numpy as np

WD = '/notebooks/Anemon/experiments/work_dir'
BASE = f'{WD}/depth_small_r2_fg83_restored_20260528_033028/best_logits.npz'
FIXED_WS = [0.25, 0.5, 0.75, 1.0, 1.5]
HEADLINE_W = 1.0  # uniform 1/K in log-prob space (base:branch = 1:1)


def lsm(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def topk_acc(logp, y, k=1):
    if k == 1:
        return (logp.argmax(1) == y).mean() * 100
    top = np.argsort(-logp, 1)[:, :k]
    return np.mean([y[i] in top[i] for i in range(len(y))]) * 100


def load_aligned(path, ref_sig):
    d = np.load(path, allow_pickle=True)
    logits = d['logits'] if 'logits' in d.files else d['base_logits']
    y = d['labels']
    sig = d['sigs'] if 'sigs' in d.files else None
    if ref_sig is not None and sig is not None:
        order = {str(s): i for i, s in enumerate(sig)}
        idx = np.array([order[str(s)] for s in ref_sig])
        logits, y = logits[idx], y[idx]
    return np.asarray(logits, dtype=np.float64), np.asarray(y)


def tuned_fusion(base, branch, y):
    temps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    weights = [i / 20.0 for i in range(0, 61)]
    best = 0.0
    for tb in temps:
        bp = lsm(base / tb)
        for tc in temps:
            cp = lsm(branch / tc)
            for w in weights:
                a = (bp + w * cp).argmax(1)
                acc = (a == y).mean() * 100
                if acc > best:
                    best = acc
    return best


bd = np.load(BASE, allow_pickle=True)
ref_sig = bd['sigs'] if 'sigs' in bd.files else None
base, ybase = load_aligned(BASE, None)
base_acc = topk_acc(lsm(base), ybase)
print(f'FG83 base solo: {base_acc:.3f}  (N={len(ybase)})')
print(f'fixed fusion rule: log_softmax(base) + w*log_softmax(branch_final), temp=1.0')
print(f'HEADLINE w={HEADLINE_W} (uniform)\n')


def cfg(n):
    for c in ('clean', 'qrot', 'ident', 'amp'):
        if n.startswith(c):
            return c
    return '?'


rows = []
for wd in sorted(glob.glob(f'{WD}/qexp/*')):
    name = os.path.basename(wd)
    bl = os.path.join(wd, 'branch_logits.npz')
    runlog = os.path.join(wd, 'run.log')
    if not os.path.exists(bl):
        continue
    txt = open(runlog, errors='replace').read() if os.path.exists(runlog) else ''
    done = ('FINAL' in txt) or ('EARLY_STOP' in txt)
    br, ybr = load_aligned(bl, ref_sig)
    assert np.array_equal(ybr, ybase), f'{name} label mismatch'
    lbr = lsm(br)
    solo = topk_acc(lbr, ybase)
    fixed = {w: topk_acc(lsm(base) + w * lbr, ybase) for w in FIXED_WS}
    # tuned reference from best_branch (test-tuned epoch) if present
    bb = os.path.join(wd, 'best_branch_logits.npz')
    tuned = None
    if os.path.exists(bb):
        bbl, ybb = load_aligned(bb, ref_sig)
        tuned = tuned_fusion(base, bbl, ybase)
    rows.append((name, done, solo, fixed, tuned))

# per-run table
hdr = f'{"run":14s} {"st":3s} {"solo":>6s} ' + ' '.join(f'f{w:g}'.rjust(6) for w in FIXED_WS) + f' {"tuned":>6s}'
print(hdr)
for name, done, solo, fixed, tuned in rows:
    st = 'D' if done else '.'
    fx = ' '.join(f'{fixed[w]:6.2f}' for w in FIXED_WS)
    tn = f'{tuned:6.2f}' if tuned is not None else '   -  '
    print(f'{name:14s} {st:>3s} {solo:6.2f} {fx} {tn}')

# per-config aggregates over DONE runs
print('\n=== per-config means over DONE runs ===')
agg = {}
for name, done, solo, fixed, tuned in rows:
    if not done:
        continue
    c = cfg(name)
    agg.setdefault(c, {'solo': [], 'tuned': []})
    agg[c]['solo'].append(solo)
    if tuned is not None:
        agg[c]['tuned'].append(tuned)
    for w in FIXED_WS:
        agg[c].setdefault(f'f{w}', []).append(fixed[w])

order = ['clean', 'qrot', 'ident', 'amp']
print(f'{"cfg":6s} {"n":>2s} {"solo":>6s} ' + ' '.join(f'f{w:g}'.rjust(6) for w in FIXED_WS) + f' {"tuned":>6s}')
for c in order:
    if c not in agg:
        continue
    a = agg[c]
    n = len(a['solo'])
    solo = np.mean(a['solo'])
    fx = ' '.join(f'{np.mean(a[f"f{w}"]):6.2f}' for w in FIXED_WS)
    tn = f'{np.mean(a["tuned"]):6.2f}' if a['tuned'] else '   -  '
    print(f'{c:6s} {n:>2d} {solo:6.2f} {fx} {tn}')

# ---- PAIRED verdict on seeds DONE for all three controls ----
def seed_of(n):
    m = re.search(r'_s(\d+)$', n)
    return m.group(1) if m else None

bycfg = {'clean': {}, 'qrot': {}, 'ident': {}}
for name, done, solo, fixed, tuned in rows:
    c = cfg(name)
    s = seed_of(name)
    if c in bycfg and done and s is not None:
        bycfg[c][s] = {'solo': solo, **{f'f{w}': fixed[w] for w in FIXED_WS}}

common = set(bycfg['clean']) & set(bycfg['qrot']) & set(bycfg['ident'])
common = sorted(common)
print(f'\n=== PAIRED verdict on common DONE seeds {common} (n={len(common)}) ===')
if common:
    metrics = ['solo', 'f0.5', 'f0.75', 'f1.0']
    print(f'{"cfg":6s} ' + ' '.join(m.rjust(7) for m in metrics))
    means = {}
    for c in ('clean', 'qrot', 'ident'):
        means[c] = {m: np.mean([bycfg[c][s][m] for s in common]) for m in metrics}
        print(f'{c:6s} ' + ' '.join(f'{means[c][m]:7.2f}' for m in metrics))
    print('\n-- paired deltas (mean over seeds) + per-seed sign --')
    for a, b in [('qrot', 'clean'), ('ident', 'clean'), ('qrot', 'ident')]:
        line = f'  {a:5s} - {b:5s}: '
        for m in metrics:
            diffs = [bycfg[a][s][m] - bycfg[b][s][m] for s in common]
            pos = sum(1 for d in diffs if d > 1e-9)
            line += f'{m}={np.mean(diffs):+.2f}({pos}/{len(common)}) '
        print(line)

# deltas vs clean at headline w
if 'clean' in agg:
    print(f'\n=== delta vs clean at fixed w={HEADLINE_W} (DONE runs) ===')
    cb = np.mean(agg['clean'][f'f{HEADLINE_W}'])
    cs = np.mean(agg['clean']['solo'])
    for c in order:
        if c not in agg:
            continue
        dfix = np.mean(agg[c][f'f{HEADLINE_W}']) - cb
        dsolo = np.mean(agg[c]['solo']) - cs
        print(f'  {c:6s}  d_fixed={dfix:+.3f}  d_solo={dsolo:+.3f}')
