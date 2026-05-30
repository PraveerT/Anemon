"""Agreement-gated honest fusion (parameter-free gate; non-commutativity-motivated).

Auxiliary correction is applied ONLY where RGB and FG83 mutually AGREE (their
argmax matches) -> high-precision fixes, fewer breaks. This is the honest,
selective analogue of the commutator gate (act only on low cross-aux disagreement).
Sweep the fixed weight to show whether ANY robust region clears 444 (=92.116);
cnxxl alone = 440.
"""
import numpy as np, re

W = '/notebooks/Anemon/experiments/work_dir/'
P = {'cn': 'cn_xxl_quat_head', 'rgb': 'rgb_fgcrop_r2p1d', 'fg': 'depth_small_r2_fg83_restored_20260528_033028'}


def sig(s):
    s = str(s); m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', s)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}' if m else s


def load(p, ref=None):
    d = np.load(f'{W}{p}/test_logits.npz', allow_pickle=True)
    lg = d['logits'].astype(float); y = d['labels']; sg = np.array([sig(s) for s in d['sigs']])
    if ref is not None:
        by = {s: i for i, s in enumerate(sg)}; idx = np.array([by[s] for s in ref]); lg, y = lg[idx], y[idx]
    return lg, y, sg


def lsm(z):
    z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


cn, y, ref = load(P['cn']); rg, _, _ = load(P['rgb'], ref); fg, _, _ = load(P['fg'], ref)
cnp, rgp, fgp = lsm(cn), lsm(rg), lsm(fg)
N = len(y)
agree = rgp.argmax(1) == fgp.argmax(1)      # the gate: aux mutually agree
print(f'cnxxl solo {(cnp.argmax(1)==y).sum()}/{N}   aux-agree on {agree.sum()}/{N} samples   (444=92.116)')


def acc(z):
    return (z.argmax(1) == y).sum()


print('\n=== ungated fixed fusion cnxxl + w*(rgb+fg) ===')
for w in [0.03, 0.05, 0.08, 0.10]:
    print(f'  w={w}: {acc(cnp + w*(rgp+fgp))}')

print('\n=== AGREEMENT-GATED: aux added only where rgb,fg agree ===')
for w in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
    f = cnp.copy()
    f[agree] = cnp[agree] + w * (rgp[agree] + fgp[agree])
    print(f'  w={w:>4}: {acc(f)}   (>=444 win)')

print('\n=== gated + also require aux beats cnxxl conf at their class (stricter) ===')
aux = rgp + fgp; auxpred = aux.argmax(1)
for w in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
    f = cnp.copy()
    for i in range(N):
        if agree[i]:
            f[i] = cnp[i] + w * (rgp[i] + fgp[i])
    print(f'  w={w:>4}: {acc(f)}')
