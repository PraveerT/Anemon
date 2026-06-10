"""Fuse dense-PGCNet (87.76) with the off-floor PGCNet (~89) from saved test_logits.npz,
uniform softmax-sum (no test-tuning), and report error overlap + oracle."""
import numpy as np


def load(f):
    z = np.load(f, allow_pickle=True); ks = list(z.keys())
    L = y = None
    for k in ks:
        a = z[k]
        if a.ndim == 2 and a.shape[-1] == 25:
            L = a.astype(np.float64)
        kl = k.lower()
        if a.ndim == 1 and ('label' in kl or 'target' in kl or 'true' in kl or kl == 'y'):
            y = a.astype(np.int64)
    if y is None:
        for k in ks:
            if z[k].ndim == 1 and z[k].dtype.kind in 'iu':
                y = z[k].astype(np.int64); break
    return L, y, ks


def sm(L):
    e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


Ld, yd, kd = load('dense_logits.npz')
Lo, yo, ko = load('off_logits.npz')
print('dense npz keys:', kd, '| logits', None if Ld is None else Ld.shape)
print('off   npz keys:', ko, '| logits', None if Lo is None else Lo.shape)
y = yd if yd is not None else yo
if yd is not None and yo is not None:
    print('labels identical (same test order):', np.array_equal(yd, yo))

pd, po = Ld.argmax(1), Lo.argmax(1)
print('\nSOLO   dense %.4f (%d/482)   off %.4f (%d/482)'
      % ((pd == y).mean(), (pd == y).sum(), (po == y).mean(), (po == y).sum()))

fused = (sm(Ld) + sm(Lo)).argmax(1)
print('FUSED  uniform 1/2 %.4f (%d/482)' % ((fused == y).mean(), (fused == y).sum()))

ed = set(np.where(pd != y)[0]); eo = set(np.where(po != y)[0])
shared, union = ed & eo, ed | eo
print('\nERRORS dense=%d  off=%d  shared=%d  union=%d  overlap(shared/union)=%.2f'
      % (len(ed), len(eo), len(shared), len(union), len(shared) / max(1, len(union))))
print('ORACLE (either model correct): %.4f (%d/482)' % (1 - len(shared) / 482.0, 482 - len(shared)))
print('off-only-wrong dense-fixes: %d   dense-only-wrong off-fixes: %d' % (len(eo - ed), len(ed - eo)))
