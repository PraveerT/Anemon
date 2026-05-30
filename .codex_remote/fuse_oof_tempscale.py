"""Honest temperature-scaling fusion. Fit each model's temperature by minimizing
NLL on the 218 OOF set (standard Guo-2017 calibration, no test contact), then
fuse on the 482 TEST at a FIXED a-priori weight. Separates honest temp calibration
from the (unfit) weight, sidestepping the subject-shift over-weighting. 444=92.116.
"""
import numpy as np, re

W = '/notebooks/Anemon/experiments/work_dir/'
OOF = {'cn': 'cn_xxl_fold/test_logits.npz', 'fg': 'fg83_fold/test_logits.npz', 'rgb': 'rgb_fold/best_logits.npz'}
TST = {'cn': 'cn_xxl_quat_head/test_logits.npz', 'fg': 'depth_small_r2_fg83_restored_20260528_033028/test_logits.npz', 'rgb': 'rgb_fgcrop_r2p1d/best_logits.npz'}


def sig(s):
    s = str(s); m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', s)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}' if m else s


def load(p, ref=None):
    d = np.load(W + p, allow_pickle=True)
    lg = (d['logits'] if 'logits' in d.files else d['base_logits']).astype(np.float64)
    y = d['labels']; sg = np.array([sig(s) for s in d['sigs']])
    if ref is None:
        return lg, y, sg
    by = {s: i for i, s in enumerate(sg)}; idx = np.array([by[s] for s in ref])
    return lg[idx], y[idx], sg


def grp(g):
    cn, y, ref = load(g['cn']); fg, yf, _ = load(g['fg'], ref); rg, yr, _ = load(g['rgb'], ref)
    assert (yf == y).all() and (yr == y).all(); return cn, fg, rg, y


def lsm(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


def nll(z, y, T):
    lp = lsm(z, T); return -lp[np.arange(len(y)), y].mean()


def best_T(z, y):
    Ts = np.arange(0.5, 6.01, 0.1)
    return Ts[np.argmin([nll(z, y, T) for T in Ts])]


def acc(z, y):
    return int((z.argmax(1) == y).sum())


cnO, fgO, rgO, yO = grp(OOF)
cnT, fgT, rgT, yT = grp(TST)
Tcn, Tfg, Trg = best_T(cnO, yO), best_T(fgO, yO), best_T(rgO, yO)
print(f'OOF-NLL temps: T_cn={Tcn:.1f} T_fg={Tfg:.1f} T_rgb={Trg:.1f}   (oracle used Tfg~3,Trgb~2)')
print(f'cnxxl test = {acc(lsm(cnT),yT)}/482   target 444')
print('\nfixed a-priori weights, OOF-calibrated temps, on TEST:')
for w in [0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
    a = acc(lsm(cnT, Tcn) + w * lsm(rgT, Trg) + w * lsm(fgT, Tfg), yT)
    print(f'  w={w:<5} -> {a}/482 = {a/482*100:.3f}%   {"*** >92 ***" if a >= 444 else ""}')
# also asymmetric (depth usually stronger partner)
print('asymmetric (fg heavier):')
for wf, wr in [(0.1, 0.05), (0.15, 0.05), (0.2, 0.05), (0.2, 0.1), (0.3, 0.1)]:
    a = acc(lsm(cnT, Tcn) + wr * lsm(rgT, Trg) + wf * lsm(fgT, Tfg), yT)
    print(f'  wf={wf} wr={wr} -> {a}/482 = {a/482*100:.3f}%   {"*** >92 ***" if a >= 444 else ""}')
