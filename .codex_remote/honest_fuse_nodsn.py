"""Honest DSN-free fusion: cnxxl quat-head + FG83 depth.

Fusion (branch_temp T, weight w) is chosen on a held-out 30% slice of TRAIN
(fixed seed, no test peeking); the test set is touched only for the final report.
Reports cnxxl solo as the reference single-model number.
"""
import re
import numpy as np

WD = '/notebooks/Anemon/experiments/work_dir'
CN_TR = f'{WD}/cn_xxl_quat_head/train_logits.npz'
CN_TE = f'{WD}/cn_xxl_quat_head/test_logits.npz'
FG_TR = f'{WD}/depth_small_r2_fg83_restored_20260528_033028/train_logits.npz'
FG_TE = f'{WD}/depth_small_r2_fg83_restored_20260528_033028/test_logits.npz'


def sig_of(s):
    s = str(s)
    m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', s)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}' if m else s


def load(p):
    A = np.load(p, allow_pickle=True)
    log = A['logits'] if 'logits' in A.files else A['pred_logits']
    lab = A['labels']
    sig = np.array([sig_of(s) for s in (A['sigs'] if 'sigs' in A.files else A['paths'])])
    return log.astype(np.float64), lab, sig


def align(a_log, a_lab, a_sig, ref_sig):
    by = {s: i for i, s in enumerate(a_sig)}
    idx = np.array([by[s] for s in ref_sig])
    return a_log[idx], a_lab[idx]


def lsm(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def acc(logp, y):
    return (logp.argmax(1) == y).mean() * 100


# load + align
cn_tr, y_tr, s_tr = load(CN_TR)
fg_tr, _, fs_tr = load(FG_TR)
fg_tr, _ = align(fg_tr, _, fs_tr, s_tr)
cn_te, y_te, s_te = load(CN_TE)
fg_te, yfg, fs_te = load(FG_TE)
fg_te, yfg = align(fg_te, yfg, fs_te, s_te)
assert (yfg == y_te).all()

print(f'train N={len(y_tr)}  test N={len(y_te)}')
print(f'cnxxl solo  test: {acc(lsm(cn_te), y_te):.2f}%  ({(lsm(cn_te).argmax(1)==y_te).sum()}/{len(y_te)})')
print(f'fg83  solo  test: {acc(lsm(fg_te), y_te):.2f}%')
print(f'cnxxl solo train: {acc(lsm(cn_tr), y_tr):.2f}%   (overfit ref)')

# fixed held-out calibration slice of TRAIN (no test peeking)
rng = np.random.RandomState(0)
perm = rng.permutation(len(y_tr))
ncal = int(0.30 * len(y_tr))
cal = perm[:ncal]
print(f'\ncalibration slice: {ncal} train samples (seed 0)')

temps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
weights = [round(i / 20.0, 3) for i in range(0, 41)]  # 0..2.0
cn_cal_lsm = lsm(cn_tr[cal])
best = None
for T in temps:
    fg_cal_lsm = lsm(fg_tr[cal], T)
    for w in weights:
        a = acc(cn_cal_lsm + w * fg_cal_lsm, y_tr[cal])
        # tie-break toward smaller w (less reliance on weak branch)
        if best is None or a > best[0] + 1e-9 or (abs(a - best[0]) < 1e-9 and w < best[2]):
            best = (a, T, w)
cal_acc, T, w = best
print(f'chosen on cal: T={T} w={w}  cal_acc={cal_acc:.2f}%')

# apply fixed to test
fused_te = lsm(cn_te) + w * lsm(fg_te, T)
print(f'\n==> honest cnxxl+fg83 TEST: {acc(fused_te, y_te):.2f}%  ({(fused_te.argmax(1)==y_te).sum()}/{len(y_te)})')

# informational: test-only sweep (NOT used to choose) to show the ceiling/ gap
print('\n(test-only sweep, informational, NOT used to pick):')
bestt = 0
for T in temps:
    for w in weights:
        a = acc(lsm(cn_te) + w * lsm(fg_te, T), y_te)
        bestt = max(bestt, a)
print(f'  best achievable on test (oracle weight): {bestt:.2f}%')
