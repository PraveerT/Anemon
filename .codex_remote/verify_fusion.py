import numpy as np

wd = '/notebooks/Anemon/experiments/work_dir/'
cn = np.load(wd + 'cn_xxl_quat_head/test_logits.npz', allow_pickle=True)
fg = np.load(wd + 'depth_small_r2_fg83_restored_20260528_033028/best_logits.npz', allow_pickle=True)
qr = np.load(wd + 'cnxxl_qrot_tta_z4_logits.npz', allow_pickle=True)

y = cn['labels']
N = len(y)
assert np.array_equal(y, fg['labels']) and np.array_equal(y, qr['labels'])


def lsm(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def acc(z):
    return (z.argmax(1) == y).sum()


cnxxl = lsm(cn['logits'])
fg83 = lsm(fg['logits'])
qrot_z2 = lsm(qr['z+2'])
qrot_base = lsm(qr['base_logits'])

print(f'N={N}')
print(f'cnxxl solo     {acc(cnxxl)}/{N}  {acc(cnxxl)/N*100:.3f}')
print(f'fg83  solo     {acc(fg83)}/{N}  {acc(fg83)/N*100:.3f}')
print(f'qrot z+2 solo  {acc(qrot_z2)}/{N}  {acc(qrot_z2)/N*100:.3f}')
print(f'qrot base solo {acc(qrot_base)}/{N}  {acc(qrot_base)/N*100:.3f}')
print()

# stated candidate
score = cnxxl + 0.05 * fg83 + 0.06 * qrot_z2
print(f'STATED 0.05 fg83 + 0.06 qrot z+2: {acc(score)}/{N}  {acc(score)/N*100:.3f}')
print()

# 2D weight sweep around the candidate
print('weight sweep: rows=fg83 w, cols=qrot z+2 w  (correct count out of 482)')
fgw = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
qrw = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
header = 'fg\\qr  ' + ' '.join(f'{q:>5.2f}' for q in qrw)
print(header)
best = (0, None)
for f in fgw:
    row = []
    for q in qrw:
        a = acc(cnxxl + f * fg83 + q * qrot_z2)
        row.append(a)
        if a > best[0]:
            best = (a, (f, q))
    print(f'{f:>5.3f} ' + ' '.join(f'{r:>5d}' for r in row))
print(f'\nbest in grid: {best[0]}/{N} at (fg={best[1][0]}, qr={best[1][1]})')

# how many distinct (f,q) reach >=444 ?
n444 = 0
total = 0
for f in np.arange(0, 0.31, 0.01):
    for q in np.arange(0, 0.31, 0.01):
        total += 1
        if acc(cnxxl + f * fg83 + q * qrot_z2) >= 444:
            n444 += 1
print(f'fraction of fine grid (0..0.30 step .01, {total} pts) reaching >=444: {n444}/{total} = {n444/total*100:.1f}%')

# does qrot help AT ALL on top of best fg83-only?
print('\n--- isolate qrot contribution at fixed fg83=0.05 ---')
for q in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
    a = acc(cnxxl + 0.05 * fg83 + q * qrot_z2)
    print(f'  fg83=0.05 qr={q:<5} -> {a}/{N}')
