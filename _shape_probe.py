import sys; sys.path.insert(0, '.')
import numpy as np
from scipy.spatial import cKDTree
from nvidia_dataloader import NvidiaLoader
from collections import Counter

SS = 172; K = 16


def sample_clip(arr):                 # (T,512,4) -> (T*SS,3)
    T, N, _ = arr.shape
    idx = np.linspace(0, N - 1, min(SS, N)).astype(int)
    return arr[:, idx, :3].reshape(-1, 3).astype(np.float64)


def dim_feats(pts):                   # Weinmann local-covariance dimensionality
    M = len(pts)
    tr = cKDTree(pts)
    _, nn = tr.query(pts, k=min(K, M))
    nb = pts[nn]                                   # (M,K,3)
    c = nb - nb.mean(1, keepdims=True)
    cov = np.einsum('mki,mkj->mij', c, c) / nb.shape[1]
    w = np.linalg.eigvalsh(cov)                    # ascending
    e = w[:, ::-1]                                 # e1>=e2>=e3
    s = e.sum(1) + 1e-12
    en = e / s[:, None]
    e1, e2, e3 = en[:, 0], en[:, 1], en[:, 2]
    lin = (e1 - e2) / (e1 + 1e-9)
    pla = (e2 - e3) / (e1 + 1e-9)
    sph = e3 / (e1 + 1e-9)
    omn = np.cbrt(np.clip(e1 * e2 * e3, 0, None))
    ani = (e1 - e3) / (e1 + 1e-9)
    ent = -(en * np.log(en + 1e-9)).sum(1)
    cur = e3
    F = np.stack([lin, pla, sph, omn, ani, ent, cur], 1)   # (M,7)
    return np.concatenate([F.mean(0), F.std(0)])           # (14,)


print('loading...', flush=True)
te = NvidiaLoader(framerate=32, phase='test', datatype='depth')
te_arr = te.tensor.numpy(); te_lab = te.labels_tensor.numpy()
trn = NvidiaLoader(framerate=32, phase='train', datatype='depth')
trn_arr = trn.tensor.numpy(); trn_lab = trn.labels_tensor.numpy()
d = np.load('quat9108_dump.npz'); trues, preds = d['trues'], d['preds']
assert np.array_equal(trues, te_lab)

print('descriptors...', flush=True)
Xtr = np.stack([dim_feats(sample_clip(a)) for a in trn_arr]); ytr = trn_lab
Xte = np.stack([dim_feats(sample_clip(a)) for a in te_arr]); yte = trues

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

wrong = np.where(trues != preds)[0]
pairs = Counter(tuple(sorted((int(trues[i]), int(preds[i])))) for i in wrong)
print('\n#wrong=%d  pairwise AUC (dimensionality feats):' % len(wrong), flush=True)
for (a, b), c in pairs.most_common(8):
    m = np.isin(yte, [a, b])
    if m.sum() < 6:
        print('  (%d,%d) n<6' % (a, b)); continue
    yb = (yte[m] == a).astype(int)
    try:
        auc = cross_val_score(LogisticRegression(max_iter=400),
                              StandardScaler().fit_transform(Xte[m]), yb,
                              cv=5, scoring='roc_auc').mean()
    except Exception:
        auc = float('nan')
    print('  (%2d,%2d) x%d  AUC=%.2f' % (a, b, c, auc), flush=True)

sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=600).fit(sc.transform(Xtr), ytr)
proba = clf.predict_proba(sc.transform(Xte)); col = {c: i for i, c in enumerate(clf.classes_)}
hit = tot = 0
for i in wrong:
    t, p = int(trues[i]), int(preds[i])
    if t in col and p in col:
        tot += 1; hit += int(proba[i, col[t]] > proba[i, col[p]])
print('\nTest B oracle tie-break: %d/%d = %.1f%% (chance=50)' % (hit, tot, 100 * hit / max(tot, 1)), flush=True)
fp_acc = (clf.predict(sc.transform(Xte)) == yte).mean() * 100
print('dimensionality alone 25-way = %.1f%% (backbone 91.1)' % fp_acc, flush=True)
print('DONE', flush=True)
