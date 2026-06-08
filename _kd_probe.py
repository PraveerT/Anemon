import sys; sys.path.insert(0, '.')
import numpy as np
from nvidia_dataloader import NvidiaLoader
from collections import Counter

L = 6                       # kd-tree depth for split-direction histogram
SS = 172                    # match model's deterministic sampling


def sample_clip(arr):       # arr (T,512,4)-> (T*SS,3) using model's linspace idx
    T, N, _ = arr.shape
    idx = np.linspace(0, N - 1, min(SS, N)).astype(int)
    return arr[:, idx, :3].reshape(-1, 3)


def kd_dirhist(pts, L=L):
    dc = np.zeros((L, 3))
    nodes = [pts]
    for lvl in range(L):
        nxt = []
        for nd in nodes:
            if len(nd) < 2:
                nxt += [nd[:0], nd[:0]]; continue
            ax = int(np.argmax(nd.max(0) - nd.min(0)))
            dc[lvl, ax] += 1
            med = np.median(nd[:, ax])
            nxt += [nd[nd[:, ax] <= med], nd[nd[:, ax] > med]]
        nodes = nxt
    return (dc / (dc.sum(1, keepdims=True) + 1e-9)).reshape(-1)   # (3L,)


def skew3(pts):
    m = pts.mean(0); s = pts.std(0) + 1e-9
    return (((pts - m) / s) ** 3).mean(0)                          # (3,) u,v,d


def fp(arr):
    p = sample_clip(arr)
    return np.concatenate([kd_dirhist(p), skew3(p)])               # (3L+3,)


print('loading clouds...', flush=True)
tr = NvidiaLoader(framerate=32, phase='test', datatype='depth')   # warms cache
tr_arr = tr.tensor.numpy()                                        # actually test
te_arr = tr_arr
te_lab = tr.labels_tensor.numpy()
trn = NvidiaLoader(framerate=32, phase='train', datatype='depth')
trn_arr = trn.tensor.numpy(); trn_lab = trn.labels_tensor.numpy()

d = np.load('quat9108_dump.npz')
trues, preds = d['trues'], d['preds']
assert np.array_equal(trues, te_lab), 'order mismatch'

print('building fingerprints...', flush=True)
Xtr = np.stack([fp(a) for a in trn_arr]); ytr = trn_lab
Xte = np.stack([fp(a) for a in te_arr]); yte = trues
DIR = slice(0, 3 * L); SKW = slice(3 * L, 3 * L + 3)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ---- Test A: pairwise separability on test (5-fold AUC), top confusion pairs
wrong = np.where(trues != preds)[0]
pairs = Counter(tuple(sorted((int(trues[i]), int(preds[i])))) for i in wrong)
print('\n#wrong=%d  top confusion pairs:' % len(wrong), flush=True)
for (a, b), c in pairs.most_common(8):
    m = np.isin(yte, [a, b])
    if m.sum() < 6:
        print('  (%d,%d) n<6' % (a, b)); continue
    yb = (yte[m] == a).astype(int)
    aucs = {}
    for name, sl in [('dir', DIR), ('skew', SKW), ('all', slice(None))]:
        try:
            s = cross_val_score(LogisticRegression(max_iter=400),
                                StandardScaler().fit_transform(Xte[m][:, sl]),
                                yb, cv=5, scoring='roc_auc')
            aucs[name] = s.mean()
        except Exception:
            aucs[name] = float('nan')
    print('  (%2d,%2d) x%d  AUC dir=%.2f skew=%.2f all=%.2f'
          % (a, b, c, aucs['dir'], aucs['skew'], aucs['all']), flush=True)

# ---- Test B (decisive): fingerprint classifier trained on TRAIN, oracle
#      tie-break on backbone's WRONG samples restricted to {true,pred}.
print('\nTest B: oracle tie-break on the %d wrong samples (chance=50%%)' % len(wrong), flush=True)
for name, sl in [('dir', DIR), ('skew', SKW), ('all', slice(None))]:
    sc = StandardScaler().fit(Xtr[:, sl])
    clf = LogisticRegression(max_iter=600, C=1.0)
    clf.fit(sc.transform(Xtr[:, sl]), ytr)
    proba = clf.predict_proba(sc.transform(Xte[:, sl]))
    cls = clf.classes_
    col = {c: i for i, c in enumerate(cls)}
    hit = 0; tot = 0
    for i in wrong:
        t, p = int(trues[i]), int(preds[i])
        if t not in col or p not in col:
            continue
        tot += 1
        hit += int(proba[i, col[t]] > proba[i, col[p]])
    print('  %-4s tie-break recovers true on %d/%d = %.1f%%'
          % (name, hit, tot, 100.0 * hit / max(tot, 1)), flush=True)

# ---- also: standalone fingerprint full-25-way test acc (context)
sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=600).fit(sc.transform(Xtr), ytr)
fp_acc = (clf.predict(sc.transform(Xte)) == yte).mean() * 100
print('\nfingerprint alone 25-way test acc = %.1f%% (backbone=%.1f%%)'
      % (fp_acc, (trues == preds).mean() * 100), flush=True)
print('DONE', flush=True)
