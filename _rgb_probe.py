"""Is RGB error-orthogonal to the depth backbone? Extract pretrained-ResNet18
features from sk_color.avi per clip (8 frames, mean-pooled), then test whether RGB
separates the depth 91.08 backbone's 43 errors. If yes -> RGB recovers depth's
blind spot (the asymmetric-fusion novelty is earned). If chance -> gestures are
intrinsically confusable across modalities. Train order matches test_depth_list.
"""
import sys, os; sys.path.insert(0, '.')
import numpy as np, torch, cv2
import torchvision as tv
from collections import Counter

dev = 'cuda'
RAW = '/notebooks/Manta/dataset_full/nvGesture_v1.1/nvGesture_v1/Video_data'
NF = 8
net = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
net.fc = torch.nn.Identity(); net = net.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def clip_dir(line):                 # -> 'class_01/subject13_r0'
    p = line.split('\t')[1] if '\t' in line else line.split()[1]
    parts = p.split('/')
    i = [j for j, s in enumerate(parts) if s.startswith('class_')][0]
    return parts[i] + '/' + parts[i + 1]


def feat_clip(cd):
    avi = os.path.join(RAW, cd, 'sk_color.avi')
    cap = cv2.VideoCapture(avi)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        cap.release(); return None
    idx = np.linspace(0, n - 1, NF).astype(int)
    frames = []
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        if i in idx:
            fr = cv2.cvtColor(cv2.resize(fr, (224, 224)), cv2.COLOR_BGR2RGB)
            frames.append(fr)
    cap.release()
    if not frames: return None
    x = torch.from_numpy(np.stack(frames)).float().to(dev).permute(0, 3, 1, 2) / 255.
    x = (x - MEAN) / STD
    with torch.no_grad():
        f = net(x)                  # (NF,512)
    return f.mean(0).cpu().numpy()


def load(listfile):
    lines = [l for l in open(listfile) if l.strip()]
    X, y = [], []
    for l in lines:
        cd = clip_dir(l)
        lab = int((l.split('\t') if '\t' in l else l.split())[2])
        f = feat_clip(cd)
        X.append(f if f is not None else np.zeros(512, np.float32)); y.append(lab)
    return np.stack(X), np.array(y)


pre = '/notebooks/wt9191/dataset/Nvidia/Processed'
print('extracting train RGB...', flush=True)
Xtr, ytr = load(pre + '/train_depth_list.txt')
print('extracting test RGB...', flush=True)
Xte, yte = load(pre + '/test_depth_list.txt')
d = np.load('quat9108_dump.npz'); trues, preds = d['trues'], d['preds']
print('order match:', bool(np.array_equal(yte, trues)), 'Xte', Xte.shape, flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
wrong = np.where(trues != preds)[0]
pairs = Counter(tuple(sorted((int(trues[i]), int(preds[i])))) for i in wrong)
print('\n#wrong=%d  RGB pairwise AUC:' % len(wrong), flush=True)
for (a, b), c in pairs.most_common(8):
    m = np.isin(yte, [a, b])
    if m.sum() < 6: continue
    yb = (yte[m] == a).astype(int)
    try:
        auc = cross_val_score(LogisticRegression(max_iter=500),
                              StandardScaler().fit_transform(Xte[m]), yb, cv=5, scoring='roc_auc').mean()
    except Exception:
        auc = float('nan')
    print('  (%2d,%2d) x%d  AUC=%.2f' % (a, b, c, auc), flush=True)

sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=800).fit(sc.transform(Xtr), ytr)
proba = clf.predict_proba(sc.transform(Xte)); col = {c: i for i, c in enumerate(clf.classes_)}
hit = tot = 0
for i in wrong:
    t, p = int(trues[i]), int(preds[i])
    if t in col and p in col:
        tot += 1; hit += int(proba[i, col[t]] > proba[i, col[p]])
print('\nTest B oracle tie-break: %d/%d = %.1f%% (chance 50)' % (hit, tot, 100*hit/max(tot,1)), flush=True)
print('RGB-alone 25-way test acc = %.1f%% (depth backbone 91.1)'
      % ((clf.predict(sc.transform(Xte)) == yte).mean()*100), flush=True)
print('DONE', flush=True)
