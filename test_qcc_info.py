"""Information diagnostic: train shallow classifier on QCC features alone.
Tests: per-finger Kabsch quaternions (5x4 per frame pair) → 25 gesture classes.
If chance (4%), QCC carries no signal. If >20%, signal exists.
"""
import os, re, numpy as np, sys
import torch, torch.nn as nn

TARGETS = '/notebooks/PMamba/dataset/Nvidia/Processed/finger_quat_targets.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'

def parse_annot(path):
    samples = {}
    with open(path) as f:
        for line in f:
            m_path = re.search(r'path:(\S+)', line)
            m_label = re.search(r'label:(\d+)', line)
            if m_path and m_label:
                samples[m_path.group(1)] = int(m_label.group(1)) - 1
    return samples

train_lbl = parse_annot(f'{ANNOT_ROOT}/nvgesture_train_correct_cvpr2016_v2.lst')
test_lbl = parse_annot(f'{ANNOT_ROOT}/nvgesture_test_correct_cvpr2016_v2.lst')
print(f'train annot: {len(train_lbl)}, test annot: {len(test_lbl)}')

print('loading targets...')
targets = dict(np.load(TARGETS, allow_pickle=False))
print(f'{len(targets)} samples in targets')

# Resample each (T-1, 5, 4) → fixed (T_fixed, 5, 4)
T_FIXED = 32
def resample(arr):
    if arr.shape[0] < 2:
        return None
    idx = np.linspace(0, arr.shape[0]-1, T_FIXED).astype(np.int64)
    return arr[idx].reshape(-1)  # (T_FIXED * 5 * 4,) = 640

X_tr, y_tr, X_te, y_te = [], [], [], []
miss = 0
for k, v in targets.items():
    feat = resample(v)
    if feat is None: continue
    if k in train_lbl:
        X_tr.append(feat); y_tr.append(train_lbl[k])
    elif k in test_lbl:
        X_te.append(feat); y_te.append(test_lbl[k])
    else:
        miss += 1
print(f'train: {len(X_tr)}, test: {len(X_te)}, missing: {miss}')
X_tr = np.array(X_tr, dtype=np.float32); y_tr = np.array(y_tr, dtype=np.int64)
X_te = np.array(X_te, dtype=np.float32); y_te = np.array(y_te, dtype=np.int64)
print(f'X_tr shape: {X_tr.shape}, y unique: {len(np.unique(y_tr))}')

# Standardize
mean = X_tr.mean(0); std = X_tr.std(0) + 1e-7
X_tr = (X_tr - mean) / std; X_te = (X_te - mean) / std

# Logistic regression
from sklearn.linear_model import LogisticRegression
print('\n--- Logistic Regression ---')
clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
clf.fit(X_tr, y_tr)
print(f'  train acc: {clf.score(X_tr, y_tr)*100:.2f}%')
print(f'   test acc: {clf.score(X_te, y_te)*100:.2f}%  (chance = 4.00%)')

# 2-layer MLP
print('\n--- 2-layer MLP ---')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mlp = nn.Sequential(
    nn.Linear(X_tr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 25),
).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
Xt = torch.from_numpy(X_tr).to(device); yt = torch.from_numpy(y_tr).to(device)
Xv = torch.from_numpy(X_te).to(device); yv = torch.from_numpy(y_te).to(device)
best_te = 0
for ep in range(100):
    mlp.train()
    perm = torch.randperm(len(Xt))
    losses = []
    for i in range(0, len(Xt), 64):
        idx = perm[i:i+64]
        logits = mlp(Xt[idx])
        loss = nn.functional.cross_entropy(logits, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    mlp.eval()
    with torch.no_grad():
        tr_acc = (mlp(Xt).argmax(1) == yt).float().mean().item() * 100
        te_acc = (mlp(Xv).argmax(1) == yv).float().mean().item() * 100
    if te_acc > best_te: best_te = te_acc
    if ep % 10 == 0 or ep == 99:
        print(f'  ep {ep:3d}: loss={np.mean(losses):.4f}  tr={tr_acc:.2f}  te={te_acc:.2f}  best_te={best_te:.2f}')
print(f'\n>>> SHALLOW MLP BEST TEST: {best_te:.2f}%  (chance = 4.00%) <<<')
