"""Step 5: Temporal ablation battery — test if NVGesture is really a motion task
or mostly static pose recognition. Use same shallow MLP framework as step 2.

Tests:
  A) Single-frame baselines (frame 0, mid, last, mean-pool)
  B) Time-shuffled sequences (frames permuted)
  C) Frame-count sweep (N=1,2,4,8,16,32)
  D) Time-reversed (train forward, test reverse)
"""
import os, re, time, numpy as np, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'

def parse_annot(path):
    out = {}
    with open(path) as f:
        for line in f:
            mp_ = re.search(r'path:(\S+)', line); ml = re.search(r'label:(\d+)', line)
            if mp_ and ml: out[mp_.group(1)] = int(ml.group(1)) - 1
    return out

train_lbl = parse_annot(f'{ANNOT_ROOT}/nvgesture_train_correct_cvpr2016_v2.lst')
test_lbl  = parse_annot(f'{ANNOT_ROOT}/nvgesture_test_correct_cvpr2016_v2.lst')
print('loading landmarks...')
sk = dict(np.load(SK, allow_pickle=False))

def fillnan(arr):
    valid = np.isfinite(arr[..., 0]).all(axis=-1)
    last = None; out = arr.copy()
    for t in range(out.shape[0]):
        if valid[t]: last = out[t]
        elif last is not None: out[t] = last
    for t in range(out.shape[0]):
        if not np.isfinite(out[t]).all():
            for t2 in range(t+1, out.shape[0]):
                if valid[t2]: out[t] = out[t2]; break
            else: out[t] = 0
    return out

def resample(x, T_target):
    if len(x) < 2: return None
    idx = np.linspace(0, len(x)-1, T_target).astype(np.int64)
    return x[idx]

def shallow(X_tr, y_tr, X_te, y_te, ep_max=80, seed=0):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = X_tr.shape[1]
    if D == 0: return 0.0, None
    mlp = nn.Sequential(
        nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 25),
    ).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X_tr).to(device); yt = torch.from_numpy(y_tr).to(device)
    Xv = torch.from_numpy(X_te).to(device); yv = torch.from_numpy(y_te).to(device)
    best_te = 0
    best_state = None
    for ep in range(ep_max):
        mlp.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]
            loss = nn.functional.cross_entropy(mlp(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            te_acc = (mlp(Xv).argmax(1) == yv).float().mean().item() * 100
        if te_acc > best_te:
            best_te = te_acc
            best_state = {k:v.detach().clone() for k,v in mlp.state_dict().items()}
    return best_te, best_state

# ============ helpers ============
def build_dataset(transform):
    """transform(lm_filled) -> (D,) flat feature vector per sample."""
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for k, lm_raw in sk.items():
        if lm_raw.shape[0] < 4: continue
        lm = fillnan(lm_raw)
        x = transform(lm)
        if x is None or not np.isfinite(x).all(): continue
        x = x.astype(np.float32).reshape(-1)
        if k in train_lbl: X_tr.append(x); y_tr.append(train_lbl[k])
        elif k in test_lbl: X_te.append(x); y_te.append(test_lbl[k])
    X_tr = np.array(X_tr, dtype=np.float32); y_tr = np.array(y_tr, dtype=np.int64)
    X_te = np.array(X_te, dtype=np.float32); y_te = np.array(y_te, dtype=np.int64)
    mean = X_tr.mean(0); std = X_tr.std(0) + 1e-7
    return (X_tr - mean) / std, y_tr, (X_te - mean) / std, y_te

# ============ A) single frame ============
print('\n=== A) Single-frame baselines ===')
def get_frame(lm, idx_frac):
    T = lm.shape[0]
    return lm[int((T-1) * idx_frac), :, :]
def get_mean(lm): return lm.mean(axis=0)

specs_A = [
    ('frame_0',           lambda lm: get_frame(lm, 0.0)),
    ('frame_mid',         lambda lm: get_frame(lm, 0.5)),
    ('frame_last',        lambda lm: get_frame(lm, 1.0)),
    ('frame_quarter',     lambda lm: get_frame(lm, 0.25)),
    ('frame_3quarter',    lambda lm: get_frame(lm, 0.75)),
    ('mean_pool_all',     get_mean),
]
results_A = {}
for name, tf in specs_A:
    Xtr, ytr, Xte, yte = build_dataset(tf)
    acc, _ = shallow(Xtr, ytr, Xte, yte)
    results_A[name] = acc
    print(f'  {name:<20} dim={Xtr.shape[1]:>4}  acc={acc:6.2f}  vs_chance={acc/4.0:.1f}x', flush=True)

# Best-of-3 ensemble (frame 0, mid, last)
print('\n=== A2) Ensemble of single-frame classifiers ===')
def get_probs(lm_filler, frac, X_te_ref):
    Xtr, ytr, Xte, yte = build_dataset(lambda lm: get_frame(lm, frac))
    acc, state = shallow(Xtr, ytr, Xte, yte)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 25),
    ).to(device)
    mlp.load_state_dict(state); mlp.eval()
    with torch.no_grad():
        p = torch.softmax(mlp(torch.from_numpy(Xte).to(device)), -1).cpu().numpy()
    return p, yte

p_lst = []
for frac in [0.0, 0.5, 1.0]:
    p, yte = get_probs(None, frac, None)
    p_lst.append(p)
ens = sum(p_lst) / 3
ens_acc = (ens.argmax(1) == yte).mean() * 100
print(f'  3-frame ensemble (start+mid+end): acc={ens_acc:.2f}  vs_chance={ens_acc/4.0:.1f}x')

# ============ B) Time-shuffled sequences ============
print('\n=== B) Time-shuffled (frames permuted) ===')
T_FIXED = 32
def shuffled(lm, seed):
    rng = np.random.RandomState(seed)
    feat = resample(lm, T_FIXED).reshape(T_FIXED, -1)
    perm = rng.permutation(T_FIXED)
    return feat[perm].reshape(-1)

# Same shuffle every sample (consistent permutation), and per-sample shuffle
print('  -- consistent permutation across samples --')
def shuf_consistent(lm):
    feat = resample(lm, T_FIXED).reshape(T_FIXED, -1)
    rng = np.random.RandomState(123)
    perm = rng.permutation(T_FIXED)
    return feat[perm].reshape(-1)
Xtr, ytr, Xte, yte = build_dataset(shuf_consistent)
acc, _ = shallow(Xtr, ytr, Xte, yte)
print(f'  shuffle_consistent dim={Xtr.shape[1]:>5}  acc={acc:6.2f}  vs_chance={acc/4.0:.1f}x')

print('  -- per-sample random permutation (different seed per sample) --')
def shuf_random(lm):
    seed = hash(lm.tobytes()) & 0xffff
    return shuffled(lm, seed)
Xtr, ytr, Xte, yte = build_dataset(shuf_random)
acc, _ = shallow(Xtr, ytr, Xte, yte)
print(f'  shuffle_persample dim={Xtr.shape[1]:>5}  acc={acc:6.2f}  vs_chance={acc/4.0:.1f}x')

# ============ C) Frame-count sweep ============
print('\n=== C) Frame-count sweep ===')
def n_frames(lm, N):
    return resample(lm, N).reshape(-1)
for N in [1, 2, 4, 8, 16, 32]:
    Xtr, ytr, Xte, yte = build_dataset(lambda lm, N=N: n_frames(lm, N))
    acc, _ = shallow(Xtr, ytr, Xte, yte)
    print(f'  N={N:>3}  dim={Xtr.shape[1]:>5}  acc={acc:6.2f}  vs_chance={acc/4.0:.1f}x')

# ============ D) Time-reversed test ============
print('\n=== D) Train forward, test reversed ===')
def get_xy_dual():
    Xfwd_tr, Xrev_tr, ytr, Xfwd_te, Xrev_te, yte = [], [], [], [], [], []
    for k, lm_raw in sk.items():
        if lm_raw.shape[0] < 4: continue
        lm = fillnan(lm_raw)
        feat = resample(lm, T_FIXED).reshape(T_FIXED, -1)
        fwd = feat.reshape(-1).astype(np.float32)
        rev = feat[::-1].reshape(-1).astype(np.float32)
        if k in train_lbl: Xfwd_tr.append(fwd); Xrev_tr.append(rev); ytr.append(train_lbl[k])
        elif k in test_lbl: Xfwd_te.append(fwd); Xrev_te.append(rev); yte.append(test_lbl[k])
    Xfwd_tr = np.array(Xfwd_tr, np.float32); ytr = np.array(ytr, np.int64)
    Xrev_tr = np.array(Xrev_tr, np.float32)
    Xfwd_te = np.array(Xfwd_te, np.float32); yte = np.array(yte, np.int64)
    Xrev_te = np.array(Xrev_te, np.float32)
    mean = Xfwd_tr.mean(0); std = Xfwd_tr.std(0) + 1e-7
    return (Xfwd_tr-mean)/std, (Xrev_tr-mean)/std, ytr, (Xfwd_te-mean)/std, (Xrev_te-mean)/std, yte
Xfwd_tr, Xrev_tr, ytr, Xfwd_te, Xrev_te, yte = get_xy_dual()

acc_fwd, state_fwd = shallow(Xfwd_tr, ytr, Xfwd_te, yte)
print(f'  forward train -> forward test:  {acc_fwd:.2f}')

# Now load same model, test on reversed test set
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mlp = nn.Sequential(
    nn.Linear(Xfwd_tr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 25),
).to(device)
mlp.load_state_dict(state_fwd); mlp.eval()
with torch.no_grad():
    pred_rev = mlp(torch.from_numpy(Xrev_te).to(device)).argmax(1).cpu().numpy()
    acc_rev = (pred_rev == yte).mean() * 100
print(f'  forward train -> REVERSED test: {acc_rev:.2f}  (drop = {acc_fwd-acc_rev:.2f})')

# Symmetric: train reversed
acc_rev_train, _ = shallow(Xrev_tr, ytr, Xrev_te, yte)
print(f'  reverse train -> reverse test:  {acc_rev_train:.2f}')

# ============ Summary ============
print('\n' + '='*55)
print('SUMMARY (chance = 4.00%, baseline lm_raw=62.03)')
print('='*55)
print(f'A) Single-frame best (mid):    {results_A.get("frame_mid", 0):.2f}')
print(f'A) Mean-pool all frames:       {results_A.get("mean_pool_all", 0):.2f}')
print(f'A) 3-frame ensemble:           {ens_acc:.2f}')
print(f'B) Time-shuffled per-sample:   (see above, shuffle_persample)')
print(f'C) Frame-count plateau:        (see above)')
print(f'D) Forward->Forward:           {acc_fwd:.2f}')
print(f'D) Forward->Reversed:          {acc_rev:.2f}')
print(f'D) Reversed-only training:     {acc_rev_train:.2f}')
