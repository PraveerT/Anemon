"""Test Method B (Learnable Riemannian Metric) and Method C (Implicit Configuration
Field) under fair comparison: same encoder, different classification heads.

Encoder E: landmark sequence -> 128-dim embedding (standard MLP)
Heads:
  Linear (baseline):   logits = W z + b
  Method B:            logits_c = - || M(z) (z - p_c) ||^2  with M = learnable matrix
                                   parametrized by another MLP M_phi(z)
                       (Riemannian/Mahalanobis-style with position-dependent metric)
  Method C:            logits_c = f_psi(z, e_c)  via conditional MLP, where e_c are
                       learnable class embeddings (implicit configuration field on manifold)

Each trained 80 epochs, same data, same encoder hyperparams.
"""
import os, re, time, numpy as np
import torch, torch.nn as nn

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
T_FIXED = 32
NUM_CLASSES = 25
D_EMB = 128
NUM_EPOCHS = 100
BS = 64
LR = 1e-3
WD = 1e-4

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

print('precomputing landmark features...')
def resample(x, T):
    idx = np.linspace(0, len(x)-1, T).astype(np.int64)
    return x[idx]

X_tr, y_tr, X_te, y_te = [], [], [], []
for k, lm_raw in sk.items():
    if lm_raw.shape[0] < 4: continue
    lm = fillnan(lm_raw)
    feat = resample(lm, T_FIXED).reshape(-1).astype(np.float32)
    if k in train_lbl: X_tr.append(feat); y_tr.append(train_lbl[k])
    elif k in test_lbl: X_te.append(feat); y_te.append(test_lbl[k])
X_tr = np.array(X_tr); y_tr = np.array(y_tr, dtype=np.int64)
X_te = np.array(X_te); y_te = np.array(y_te, dtype=np.int64)
mean = X_tr.mean(0); std = X_tr.std(0) + 1e-7
X_tr = (X_tr - mean) / std; X_te = (X_te - mean) / std
D_IN = X_tr.shape[1]
print(f'D_in={D_IN}  train {len(X_tr)} test {len(X_te)}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Xt = torch.from_numpy(X_tr).to(device); yt = torch.from_numpy(y_tr).to(device)
Xv = torch.from_numpy(X_te).to(device); yv = torch.from_numpy(y_te).to(device)


class Encoder(nn.Module):
    def __init__(self, d_in, d_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, d_emb),
        )
    def forward(self, x): return self.net(x)


class HeadLinear(nn.Module):
    def __init__(self, d_emb, n_class):
        super().__init__()
        self.fc = nn.Linear(d_emb, n_class)
    def forward(self, z): return self.fc(z)


class HeadRiemannian(nn.Module):
    """Method B: position-dependent metric tensor + class prototypes.
    Per input z, predict matrix M(z) (D x D, but rank-K via M = U V^T with U,V learned
    from z). Distance to class prototype p_c: || M(z) (z - p_c) ||^2.
    Logits = -dist (so closer = higher score). Cross-entropy on logits.
    """
    def __init__(self, d_emb, n_class, rank=16):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_class, d_emb) * 0.1)
        self.metric_net = nn.Sequential(
            nn.Linear(d_emb, 128), nn.GELU(),
            nn.Linear(128, rank * d_emb),
        )
        self.rank = rank; self.d = d_emb
    def forward(self, z):
        # z: (B, d)
        M = self.metric_net(z).reshape(-1, self.rank, self.d)  # (B, K, D)
        # Distances per class: || M (z - p_c) ||^2 for each c
        diffs = z.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (B, C, D)
        # M @ diff for each (b, c): einsum
        Md = torch.einsum('bkd,bcd->bck', M, diffs)  # (B, C, K)
        dist2 = (Md ** 2).sum(-1)  # (B, C)
        return -dist2  # logits


class HeadImplicit(nn.Module):
    """Method C: implicit configuration field. Per class c, evaluate f_psi(z, e_c).
    e_c are learnable class embeddings. f_psi is a small MLP that takes
    [z, e_c] -> score scalar.
    """
    def __init__(self, d_emb, n_class, d_class=64):
        super().__init__()
        self.class_emb = nn.Parameter(torch.randn(n_class, d_class) * 0.1)
        self.field = nn.Sequential(
            nn.Linear(d_emb + d_class, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
    def forward(self, z):
        B = z.shape[0]; C = self.class_emb.shape[0]
        z_exp = z.unsqueeze(1).expand(-1, C, -1)  # (B, C, d)
        c_exp = self.class_emb.unsqueeze(0).expand(B, -1, -1)  # (B, C, d_class)
        h = torch.cat([z_exp, c_exp], dim=-1).reshape(B * C, -1)
        scores = self.field(h).reshape(B, C)
        return scores


class Pipeline(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.encoder = Encoder(D_IN, D_EMB)
        self.head = head
    def forward(self, x): return self.head(self.encoder(x))


def train_eval(name, head, ep_max=NUM_EPOCHS, seed=42):
    torch.manual_seed(seed)
    model = Pipeline(head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_max)
    best_te = 0; best_ep = 0
    for ep in range(ep_max):
        model.train()
        perm = torch.randperm(len(Xt))
        losses = []
        for i in range(0, len(Xt), BS):
            idx = perm[i:i+BS]
            logits = model(Xt[idx])
            loss = nn.functional.cross_entropy(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            te_acc = (model(Xv).argmax(1) == yv).float().mean().item() * 100
        if te_acc > best_te: best_te = te_acc; best_ep = ep
    print(f'{name:<35} params={n_params:>7,}  best={best_te:.2f}@ep{best_ep}', flush=True)
    return best_te

print(f'\n{"head":<35} {"params":>14}  best')
print('-' * 70)
acc_lin  = train_eval('Linear (baseline)',          HeadLinear(D_EMB, NUM_CLASSES))
acc_riem = train_eval('Method B: Riemannian metric', HeadRiemannian(D_EMB, NUM_CLASSES, rank=16))
acc_imp  = train_eval('Method C: Implicit field',    HeadImplicit(D_EMB, NUM_CLASSES, d_class=64))

print(f'\n--- summary ---')
print(f'Linear:        {acc_lin:.2f}')
print(f'Riemannian:    {acc_riem:.2f}  ({acc_riem-acc_lin:+.2f} vs linear)')
print(f'Implicit:      {acc_imp:.2f}  ({acc_imp-acc_lin:+.2f} vs linear)')
print(f'\nReference shallow MLP (no encoder, raw): 62.03')
