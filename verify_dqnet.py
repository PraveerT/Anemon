"""Verify DQNet-v2 result is real:
  1. Reload saved best_model, re-eval on test set
  2. Check train/test split has no leakage (subjects disjoint)
  3. Check the saved best_probs.npz matches re-eval
"""
import os, re, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
T_FIXED = 32
BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
B = 20

def parse_annot(path):
    out = {}
    with open(path) as f:
        for line in f:
            mp_ = re.search(r'path:(\S+)', line); ml = re.search(r'label:(\d+)', line)
            if mp_ and ml: out[mp_.group(1)] = int(ml.group(1)) - 1
    return out

train_lbl = parse_annot(f'{ANNOT_ROOT}/nvgesture_train_correct_cvpr2016_v2.lst')
test_lbl  = parse_annot(f'{ANNOT_ROOT}/nvgesture_test_correct_cvpr2016_v2.lst')

# === Check 1: subjects disjoint ===
def subjects(d):
    out = set()
    for k in d:
        m = re.search(r'subject(\S+?)_', k)
        if m: out.add(m.group(1))
    return out
sub_tr = subjects(train_lbl)
sub_te = subjects(test_lbl)
print(f'Train subjects: {sorted(sub_tr)}')
print(f'Test subjects:  {sorted(sub_te)}')
overlap = sub_tr & sub_te
print(f'Subject overlap: {sorted(overlap)}')
print(f'  -> {"DISJOINT (good)" if not overlap else f"OVERLAP ({len(overlap)} shared)"}')

# === Check 2: sample paths disjoint ===
key_tr = set(train_lbl); key_te = set(test_lbl)
key_overlap = key_tr & key_te
print(f'Sample-key overlap: {len(key_overlap)} (should be 0)')

# === Check 3: re-encode test set + re-eval ===
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

def vec_to_quat_np(V):
    n = np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9
    u = V / n
    cos_h = np.clip((1 + u[..., 2:3]) * 0.5, 1e-9, 1.0)
    w = np.sqrt(cos_h)
    sin_h = np.sqrt(np.clip(1 - cos_h, 0, 1))
    axis = np.zeros_like(u); axis[..., 0] = -u[..., 1]; axis[..., 1] = u[..., 0]
    s = np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9
    return np.concatenate([w, axis/s*sin_h], axis=-1).astype(np.float32)

def qmul_np(p, q):
    pw,px,py,pz = p[...,0],p[...,1],p[...,2],p[...,3]
    qw,qx,qy,qz = q[...,0],q[...,1],q[...,2],q[...,3]
    return np.stack([pw*qw-px*qx-py*qy-pz*qz, pw*qx+px*qw+py*qz-pz*qy,
                      pw*qy-px*qz+py*qw+pz*qx, pw*qz+px*qy-py*qx+pz*qw], axis=-1)

def encode_sample(lm):
    T = lm.shape[0]
    out = np.zeros((T, B, 8), dtype=np.float32)
    for b, (p, c) in enumerate(BONES):
        bone_vec = lm[:, c, :] - lm[:, p, :]
        rot_q = vec_to_quat_np(bone_vec)
        mid = (lm[:, c, :] + lm[:, p, :]) / 2
        t_pure = np.concatenate([np.zeros((T, 1), dtype=np.float32), mid], axis=-1)
        pos_q = qmul_np(t_pure, rot_q) * 0.5
        out[:, b, :4] = rot_q
        out[:, b, 4:] = pos_q
    return out

print('\nRe-encoding test set...')
X_te, y_te, k_te = [], [], []
for k, lbl in test_lbl.items():
    if k not in sk: continue
    lm_raw = sk[k]
    if lm_raw.shape[0] < 4: continue
    lm = fillnan(lm_raw)
    dq = encode_sample(lm)
    idx = np.linspace(0, dq.shape[0]-1, T_FIXED).astype(np.int64)
    X_te.append(dq[idx]); y_te.append(lbl); k_te.append(k)
X_te = np.array(X_te); y_te = np.array(y_te, dtype=np.int64)
print(f'test re-encoded: {len(X_te)}')

class DQNet_Transformer(nn.Module):
    def __init__(self, d_model=192, n_heads=8, n_layers=4, num_classes=25, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Linear(B * 8, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, T_FIXED + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        b = x.shape[0]
        x = x.flatten(2)
        x = self.in_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_emb
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DQNet_Transformer().to(device).eval()
ck = torch.load('/notebooks/PMamba/experiments/work_dir/dqnet_v2/best_model.pt', map_location='cpu')
model.load_state_dict(ck['model_state_dict'])
print(f'Loaded best from epoch {ck["epoch"]}')

with torch.no_grad():
    X = torch.from_numpy(X_te).to(device)
    Y = torch.from_numpy(y_te).to(device)
    preds, probs = [], []
    for i in range(0, len(X), 32):
        L = model(X[i:i+32])
        preds.append(L.argmax(1))
        probs.append(F.softmax(L, dim=-1))
    preds = torch.cat(preds); probs = torch.cat(probs)
    acc = (preds == Y).float().mean().item() * 100
print(f'\nRE-EVAL test acc: {acc:.2f}%')

# Compare with saved probs
saved = np.load('/notebooks/PMamba/experiments/work_dir/dqnet_v2/best_probs.npz')
saved_acc = (saved['probs'].argmax(1) == saved['labels']).mean() * 100
print(f'Saved best_probs.npz acc: {saved_acc:.2f}%')
print(f'Saved labels match re-eval labels: {(saved["labels"] == y_te).all()}')

# Sanity check: dummy random prediction
np.random.seed(0)
random_pred_acc = (np.random.randint(0, 25, size=len(y_te)) == y_te).mean() * 100
print(f'Random baseline acc: {random_pred_acc:.2f}% (should be ~4%)')
