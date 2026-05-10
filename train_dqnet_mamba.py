"""DQNet-Mamba: same input/encoding as DQNet-v2 (transformer) but Mamba SSM
replaces the transformer encoder.

Architecture:
  Input: (B, T=32, B=20, dq=8) -> flatten -> (B, T, 160)
  Linear projection: 160 -> 192 (d_model)
  + learnable positional embedding
  + 4x Mamba(d_model=192, d_state=16, d_conv=4, expand=2) with pre-LN, dropout, residual
  Final LayerNorm
  Mean-pool over time (Mamba is causal — CLS at position 0 would see nothing)
  Linear head: 192 -> 25 classes
"""
import os, re, time, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mamba_ssm.modules.mamba_simple import Mamba

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
T_FIXED = 32
NUM_CLASSES = 25
NUM_EPOCHS = 120
BS = 32
LR = 5e-4
WD = 1e-4
WORK_DIR = '/notebooks/PMamba/experiments/work_dir/dqnet_mamba/'
os.makedirs(WORK_DIR, exist_ok=True)

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

print('precomputing dual quaternions...')
encoded = {}
for k, lm_raw in sk.items():
    if lm_raw.shape[0] < 4: continue
    lm = fillnan(lm_raw)
    dq = encode_sample(lm)
    idx = np.linspace(0, dq.shape[0]-1, T_FIXED).astype(np.int64)
    encoded[k] = dq[idx]
print(f'{len(encoded)} encoded')


class DQNet_Mamba(nn.Module):
    def __init__(self, d_model=192, n_layers=4, num_classes=25, dropout=0.2,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.in_proj = nn.Linear(B * 8, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, T_FIXED, d_model) * 0.02)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        # x: (B, T, B_bones, 8)
        x = x.flatten(2)  # (B, T, 160)
        x = self.in_proj(x) + self.pos_emb  # (B, T, d_model)
        for mamba, norm in zip(self.mamba_layers, self.norms):
            residual = x
            x = norm(x)
            x = mamba(x)
            x = self.dropout(x)
            x = x + residual
        x = self.final_norm(x)
        x = x.mean(dim=1)  # mean over T (Mamba processes causally; mean-pool aggregates)
        return self.head(x)


class DQDataset(Dataset):
    def __init__(self, label_map):
        self.items = [(k, label_map[k]) for k in encoded if k in label_map]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        k, lbl = self.items[i]
        return torch.from_numpy(encoded[k]), lbl

ds_tr = DQDataset(train_lbl); ds_te = DQDataset(test_lbl)
print(f'train {len(ds_tr)} test {len(ds_te)}')

loader_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True, num_workers=4, drop_last=False)
loader_te = DataLoader(ds_te, batch_size=BS, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DQNet_Mamba(d_model=192, n_layers=4, num_classes=NUM_CLASSES, dropout=0.2).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'DQNet-Mamba params: {n_params:,}')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

best_te = 0
for ep in range(1, NUM_EPOCHS + 1):
    model.train()
    losses = []
    for x, y in loader_tr:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    sched.step()
    model.eval()
    correct = 0; total = 0; all_probs = []; all_lbl = []
    with torch.no_grad():
        for x, y in loader_te:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy()); all_lbl.append(y.cpu().numpy())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    te_acc = correct / total * 100
    if te_acc > best_te:
        best_te = te_acc
        np.savez(os.path.join(WORK_DIR, 'best_probs.npz'),
                 probs=np.concatenate(all_probs), labels=np.concatenate(all_lbl))
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                   os.path.join(WORK_DIR, 'best_model.pt'))
    print(f'ep {ep:3d}  loss={np.mean(losses):.4f}  test={te_acc:.2f}  best={best_te:.2f}', flush=True)
print(f'\nBEST: {best_te:.2f}  (DQNet-transformer = 71.78)')
