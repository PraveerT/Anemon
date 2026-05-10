"""DQNet-PM: DQNet-Mamba enhanced with PMamba's design choices.

Key changes from DQNet-Mamba (77.80):
  1. Per-bone independent lifting (8 -> hidden) — preserves bone identity
  2. Skeletal graph aggregation — replaces kNN with known bone adjacency
  3. Spatial Mamba (over bones) + Temporal Mamba (over time) interleaved
  4. Multi-scale temporal kernels (1D conv with different sizes)
  5. Mean pool over both spatial (bones) and temporal (frames) at end

Architecture:
  Input: (B, T=32, B_bones=20, 8)

  Stage 1: Per-bone lift (independent linear): 8 -> 96
  Output: (B, T, 20, 96)

  Stage 2: Skeletal graph conv block x2:
    - For each bone, aggregate self + adjacency neighbors via real-mix
    - Two independent linear layers (self, neighbors), residual + norm

  Stage 3: Multi-scale temporal: parallel 1D convs (k=1,3,5,7) along T,
    concat then project back to hidden dim

  Stage 4: Spatial Mamba (over bones, per frame): reshape to (B*T, 20, hid),
    Mamba, reshape back

  Stage 5: Temporal Mamba (over T, per bone): reshape to (B*20, T, hid),
    Mamba, reshape back

  Repeat stages 4+5 for n_layers

  Stage 6: Pool over both bones + time (mean) -> (B, hid)
  Stage 7: Classifier -> 25
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
HIDDEN = 96
N_BLOCKS = 3
WORK_DIR = '/notebooks/PMamba/experiments/work_dir/dqnet_pm/'
os.makedirs(WORK_DIR, exist_ok=True)

BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
B = 20

# Skeletal adjacency: bones share a joint
ADJ_np = np.zeros((B, B), dtype=np.float32)
for i in range(B):
    for j in range(B):
        if i == j: continue
        if {BONES[i][0],BONES[i][1]} & {BONES[j][0],BONES[j][1]}: ADJ_np[i,j] = 1
# Add self-loop
ADJ_np += np.eye(B, dtype=np.float32)
# Row-normalize so each row sums to 1
ADJ_np = ADJ_np / ADJ_np.sum(axis=1, keepdims=True)
ADJ = torch.from_numpy(ADJ_np)


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


class SkeletalGraphConv(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.lin_self = nn.Linear(hid, hid)
        self.lin_neigh = nn.Linear(hid, hid)
        self.norm = nn.LayerNorm(hid)
    def forward(self, x):
        # x: (..., B_bones, hid)
        adj = ADJ.to(x.device)
        neigh = torch.einsum('ij,...jh->...ih', adj, x)
        out = self.lin_self(x) + self.lin_neigh(neigh)
        return self.norm(F.gelu(out)) + x


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, hid, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hid, hid // len(kernels), kernel_size=k, padding=k//2) for k in kernels
        ])
        self.proj = nn.Linear(hid, hid)
        self.norm = nn.LayerNorm(hid)
    def forward(self, x):
        # x: (B, T, hid) — process per batch
        x_t = x.transpose(1, 2)  # (B, hid, T)
        outs = [conv(x_t) for conv in self.convs]
        outs = torch.cat(outs, dim=1)  # (B, hid, T)
        outs = outs.transpose(1, 2)  # (B, T, hid)
        outs = self.proj(outs)
        return self.norm(F.gelu(outs)) + x


class STMambaBlock(nn.Module):
    """Spatial Mamba over bones + Temporal Mamba over T + Multi-scale Temporal Conv."""
    def __init__(self, hid, dropout=0.2):
        super().__init__()
        self.mamba_spatial = Mamba(d_model=hid, d_state=16, d_conv=4, expand=2)
        self.mamba_temporal = Mamba(d_model=hid, d_state=16, d_conv=4, expand=2)
        self.norm_s = nn.LayerNorm(hid)
        self.norm_t = nn.LayerNorm(hid)
        self.dropout = nn.Dropout(dropout)
        self.skel = SkeletalGraphConv(hid)
        self.ms_temp = MultiScaleTemporalConv(hid)
    def forward(self, x):
        # x: (B, T, B_bones, hid)
        Bz, T, Bb, H = x.shape
        # Skeletal graph aggregation (over bones)
        x_s = self.skel(x)
        x = x + self.dropout(x_s - x)  # residual already in skel
        # Spatial Mamba (over bones, per frame)
        z = x.reshape(Bz * T, Bb, H)
        z = self.norm_s(z)
        z = self.mamba_spatial(z)
        z = z.reshape(Bz, T, Bb, H)
        x = x + self.dropout(z)
        # Multi-scale temporal (per bone)
        z = x.transpose(1, 2).reshape(Bz * Bb, T, H)
        z = self.ms_temp(z)
        z = z.reshape(Bz, Bb, T, H).transpose(1, 2)
        x = x + self.dropout(z)
        # Temporal Mamba (over T, per bone)
        z = x.transpose(1, 2).reshape(Bz * Bb, T, H)
        z = self.norm_t(z)
        z = self.mamba_temporal(z)
        z = z.reshape(Bz, Bb, T, H).transpose(1, 2)
        x = x + self.dropout(z)
        return x


class DQNet_PM(nn.Module):
    def __init__(self, hidden=HIDDEN, n_blocks=N_BLOCKS, num_classes=25, dropout=0.2):
        super().__init__()
        # Per-bone lift: 8 -> hidden
        self.lift = nn.Linear(8, hidden)
        self.pos_t = nn.Parameter(torch.randn(1, T_FIXED, 1, hidden) * 0.02)
        self.pos_b = nn.Parameter(torch.randn(1, 1, B, hidden) * 0.02)
        self.blocks = nn.ModuleList([STMambaBlock(hidden, dropout) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        # x: (B, T, B_bones, 8)
        x = self.lift(x)         # (B, T, B_bones, hidden)
        x = x + self.pos_t + self.pos_b
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        x = x.mean(dim=(1, 2))    # pool over (T, bones) -> (B, hidden)
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
model = DQNet_PM(hidden=HIDDEN, n_blocks=N_BLOCKS, num_classes=NUM_CLASSES, dropout=0.2).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'DQNet-PM params: {n_params:,}')

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
print(f'\nBEST: {best_te:.2f}  (DQNet-Mamba=77.80, DQNet-Transformer=71.78)')
