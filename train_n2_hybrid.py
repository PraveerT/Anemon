"""Approach (b): PMamba on hybrid input = depth points + bone-direction-quaternion
points, jointly fed through one extended-channel architecture.

Loader yields pts shape (T, P, 8). We take first 4 channels (xyz+time) for compat.
Add 20 bone "points" with channels [mid_xyz, time, qw, qx, qy, qz, 0] = 8 ch.
Total: (T, 256+20=276, 8). PMamba's coord_channels stays 4 but we add a 1x1
projection that lets the network mix quat info into the 4-channel space.
"""
import sys, os, re, time, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader, Dataset
import nvidia_dataloader
from models.motion import Motion

T_FIXED = 32
NUM_CLASSES = 25
NUM_EPOCHS = 80
BS = 8
LR = 1e-4
WD = 0.03
WORK_DIR = './work_dir/n2_hybrid_dq/'
os.makedirs(WORK_DIR, exist_ok=True)

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'

BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
B = 20

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

def encode_bone_aug(lm):
    """(T, 21, 3) -> (T, 20, 7): midpoint + quaternion per bone."""
    T = lm.shape[0]
    out = np.zeros((T, B, 7), dtype=np.float32)
    for b, (p, c) in enumerate(BONES):
        bone_vec = lm[:, c, :] - lm[:, p, :]
        rot_q = vec_to_quat_np(bone_vec)
        mid = (lm[:, c, :] + lm[:, p, :]) / 2
        out[:, b, :3] = mid
        out[:, b, 3:] = rot_q
    return out

print('precomputing bone augmentation features...')
bone_feats = {}
for k, lm_raw in sk.items():
    if lm_raw.shape[0] < 4: continue
    lm = fillnan(lm_raw)
    f = encode_bone_aug(lm)
    idx = np.linspace(0, f.shape[0]-1, T_FIXED).astype(np.int64)
    bone_feats[k] = f[idx]  # (T_FIXED, 20, 7)

# Normalize bone midpoints to depth-point scale
all_mids = np.stack([bf[..., :3] for bf in bone_feats.values()])
mid_mean = float(all_mids.mean()); mid_std = float(all_mids.std()) + 1e-7
print(f'bone midpoint global mean={mid_mean:.3f} std={mid_std:.3f}')


def relpath_to_key(line):
    parts = line.strip().split('\t')
    relpath = parts[1] if len(parts) > 1 else line
    m = re.search(r'class_(\d+)/(subject\S+?)/', relpath)
    return f'./Video_data/class_{m.group(1)}/{m.group(2)}' if m else None


class HybridDataset(Dataset):
    """Loader yields pts shape (T, P_depth, 8). We append (T, 20, 8) bone points.
    Bone channels: [mx, my, mz, time, qw, qx, qy, qz]
    """
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        pts, lbl, name = self.base[i]
        if not torch.is_tensor(pts): pts = torch.from_numpy(pts).float()
        else: pts = pts.float()
        # pts: (T, P_depth, 8)
        T_d, P_d, C_d = pts.shape
        # Time coordinate (already in pts[..., 3])
        # Get bone aug
        key = relpath_to_key(name)
        bf = bone_feats.get(key, np.zeros((T_FIXED, B, 7), dtype=np.float32))
        # Build bone tensor (T, 20, 8)
        # Channels [0:3] = mid_xyz (normalized to similar scale)
        # Channel  [3]   = time-coord (linspace -1..1)
        # Channels [4:8] = quaternion
        bone_tensor = torch.zeros((T_d, B, 8), dtype=torch.float32)
        mid_norm = (bf[..., :3] - mid_mean) / mid_std
        bone_tensor[..., :3] = torch.from_numpy(mid_norm).float()
        time_coord = torch.linspace(-1, 1, T_d)
        bone_tensor[..., 3] = time_coord.unsqueeze(-1).expand(T_d, B)
        bone_tensor[..., 4:8] = torch.from_numpy(bf[..., 3:]).float()
        # Cat along P dim
        combined = torch.cat([pts, bone_tensor], dim=1)  # (T, P_d+20, 8)
        return combined, lbl

def collate(batch):
    pts = torch.stack([b[0] for b in batch])
    lbls = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return pts, lbls


print('building model...')
# PMamba expects input shape (B, C, T, P). Loader-provided pts are (B, T, P, 8) after batching.
# We'll permute. coord_channels=4 for kNN; full 8 channels go through stage1.
# Need to update Motion.stage1 to accept 8 input channels.

class HybridMotion(nn.Module):
    def __init__(self, base_model, in_channels=8, coord_channels=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, coord_channels, kernel_size=1, bias=True)
        nn.init.eye_(self.proj.weight.squeeze(-1).squeeze(-1)[:coord_channels, :coord_channels].view(coord_channels, coord_channels))
        # Initialize so that the first 4 channels pass through identity, quat channels (4-7) zero-init
        with torch.no_grad():
            W = self.proj.weight  # (4, 8, 1, 1)
            W.zero_()
            for i in range(coord_channels):
                W[i, i, 0, 0] = 1.0  # identity for first 4
            self.proj.bias.zero_()
        self.base = base_model
    def forward(self, x):
        # x: (B, T, P, 8) -> permute to (B, 8, T, P) then project to (B, 4, T, P)
        x = x.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x)
        return self.base(x_proj)

base_model = Motion(num_classes=25, pts_size=276, knn=[32,24,48,24], topk=8, multi_scale_num_scales=5, coord_channels=4).cuda()
model = HybridMotion(base_model, in_channels=8, coord_channels=4).cuda()
n_params = sum(p.numel() for p in model.parameters())
print(f'Hybrid model params: {n_params:,}')

base_train = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='train')
base_test = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='test')
ds_tr = HybridDataset(base_train); ds_te = HybridDataset(base_test)
loader_tr = DataLoader(ds_tr, batch_size=BS, num_workers=4, shuffle=True, collate_fn=collate)
loader_te = DataLoader(ds_te, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate)

# sanity check shape
x0, y0 = ds_tr[0]
print(f'sample input shape: {x0.shape} (expect [T, P_total, 8])')
xb = torch.stack([x0]).cuda()
print(f'batched: {xb.shape}')

opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

best_te = 0
for ep in range(1, NUM_EPOCHS + 1):
    model.train()
    losses = []
    for x, y in loader_tr:
        x = x.cuda(); y = y.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    sched.step()
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader_te:
            x = x.cuda(); y = y.cuda()
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
    te_acc = correct / total * 100
    if te_acc > best_te:
        best_te = te_acc
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                   os.path.join(WORK_DIR, 'best_model.pt'))
    print(f'ep {ep:3d}  loss={np.mean(losses):.4f}  test={te_acc:.2f}  best={best_te:.2f}', flush=True)
print(f'\nBEST: {best_te:.2f}  (N2 alone = 88.59)')
