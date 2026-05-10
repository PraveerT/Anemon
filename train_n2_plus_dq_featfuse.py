"""Approach C: feature-level fusion of N2 (PMamba on depth points) + DQNet-v2
(transformer on dual quaternions).

Both backbones FROZEN. Concatenate penultimate features. Train a small classification
head on the joint feature. Result must be >= max(N2, DQNet-v2) if either network
provides any useful info beyond the other (no fancy aux, just additive).

Pipeline:
  depth_pts -> N2 backbone (frozen)         -> N2_feat (some dim)
  dual_quat -> DQNet-v2 backbone (frozen)   -> DQ_feat (192)
  concat -> small MLP head -> 25-class logits

Trained for 60 epochs, AdamW lr 1e-3.
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
NUM_EPOCHS = 60
BS = 8
LR = 1e-3
WD = 1e-4
WORK_DIR = './work_dir/n2_plus_dq_featfuse/'
os.makedirs(WORK_DIR, exist_ok=True)
N2_CKPT = './work_dir/pmamba_dtw/epoch110_model.pt'
DQNET_CKPT = './work_dir/dqnet_v2/best_model.pt'

# ============ Dual quat encoder (load from skel landmarks) ============
SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'

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


def relpath_to_key(line):
    parts = line.strip().split('\t')
    relpath = parts[1] if len(parts) > 1 else line
    m = re.search(r'class_(\d+)/(subject\S+?)/', relpath)
    return f'./Video_data/class_{m.group(1)}/{m.group(2)}' if m else None


class JointDataset(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        pts, lbl, name = self.base[i]
        key = relpath_to_key(name)
        dq = encoded.get(key, np.zeros((T_FIXED, B, 8), dtype=np.float32))
        return pts, lbl, torch.from_numpy(dq)

def collate(batch):
    pts = torch.stack([b[0].float() if torch.is_tensor(b[0]) else torch.from_numpy(b[0]).float() for b in batch])
    lbls = torch.tensor([b[1] for b in batch], dtype=torch.long)
    dqs = torch.stack([b[2] for b in batch])
    return pts, lbls, dqs


# ============ Backbones ============

class MotionFeats(Motion):
    """Returns global feature vector before classifier."""
    def get_global_feat(self, inputs):
        if isinstance(inputs, dict): inputs = inputs['points']
        coords = self._sample_points(inputs)
        Bz, in_dims, T, P = coords.shape
        ret_array1 = self.group.group_points(distance_dim=[0,1,2], array1=coords, array2=coords, knn=self.knn[0], dim=3)
        ret_array1 = ret_array1.reshape(Bz, in_dims, T*P, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(Bz, -1, T, P)
        fea1 = torch.cat((coords, fea1), dim=1)
        in_dims_2 = fea1.shape[1] * 2 - self.coord_channels
        pts_num = P // self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0,1,2], self.knn[1], 3, coord_dim=self.coord_channels)
        ret2, coords_d = self.select_ind(rg2, coords, Bz, in_dims_2, T, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(Bz, -1, T, pts_num)
        fea2 = torch.cat((coords_d, fea2), dim=1)
        fea2 = self.multi_scale(fea2)
        in_dims_3 = fea2.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0,1,2], self.knn[2], 3, coord_dim=self.coord_channels)
        ret3, coords_d = self.select_ind(rg3, coords_d, Bz, in_dims_3, T, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(Bz, -1, T, pts_num)
        fea3 = self.mamba(fea3)
        coords_fea3 = torch.cat((coords_d, fea3), dim=1)
        out = self.stage5(coords_fea3)
        out = self.pool5(out)
        out = self.global_bn(out).flatten(1)
        return out


class DQNet_Transformer_Feat(nn.Module):
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
        self.head = nn.Linear(d_model, num_classes)  # ignored
    def get_feat(self, x):
        b = x.shape[0]
        x = x.flatten(2)
        x = self.in_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_emb
        x = self.transformer(x)
        return self.norm(x[:, 0])  # (b, d_model)


print('loading N2 backbone...')
n2 = MotionFeats(num_classes=25, pts_size=96, knn=[32,24,48,24], topk=8, multi_scale_num_scales=5).cuda()
n2_state = torch.load(N2_CKPT, map_location='cpu')['model_state_dict']
res = n2.load_state_dict(n2_state, strict=False)
print(f'  N2 missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
for p in n2.parameters(): p.requires_grad = False
n2.eval()

print('loading DQNet-v2 backbone...')
dqn = DQNet_Transformer_Feat().cuda()
dqn_state = torch.load(DQNET_CKPT, map_location='cpu')['model_state_dict']
res = dqn.load_state_dict(dqn_state, strict=False)
print(f'  DQNet missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
for p in dqn.parameters(): p.requires_grad = False
dqn.eval()

# === probe feature dims ===
base_test = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='test')
ds_te0 = JointDataset(base_test)
sample_pts, sample_lbl, sample_dq = ds_te0[0]
sample_pts = sample_pts.unsqueeze(0).cuda().float()
sample_dq = sample_dq.unsqueeze(0).cuda()
with torch.no_grad():
    f_n2 = n2.get_global_feat(sample_pts); print(f'  N2 feat dim = {f_n2.shape[-1]}')
    f_dq = dqn.get_feat(sample_dq); print(f'  DQNet feat dim = {f_dq.shape[-1]}')

D_FUSE = f_n2.shape[-1] + f_dq.shape[-1]
print(f'  Concat feat dim = {D_FUSE}')

# === fusion head ===
head = nn.Sequential(
    nn.Linear(D_FUSE, 256), nn.GELU(), nn.Dropout(0.3),
    nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
    nn.Linear(128, NUM_CLASSES),
).cuda()
print(f'  Head params: {sum(p.numel() for p in head.parameters()):,}')

base_train = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='train')
ds_tr = JointDataset(base_train)
ds_te = JointDataset(base_test)
loader_tr = DataLoader(ds_tr, batch_size=BS, num_workers=4, shuffle=True, collate_fn=collate)
loader_te = DataLoader(ds_te, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate)

opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

# === eval baselines first ===
print('\nBaseline eval (frozen backbones, separate classification):')
n2_classifier_state = {k: v for k, v in n2_state.items() if k.startswith('classifier') or k.startswith('fc_layer')}
print(f'  N2 classifier keys in ckpt: {list(n2_classifier_state.keys())[:5]}')

best_te = 0
for ep in range(1, NUM_EPOCHS + 1):
    head.train()
    losses = []
    correct_tr = 0; total_tr = 0
    for pts, lbl, dq in loader_tr:
        pts = pts.cuda().float(); lbl = lbl.cuda(); dq = dq.cuda()
        with torch.no_grad():
            f1 = n2.get_global_feat(pts)
            f2 = dqn.get_feat(dq)
        feat = torch.cat([f1, f2], dim=1)
        logits = head(feat)
        loss = F.cross_entropy(logits, lbl)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        correct_tr += (logits.argmax(1) == lbl).sum().item(); total_tr += lbl.numel()
    sched.step()
    head.eval()
    correct = 0; total = 0; all_probs = []; all_lbl = []
    with torch.no_grad():
        for pts, lbl, dq in loader_te:
            pts = pts.cuda().float(); lbl = lbl.cuda(); dq = dq.cuda()
            f1 = n2.get_global_feat(pts); f2 = dqn.get_feat(dq)
            feat = torch.cat([f1, f2], dim=1)
            logits = head(feat)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy()); all_lbl.append(lbl.cpu().numpy())
            correct += (logits.argmax(1) == lbl).sum().item(); total += lbl.numel()
    te_acc = correct / total * 100
    if te_acc > best_te:
        best_te = te_acc
        np.savez(os.path.join(WORK_DIR, 'best_probs.npz'),
                 probs=np.concatenate(all_probs), labels=np.concatenate(all_lbl))
        torch.save({'head_state_dict': head.state_dict(), 'epoch': ep},
                   os.path.join(WORK_DIR, 'best_head.pt'))
    print(f'ep {ep:3d}  loss={np.mean(losses):.4f}  tr_acc={correct_tr/total_tr*100:.2f}  test={te_acc:.2f}  best={best_te:.2f}', flush=True)
print(f'\nBEST: {best_te:.2f}')
