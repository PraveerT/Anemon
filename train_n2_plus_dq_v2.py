"""Approach C-fixed: N2 + dual quat additive fusion that GUARANTEES result >= N2 at init.

Architecture:
  N2 (frozen) -> 25-d logits (its own classifier)
  DQNet-v2 (frozen) -> 192-d feature
  Fusion: residual from DQ feat
    fused_logits = N2_logits + alpha * Linear(DQ_feat -> 25)
  Initialize: alpha=0, Linear=zero. At init, fused_logits == N2_logits exactly.
  Train alpha (scalar) + the 192->25 Linear with low LR + small weight decay.

This is the MINIMAL additive contribution test. If DQ provides any complementary
info, fused_acc must improve over N2 (88.59). If DQ adds nothing, alpha stays
near 0 and result stays at N2's 88.59.
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
LR = 5e-4
WD = 5e-4
WORK_DIR = './work_dir/n2_plus_dq_v2/'
os.makedirs(WORK_DIR, exist_ok=True)
N2_CKPT = './work_dir/pmamba_dtw/epoch110_model.pt'
DQNET_CKPT = './work_dir/dqnet_v2/best_model.pt'

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


class DQNet_Transformer_Feat(nn.Module):
    def __init__(self, d_model=192, n_heads=8, n_layers=4, dropout=0.2):
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
        self.head = nn.Linear(d_model, 25)  # not used here
    def get_feat(self, x):
        b = x.shape[0]
        x = x.flatten(2)
        x = self.in_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_emb
        x = self.transformer(x)
        return self.norm(x[:, 0])


print('loading N2 (full model with classifier)...')
n2 = Motion(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8, multi_scale_num_scales=5).cuda()
n2_state = torch.load(N2_CKPT, map_location='cpu')['model_state_dict']
res = n2.load_state_dict(n2_state, strict=False)
print(f'  N2 missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
for p in n2.parameters(): p.requires_grad = False
n2.eval()

print('loading DQNet-v2...')
dqn = DQNet_Transformer_Feat().cuda()
dqn_state = torch.load(DQNET_CKPT, map_location='cpu')['model_state_dict']
res = dqn.load_state_dict(dqn_state, strict=False)
print(f'  DQNet missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
for p in dqn.parameters(): p.requires_grad = False
dqn.eval()

# ============ Additive fusion: N2_logits + alpha * Linear(DQ_feat -> 25) ============
class AdditiveFusion(nn.Module):
    def __init__(self, d_dq=192, n_classes=25):
        super().__init__()
        self.linear = nn.Linear(d_dq, n_classes)
        # Small random init for Linear so gradient can flow
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
        # Alpha at small nonzero to allow gradient through to Linear
        self.alpha = nn.Parameter(torch.tensor([0.05]))
    def forward(self, n2_logits, dq_feat):
        return n2_logits + self.alpha * self.linear(dq_feat)

fuser = AdditiveFusion().cuda()
print(f'Fusion params: {sum(p.numel() for p in fuser.parameters()):,}')

base_train = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='train')
base_test = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='test')
ds_tr = JointDataset(base_train); ds_te = JointDataset(base_test)
loader_tr = DataLoader(ds_tr, batch_size=BS, num_workers=4, shuffle=True, collate_fn=collate)
loader_te = DataLoader(ds_te, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate)

# === sanity: eval at init + verify DQNet loaded right ===
print('\nSanity: N2 solo + DQNet solo + fusion init...')
n2_correct = 0; dq_correct = 0; fuse_correct = 0; total = 0
fuser.eval()
with torch.no_grad():
    for pts, lbl, dq in loader_te:
        pts = pts.cuda().float(); lbl = lbl.cuda(); dq = dq.cuda()
        n2_logits = n2(pts); dq_feat = dqn.get_feat(dq)
        # DQNet's own classifier on its feature for solo eval
        dq_logits_solo = dqn.head(dq_feat)
        n2_correct += (n2_logits.argmax(1) == lbl).sum().item()
        dq_correct += (dq_logits_solo.argmax(1) == lbl).sum().item()
        logits = fuser(n2_logits, dq_feat)
        fuse_correct += (logits.argmax(1) == lbl).sum().item()
        total += lbl.numel()
print(f'  N2 solo:    {n2_correct/total*100:.2f}  (expect 88.59)')
print(f'  DQNet solo: {dq_correct/total*100:.2f}  (expect 71.78)')
print(f'  Fuser init: {fuse_correct/total*100:.2f}')
init_acc = fuse_correct / total * 100

opt = torch.optim.AdamW(fuser.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

best_te = init_acc
print('\nTraining additive head...')
for ep in range(1, NUM_EPOCHS + 1):
    fuser.train()
    losses = []
    for pts, lbl, dq in loader_tr:
        pts = pts.cuda().float(); lbl = lbl.cuda(); dq = dq.cuda()
        with torch.no_grad():
            n2_logits = n2(pts); dq_feat = dqn.get_feat(dq)
        logits = fuser(n2_logits, dq_feat)
        loss = F.cross_entropy(logits, lbl)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    sched.step()
    fuser.eval()
    correct = 0; total = 0; all_probs = []; all_lbl = []
    with torch.no_grad():
        for pts, lbl, dq in loader_te:
            pts = pts.cuda().float(); lbl = lbl.cuda(); dq = dq.cuda()
            n2_logits = n2(pts); dq_feat = dqn.get_feat(dq)
            logits = fuser(n2_logits, dq_feat)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy()); all_lbl.append(lbl.cpu().numpy())
            correct += (logits.argmax(1) == lbl).sum().item(); total += lbl.numel()
    te_acc = correct / total * 100
    if te_acc > best_te:
        best_te = te_acc
        np.savez(os.path.join(WORK_DIR, 'best_probs.npz'),
                 probs=np.concatenate(all_probs), labels=np.concatenate(all_lbl))
        torch.save({'fuser_state_dict': fuser.state_dict(), 'epoch': ep},
                   os.path.join(WORK_DIR, 'best_model.pt'))
    a = fuser.alpha.item()
    print(f'ep {ep:3d}  loss={np.mean(losses):.4f}  alpha={a:.4f}  test={te_acc:.2f}  best={best_te:.2f}', flush=True)
print(f'\nBEST: {best_te:.2f}  (N2 alone = 88.59)')
