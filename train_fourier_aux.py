"""Train PMamba (warm from N2 ep100) with Fourier-band-energy aux loss
on the global feature embedding. 60-D regression target derived from MediaPipe
fingertip trajectories per axis per band.
"""
import sys, os, re, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader, Dataset
import nvidia_dataloader
from models.motion import Motion

NUM_EPOCHS = 60
BS = 8
LR = 1.2e-5
AUX_WEIGHT = 0.5
WORK_DIR = './work_dir/pmamba_fourier_aux/'
os.makedirs(WORK_DIR, exist_ok=True)
WARM_CKPT = './work_dir/pmamba_dtw/epoch100_model.pt'
TARGETS = '/notebooks/PMamba/dataset/Nvidia/Processed/fourier_targets.npz'

print('loading targets...')
targets = dict(np.load(TARGETS, allow_pickle=False))
print(f'{len(targets)} targets, dim={list(targets.values())[0].shape}')
# Standardize targets
all_t = np.stack([targets[k] for k in targets])
T_MEAN = all_t.mean(0); T_STD = all_t.std(0) + 1e-7
print(f'target mean range: [{T_MEAN.min():.2f}, {T_MEAN.max():.2f}]')


def relpath_to_key(line):
    parts = line.strip().split('\t')
    relpath = parts[1] if len(parts) > 1 else line
    m = re.search(r'class_(\d+)/(subject\S+?)/', relpath)
    return f'./Video_data/class_{m.group(1)}/{m.group(2)}' if m else None


class FourierDataset(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        pts, lbl, name = self.base[i]
        key = relpath_to_key(name)
        t = targets.get(key, np.zeros(60, dtype=np.float32))
        t = (t - T_MEAN) / T_STD
        return pts, lbl, t.astype(np.float32), name

def collate(batch):
    pts = torch.stack([b[0].float() if torch.is_tensor(b[0]) else torch.from_numpy(b[0]).float() for b in batch])
    lbls = torch.tensor([b[1] for b in batch], dtype=torch.long)
    targs = torch.from_numpy(np.stack([b[2] for b in batch]))
    names = [b[3] for b in batch]
    return pts, lbls, targs, names


class MotionFourier(Motion):
    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        # global emb dim depends on architecture; we'll inspect at runtime
        # The classifier gets feature size = backbone output. Use same.
        # Add lazy linear; init at first forward.
        self.aux_head = None  # initialized after first features pass

    def get_global_feat(self, inputs):
        if isinstance(inputs, dict): inputs = inputs['points']
        coords = self._sample_points(inputs)
        B, in_dims, T, P = coords.shape
        ret_array1 = self.group.group_points(distance_dim=[0,1,2], array1=coords, array2=coords, knn=self.knn[0], dim=3)
        ret_array1 = ret_array1.reshape(B, in_dims, T*P, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(B, -1, T, P)
        fea1 = torch.cat((coords, fea1), dim=1)
        in_dims_2 = fea1.shape[1] * 2 - self.coord_channels
        pts_num = P // self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0,1,2], self.knn[1], 3, coord_dim=self.coord_channels)
        ret2, coords_d = self.select_ind(rg2, coords, B, in_dims_2, T, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(B, -1, T, pts_num)
        fea2 = torch.cat((coords_d, fea2), dim=1)
        fea2 = self.multi_scale(fea2)
        in_dims_3 = fea2.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0,1,2], self.knn[2], 3, coord_dim=self.coord_channels)
        ret3, coords_d = self.select_ind(rg3, coords_d, B, in_dims_3, T, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(B, -1, T, pts_num)
        fea3 = self.mamba(fea3)
        coords_fea3 = torch.cat((coords_d, fea3), dim=1)
        out = self.stage5(coords_fea3)
        out = self.pool5(out)
        out = self.global_bn(out).flatten(1)
        return out

print('building model...')
model = MotionFourier(num_classes=25, pts_size=96, knn=[32,24,48,24], topk=8, multi_scale_num_scales=5).cuda()
state = torch.load(WARM_CKPT, map_location='cpu')['model_state_dict']
res = model.load_state_dict(state, strict=False)
print(f'loaded {len(state)-len(res.missing_keys)}/{len(state)} keys, missing={res.missing_keys[:3]}')

base_train = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='train')
base_test = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='test')
ds_train = FourierDataset(base_train)
ds_test = FourierDataset(base_test)
loader_train = DataLoader(ds_train, batch_size=BS, num_workers=8, shuffle=True, collate_fn=collate)
loader_test = DataLoader(ds_test, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate)

opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.03)

best_test = 0
for ep in range(1, NUM_EPOCHS + 1):
    model.train()
    ce_avg = 0; aux_avg = 0; n = 0
    for pts, lbl, t, names in loader_train:
        pts = pts.cuda(); lbl = lbl.cuda(); t = t.cuda()
        feat = model.get_global_feat(pts)
        if model.aux_head is None:
            model.aux_head = nn.Sequential(nn.Linear(feat.shape[1], 128), nn.GELU(), nn.Linear(128, 60)).cuda()
            opt.add_param_group({'params': model.aux_head.parameters(), 'lr': LR * 5})
        logits = model.classify_features(feat)
        ce = F.cross_entropy(logits, lbl)
        aux = F.mse_loss(model.aux_head(feat), t)
        loss = ce + AUX_WEIGHT * aux
        opt.zero_grad(); loss.backward(); opt.step()
        ce_avg += ce.item(); aux_avg += aux.item(); n += 1
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for pts, lbl, _, _ in loader_test:
            feat = model.get_global_feat(pts.cuda())
            logits = model.classify_features(feat)
            correct += (logits.argmax(1) == lbl.cuda()).sum().item()
            total += lbl.numel()
    test_acc = correct / total * 100
    if test_acc > best_test:
        best_test = test_acc
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep}, os.path.join(WORK_DIR, 'best_model.pt'))
    if ep % 5 == 0:
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep}, os.path.join(WORK_DIR, f'epoch{ep}_model.pt'))
    print(f'ep {ep:3d}  ce={ce_avg/n:.4f}  aux={aux_avg/n:.4f}  test={test_acc:.2f}  best={best_test:.2f}', flush=True)

print(f'\nBEST: {best_test:.2f}')
