"""Train Net2 (DTW) + per-finger Kabsch QCC aux supervised by skeleton-derived
finger quaternions. Real correspondence — no NN matching.
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
LR = 1.2e-5  # warm fine-tune
WORK_DIR = './work_dir/pmamba_finger_qcc/'
os.makedirs(WORK_DIR, exist_ok=True)
WARM_CKPT = './work_dir/pmamba_dtw/epoch100_model.pt'
QCC_WEIGHT = 0.5

# Load finger quat targets
TARGETS_PATH = '/notebooks/PMamba/dataset/Nvidia/Processed/finger_quat_targets.npz'
finger_q_dict = dict(np.load(TARGETS_PATH, allow_pickle=False))
print(f'Loaded {len(finger_q_dict)} finger_q targets')


def relpath_to_key(line):
    """Map list-line to skeleton key './Video_data/class_XX/subject_X'."""
    parts = line.strip().split('\t')
    relpath = parts[1] if len(parts) > 1 else line
    m = re.search(r'class_(\d+)/(subject\S+?)/', relpath)
    if m:
        return f'./Video_data/class_{m.group(1)}/{m.group(2)}'
    return None


class FingerQDataset(Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        pts, lbl, name = self.base[i]
        key = relpath_to_key(name)
        target = finger_q_dict.get(key, np.zeros((0, 5, 4), dtype=np.float32))
        return pts, lbl, target, name


def collate(batch):
    pts = torch.stack([b[0].float() if torch.is_tensor(b[0]) else torch.from_numpy(b[0]).float() for b in batch])
    lbls = torch.tensor([b[1] for b in batch], dtype=torch.long)
    targets = [torch.from_numpy(b[2]) for b in batch]
    names = [b[3] for b in batch]
    return pts, lbls, targets, names


class MotionWithFingerQ(Motion):
    def __init__(self, num_classes, pts_size, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        feat_dim = 64
        self.qcc_head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.GELU(),
            nn.Linear(feat_dim, 5 * 4),
        )

    def forward_with_features(self, inputs):
        if isinstance(inputs, dict): inputs = inputs['points']
        coords = self._sample_points(inputs)
        B, in_dims, T, P = coords.shape
        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(B, in_dims, T * P, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(B, -1, T, P)
        feat_per_frame = fea1_raw.mean(dim=-1)  # (B, 64, T)

        # Continue main pipeline
        fea1 = torch.cat((coords, fea1_raw), dim=1)
        in_dims_2 = fea1.shape[1] * 2 - self.coord_channels
        pts_num = P // self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3, coord_dim=self.coord_channels)
        ret2, coords_d = self.select_ind(rg2, coords, B, in_dims_2, T, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(B, -1, T, pts_num)
        fea2 = torch.cat((coords_d, fea2), dim=1)
        fea2 = self.multi_scale(fea2)
        in_dims_3 = fea2.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3, coord_dim=self.coord_channels)
        ret3, coords_d = self.select_ind(rg3, coords_d, B, in_dims_3, T, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(B, -1, T, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords_d, fea3_mamba), dim=1)
        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1), feat_per_frame


print('Building model and loaders...')
model = Motion(num_classes=25, pts_size=96, knn=[32, 24, 48, 24], topk=8, multi_scale_num_scales=5).cuda()
# Replace forward_with_features into instance
model_finger = MotionWithFingerQ(num_classes=25, pts_size=96, knn=[32, 24, 48, 24], topk=8, multi_scale_num_scales=5).cuda()
state = torch.load(WARM_CKPT, map_location='cpu')['model_state_dict']
res = model_finger.load_state_dict(state, strict=False)
print(f'Loaded {len(state) - len(res.missing_keys)} keys, missing {len(res.missing_keys)} (expected: qcc_head)')

base_train = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='train')
base_test = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='test')
ds_train = FingerQDataset(base_train)
ds_test = FingerQDataset(base_test)
loader_train = DataLoader(ds_train, batch_size=BS, num_workers=8, shuffle=True, collate_fn=collate)
loader_test = DataLoader(ds_test, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate)

opt = torch.optim.Adam(model_finger.parameters(), lr=LR, weight_decay=0.03)

best_test = 0
for ep in range(1, NUM_EPOCHS + 1):
    model_finger.train()
    train_loss_avg = 0; train_aux_avg = 0; n_batch = 0
    for pts, lbl, targets, names in loader_train:
        pts = pts.cuda()
        lbl = lbl.cuda()
        # Forward with intermediate features
        features, feat_per_frame = model_finger.forward_with_features(pts)
        logits = model_finger.classify_features(features)
        ce_loss = F.cross_entropy(logits, lbl)

        # Aux: predict per-finger q from feature pairs
        T = feat_per_frame.shape[-1]
        f1 = feat_per_frame[..., :-1]; f2 = feat_per_frame[..., 1:]
        feat_pair = torch.cat([f1, f2], dim=1).permute(0, 2, 1)  # (B, T-1, 128)
        q_pred = model_finger.qcc_head(feat_pair).reshape(pts.shape[0], T - 1, 5, 4)
        q_pred = F.normalize(q_pred, dim=-1)

        # Build batched target with NaN mask
        aux = torch.tensor(0.0, device=pts.device); n_aux = 0
        for b, t_arr in enumerate(targets):
            # t_arr shape: (T_landmark - 1, 5, 4); map to T - 1 via interp
            T_lm = t_arr.shape[0]
            if T_lm < 1: continue
            T_pred = T - 1
            idx = torch.linspace(0, T_lm - 1, T_pred).long()
            t_resampled = t_arr[idx].cuda()  # (T_pred, 5, 4)
            # Mask zero-quat (landmarks not detected)
            mask = (t_resampled.norm(dim=-1) > 0.5)  # (T_pred, 5)
            if mask.sum() == 0: continue
            cos = (q_pred[b] * t_resampled).sum(-1).abs()  # (T_pred, 5)
            aux = aux + ((1 - cos) * mask.float()).sum() / mask.sum().clamp(min=1)
            n_aux += 1
        aux = aux / max(1, n_aux)

        loss = ce_loss + QCC_WEIGHT * aux
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss_avg += ce_loss.item(); train_aux_avg += aux.item(); n_batch += 1

    model_finger.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for pts, lbl, targets, names in loader_test:
            pts = pts.cuda()
            lbl = lbl.cuda()
            features, _ = model_finger.forward_with_features(pts)
            logits = model_finger.classify_features(features)
            correct += (logits.argmax(1) == lbl).sum().item()
            total += lbl.numel()
    test_acc = correct / total * 100
    if test_acc > best_test:
        best_test = test_acc
        torch.save({'model_state_dict': model_finger.state_dict(), 'epoch': ep}, os.path.join(WORK_DIR, 'best_model.pt'))
    if ep % 5 == 0:
        torch.save({'model_state_dict': model_finger.state_dict(), 'epoch': ep}, os.path.join(WORK_DIR, f'epoch{ep}_model.pt'))
    print(f'ep {ep:3d}  ce={train_loss_avg/n_batch:.4f}  aux={train_aux_avg/n_batch:.4f}  test={test_acc:.2f}  best={best_test:.2f}', flush=True)

print(f'\nBEST: {best_test:.2f}')
