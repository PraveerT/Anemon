import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nvidia_dataloader

K = 6


def rot_to_quat(R):
    orig = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    m00, m01, m02 = Rf[:, 0, 0], Rf[:, 0, 1], Rf[:, 0, 2]
    m10, m11, m12 = Rf[:, 1, 0], Rf[:, 1, 1], Rf[:, 1, 2]
    m20, m21, m22 = Rf[:, 2, 0], Rf[:, 2, 1], Rf[:, 2, 2]
    tr = m00 + m11 + m22
    B = Rf.shape[0]
    device = Rf.device
    q = torch.zeros(B, 4, device=device, dtype=Rf.dtype)
    m1 = tr > 0
    if m1.any():
        s = torch.sqrt(tr[m1].clamp(min=-0.999) + 1.0) * 2
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (m21[m1] - m12[m1]) / s
        q[m1, 2] = (m02[m1] - m20[m1]) / s
        q[m1, 3] = (m10[m1] - m01[m1]) / s
    rem = ~m1
    m2a = rem & (m00 > m11) & (m00 > m22)
    if m2a.any():
        s = torch.sqrt(1 + m00[m2a] - m11[m2a] - m22[m2a]).clamp(min=1e-8) * 2
        q[m2a, 0] = (m21[m2a] - m12[m2a]) / s
        q[m2a, 1] = 0.25 * s
        q[m2a, 2] = (m01[m2a] + m10[m2a]) / s
        q[m2a, 3] = (m02[m2a] + m20[m2a]) / s
    m2b = rem & (~m2a) & (m11 > m22)
    if m2b.any():
        s = torch.sqrt(1 + m11[m2b] - m00[m2b] - m22[m2b]).clamp(min=1e-8) * 2
        q[m2b, 0] = (m02[m2b] - m20[m2b]) / s
        q[m2b, 1] = (m01[m2b] + m10[m2b]) / s
        q[m2b, 2] = 0.25 * s
        q[m2b, 3] = (m12[m2b] + m21[m2b]) / s
    m2c = rem & (~m2a) & (~m2b)
    if m2c.any():
        s = torch.sqrt(1 + m22[m2c] - m00[m2c] - m11[m2c]).clamp(min=1e-8) * 2
        q[m2c, 0] = (m10[m2c] - m01[m2c]) / s
        q[m2c, 1] = (m02[m2c] + m20[m2c]) / s
        q[m2c, 2] = (m12[m2c] + m21[m2c]) / s
        q[m2c, 3] = 0.25 * s
    return F.normalize(q, dim=-1).reshape(*orig, 4)


def kabsch_quat_masked(src, tgt, mask):
    device = src.device
    w = mask.float()
    w_sum = w.sum().clamp(min=1.0)
    if w_sum < 3:
        return torch.tensor([1, 0, 0, 0], device=device, dtype=src.dtype)
    src_mean = (src * w.unsqueeze(-1)).sum(0) / w_sum
    tgt_mean = (tgt * w.unsqueeze(-1)).sum(0) / w_sum
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    H = (src_c * w.unsqueeze(-1)).T @ tgt_c
    H = H + 1e-6 * torch.eye(3, device=device)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.T
    det = torch.det(V @ U.T)
    D = torch.diag(torch.tensor([1, 1, det.item()], device=device))
    R = V @ D @ U.T
    return rot_to_quat(R)


def corr_sample_indices(orig_flat_idx, corr_target, corr_weight, pts_size, F_, P):
    device = orig_flat_idx.device
    idx0 = torch.linspace(0, P - 1, pts_size, device=device).long()
    sampled_idx = torch.zeros(F_, pts_size, dtype=torch.long, device=device)
    matched = torch.zeros(F_, pts_size, dtype=torch.bool, device=device)
    sampled_idx[0] = idx0
    matched[0] = True
    current_prov = orig_flat_idx[0, idx0].long()
    total_pts = corr_target.shape[-1]
    raw_ppf = total_pts // F_
    for t in range(F_ - 1):
        next_prov = orig_flat_idx[t + 1].long()
        reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
        reverse_map[next_prov] = torch.arange(P, device=device)
        tgt_flat = corr_target[current_prov]
        tgt_w = corr_weight[current_prov]
        tgt_flat_safe = tgt_flat.clamp(min=0)
        tgt_frame = tgt_flat // raw_ppf
        tgt_pos = reverse_map[tgt_flat_safe]
        valid = (tgt_flat >= 0) & (tgt_w > 0) & (tgt_frame == t + 1) & (tgt_pos >= 0)
        next_idx = torch.randint(0, P, (pts_size,), device=device)
        next_idx[valid] = tgt_pos[valid]
        sampled_idx[t + 1] = next_idx
        matched[t + 1] = valid & matched[t]
        current_prov = orig_flat_idx[t + 1, next_idx].long()
    return sampled_idx, matched


def kmeans(x, k, n_iter=10):
    N = x.shape[0]
    idx_init = torch.randperm(N, device=x.device)[:k]
    centers = x[idx_init]
    for _ in range(n_iter):
        d = torch.cdist(x, centers)
        labels = d.argmin(-1)
        for ki in range(k):
            m = labels == ki
            if m.sum() > 0:
                centers[ki] = x[m].mean(0)
    return labels


print('Collecting relative-to-frame-0 K=6 quaternions...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    Q = np.zeros((N, 31, K, 4), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    for i in range(N):
        s = loader[i]
        pts_dict = s[0]
        label = s[1]
        pts = pts_dict['points'].cuda()
        F_, P, C = pts.shape
        xyz = pts[..., :3]
        orig = pts_dict['orig_flat_idx'].cuda()
        ctgt = torch.from_numpy(pts_dict['corr_full_target_idx']).long().cuda()
        cw = torch.from_numpy(pts_dict['corr_full_weight']).float().cuda()
        S = min(256, P)
        sampled_idx, matched = corr_sample_indices(orig, ctgt, cw, S, F_, P)
        xyz_samp = torch.gather(xyz, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))
        part_labels = kmeans(xyz_samp[0], K)
        src_frame0 = xyz_samp[0]
        for t in range(1, F_):
            for kp in range(K):
                mask = (part_labels == kp) & matched[t]
                q = kabsch_quat_masked(src_frame0, xyz_samp[t], mask)
                Q[i, t - 1, kp] = q.cpu().numpy()
        labels[i] = label
        if (i + 1) % 100 == 0:
            print(f'  {phase} {i + 1}/{N}')
    return torch.from_numpy(Q), torch.from_numpy(labels)


q_train, y_train = collect('train')
q_test, y_test = collect('test')
print(f'train q: {q_train.shape}  y: {y_train.shape}')
print(f'test  q: {q_test.shape}   y: {y_test.shape}')

for q in [q_train, q_test]:
    flip = q[..., 0] < 0
    q[flip] = -q[flip]


class QClf(nn.Module):
    def __init__(self, num_classes=25, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv1d(K * 4, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv1d(256, hidden, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, q):
        B, T, K_, D = q.shape
        x = q.reshape(B, T, K_ * D).transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        return self.fc(self.pool(x).squeeze(-1))


model = QClf().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
q_train_c = q_train.cuda()
y_train_c = y_train.cuda()
q_test_c = q_test.cuda()
y_test_c = y_test.cuda()
BS = 64
N_train = len(q_train)
best = 0
for ep in range(80):
    model.train()
    perm = torch.randperm(N_train)
    for i in range(0, N_train, BS):
        idx = perm[i:i + BS]
        opt.zero_grad()
        loss = F.cross_entropy(model(q_train_c[idx]), y_train_c[idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(q_test_c).argmax(-1) == y_test_c).float().mean().item()
    if acc > best:
        best = acc
    if ep % 10 == 0 or ep == 79:
        print(f'epoch {ep:2d}  test_acc={acc*100:.2f}%  best={best*100:.2f}%')

print(f'\n=== RELATIVE-TO-F0 K={K} QUAT BEST ACC: {best*100:.2f}% ===')
print(f'  vs per-pair K=6: 14.11%, whole-cloud: 13.90%, random: 4.00%')
