"""Side-by-side viz: original pointcloud vs rigid-stabilized version.

Picks one sample, runs Kabsch chain, applies inverse to each frame to remap
into frame-0 reference. Plots frames overlaid (colored by time) — left=raw,
right=stabilized. Also shows snapshot grid at t=0,8,16,24,31.

Output: viz_stab.png
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn.functional as F, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nvidia_dataloader


def rot_to_quat(R):
    m00,m01,m02 = R[0,0], R[0,1], R[0,2]
    m10,m11,m12 = R[1,0], R[1,1], R[1,2]
    m20,m21,m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = torch.sqrt(tr.clamp(min=-0.999) + 1) * 2
        q = torch.stack([0.25*s, (m21-m12)/s, (m02-m20)/s, (m10-m01)/s])
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1 + m00 - m11 - m22).clamp(min=1e-8) * 2
        q = torch.stack([(m21-m12)/s, 0.25*s, (m01+m10)/s, (m02+m20)/s])
    elif m11 > m22:
        s = torch.sqrt(1 + m11 - m00 - m22).clamp(min=1e-8) * 2
        q = torch.stack([(m02-m20)/s, (m01+m10)/s, 0.25*s, (m12+m21)/s])
    else:
        s = torch.sqrt(1 + m22 - m00 - m11).clamp(min=1e-8) * 2
        q = torch.stack([(m10-m01)/s, (m02+m20)/s, (m12+m21)/s, 0.25*s])
    q = F.normalize(q, dim=-1)
    if q[0] < 0: q = -q
    return q


def hamilton(a, b):
    aw,ax,ay,az = a[...,0], a[...,1], a[...,2], a[...,3]
    bw,bx,by,bz = b[...,0], b[...,1], b[...,2], b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def quat_rotate_pts(q, points):
    N = points.shape[0]
    q_b = q.unsqueeze(0).expand(N, -1)
    pq = torch.cat([torch.zeros(N, 1, device=points.device, dtype=points.dtype), points], dim=-1)
    q_conj = torch.cat([q_b[:, 0:1], -q_b[:, 1:]], dim=-1)
    return hamilton(hamilton(q_b, pq), q_conj)[:, 1:]


def kabsch_quat(src, tgt, mask):
    w = mask.float().unsqueeze(0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src.unsqueeze(0) * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tgt.unsqueeze(0) * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src.unsqueeze(0) - sm; tc = tgt.unsqueeze(0) - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=src.device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    q = rot_to_quat(R[0])
    t = tm.squeeze(0).squeeze(0) - quat_rotate_pts(q, sm.squeeze(0).squeeze(0).unsqueeze(0))[0]
    return q, t


def corr_sample_indices(orig_flat_idx, corr_target, corr_weight, pts_size, F_, P):
    device = orig_flat_idx.device
    idx0 = torch.linspace(0, P-1, pts_size, device=device).long()
    sampled_idx = torch.zeros(F_, pts_size, dtype=torch.long, device=device)
    matched = torch.zeros(F_-1, pts_size, dtype=torch.bool, device=device)
    sampled_idx[0] = idx0
    current_prov = orig_flat_idx[0, idx0].long()
    total_pts = corr_target.shape[-1]; raw_ppf = total_pts // F_
    for t in range(F_ - 1):
        next_prov = orig_flat_idx[t+1].long()
        reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
        reverse_map[next_prov] = torch.arange(P, device=device)
        tgt_flat = corr_target[current_prov]
        tgt_w = corr_weight[current_prov]
        tgt_flat_safe = tgt_flat.clamp(min=0)
        tgt_frame = tgt_flat // raw_ppf
        tgt_pos = reverse_map[tgt_flat_safe]
        valid = (tgt_flat >= 0) & (tgt_w > 0) & (tgt_frame == t+1) & (tgt_pos >= 0)
        next_idx = torch.randint(0, P, (pts_size,), device=device)
        next_idx[valid] = tgt_pos[valid]
        sampled_idx[t+1] = next_idx; matched[t] = valid
        current_prov = orig_flat_idx[t+1, next_idx].long()
    return sampled_idx, matched


PTS = 256
SAMPLE_IDX = 42  # arbitrary test sample

print(f'Loading sample {SAMPLE_IDX}...')
loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=True, assignment_mode='hungarian')
s = loader[SAMPLE_IDX]; pts_d = s[0]; label = s[1]
pts = pts_d['points'].cuda()
F_, P, C = pts.shape
xyz = pts[..., :3]
orig = pts_d['orig_flat_idx'].cuda()
ctgt = torch.from_numpy(pts_d['corr_full_target_idx']).long().cuda()
cw = torch.from_numpy(pts_d['corr_full_weight']).float().cuda()
sidx, matched = corr_sample_indices(orig, ctgt, cw, PTS, F_, P)
xyz_s = torch.gather(xyz, 1, sidx.unsqueeze(-1).expand(-1, -1, 3))  # (T,P,3)
print(f'sample label = {label}, frames={F_}, pts={PTS}')

# Compute stabilized: chain inverse so each frame lands in frame-0 coords
chain_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=xyz.device, dtype=xyz.dtype)
chain_t = torch.zeros(3, device=xyz.device, dtype=xyz.dtype)
xyz_stab = torch.zeros_like(xyz_s)
xyz_stab[0] = xyz_s[0]
for t in range(F_ - 1):
    qf, trf = kabsch_quat(xyz_s[t], xyz_s[t+1], matched[t])
    chain_q = hamilton(qf.unsqueeze(0), chain_q.unsqueeze(0))[0]
    chain_q = F.normalize(chain_q, dim=-1)
    chain_t = quat_rotate_pts(qf, chain_t.unsqueeze(0))[0] + trf
    # Inverse: xyz_stab[t+1] = chain_q* · (xyz[t+1] - chain_t) · chain_q
    q_conj = torch.cat([chain_q[0:1], -chain_q[1:]], dim=-1)
    shifted = xyz_s[t+1] - chain_t
    xyz_stab[t+1] = quat_rotate_pts(q_conj, shifted)

xyz_raw_np = xyz_s.cpu().numpy()       # (T,P,3)
xyz_stab_np = xyz_stab.cpu().numpy()


# === Plot: 2 main panels (overlaid frames) + grid of snapshots ===
fig = plt.figure(figsize=(18, 10))

# Top row: all 32 frames overlaid, colored by time
ax_raw = fig.add_subplot(2, 6, (1, 3), projection='3d')
ax_stab = fig.add_subplot(2, 6, (4, 6), projection='3d')
cmap = plt.cm.viridis
for t in range(F_):
    color = cmap(t / max(F_-1, 1))
    ax_raw.scatter(xyz_raw_np[t, :, 0], xyz_raw_np[t, :, 1], xyz_raw_np[t, :, 2],
                   c=[color], s=2, alpha=0.6)
    ax_stab.scatter(xyz_stab_np[t, :, 0], xyz_stab_np[t, :, 1], xyz_stab_np[t, :, 2],
                    c=[color], s=2, alpha=0.6)

ax_raw.set_title(f'Raw (label={label}) — all 32 frames overlaid\ncolor=time')
ax_stab.set_title(f'Rigid-stabilized — all frames in frame-0 coords\ncolor=time')

# Match axis ranges across both panels for fair compare
all_pts = np.concatenate([xyz_raw_np.reshape(-1, 3), xyz_stab_np.reshape(-1, 3)], 0)
mn = all_pts.min(0); mx = all_pts.max(0)
for ax in (ax_raw, ax_stab):
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

# Bottom row: snapshots at t=0, 8, 16, 24, 31 — interleaved raw/stab
snap_frames = [0, 8, 16, 24, 31]
for col, t in enumerate(snap_frames):
    ax = fig.add_subplot(2, 6, 7 + col, projection='3d')
    ax.scatter(xyz_raw_np[t, :, 0], xyz_raw_np[t, :, 1], xyz_raw_np[t, :, 2],
               c='red', s=3, alpha=0.5, label='raw')
    ax.scatter(xyz_stab_np[t, :, 0], xyz_stab_np[t, :, 1], xyz_stab_np[t, :, 2],
               c='blue', s=3, alpha=0.5, label='stab')
    ax.set_title(f't={t}')
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    if col == 0: ax.legend(loc='upper right', fontsize=8)

# Far right cell: drift magnitude over time
ax_drift = fig.add_subplot(2, 6, 12)
drift_mean = np.linalg.norm(xyz_raw_np - xyz_stab_np, axis=-1).mean(-1)
drift_max = np.linalg.norm(xyz_raw_np - xyz_stab_np, axis=-1).max(-1)
ax_drift.plot(drift_mean, label='mean ||raw-stab||', color='C0')
ax_drift.plot(drift_max, label='max', color='C3', linestyle='--')
ax_drift.set_xlabel('frame'); ax_drift.set_ylabel('drift')
ax_drift.set_title('Stabilization shift\n(raw - stab)'); ax_drift.legend(fontsize=8)
ax_drift.grid(True, alpha=0.3)

plt.suptitle(f'Rigid stabilization viz — sample {SAMPLE_IDX}, gesture class {label}', fontsize=14)
plt.tight_layout()
plt.savefig('/tmp/viz_stab.png', dpi=110, bbox_inches='tight')
print(f"saved /tmp/viz_stab.png  drift mean@t31={drift_mean[-1]:.3f}, max@t31={drift_max[-1]:.3f}")
