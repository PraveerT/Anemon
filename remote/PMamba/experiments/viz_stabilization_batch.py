"""Render rigid-stabilization viz for 6 samples spanning the test set.

Each row: raw overlay | stab overlay | drift curve.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn.functional as F, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def stabilize(xyz_s, matched):
    F_, P, _ = xyz_s.shape
    chain_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=xyz_s.device, dtype=xyz_s.dtype)
    chain_t = torch.zeros(3, device=xyz_s.device, dtype=xyz_s.dtype)
    out = torch.zeros_like(xyz_s)
    out[0] = xyz_s[0]
    for t in range(F_ - 1):
        qf, trf = kabsch_quat(xyz_s[t], xyz_s[t+1], matched[t])
        chain_q = hamilton(qf.unsqueeze(0), chain_q.unsqueeze(0))[0]
        chain_q = F.normalize(chain_q, dim=-1)
        chain_t = quat_rotate_pts(qf, chain_t.unsqueeze(0))[0] + trf
        q_conj = torch.cat([chain_q[0:1], -chain_q[1:]], dim=-1)
        out[t+1] = quat_rotate_pts(q_conj, xyz_s[t+1] - chain_t)
    return out


PTS = 256
SAMPLES = [10, 75, 150, 225, 350, 425]

print('Loading dataset...')
loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=True, assignment_mode='hungarian')

fig = plt.figure(figsize=(20, 4 * len(SAMPLES)))
cmap = plt.cm.viridis

for row, idx in enumerate(SAMPLES):
    s = loader[idx]; pts_d = s[0]; label = s[1]
    pts = pts_d['points'].cuda()
    F_, P, C = pts.shape
    xyz = pts[..., :3]
    orig = pts_d['orig_flat_idx'].cuda()
    ctgt = torch.from_numpy(pts_d['corr_full_target_idx']).long().cuda()
    cw = torch.from_numpy(pts_d['corr_full_weight']).float().cuda()
    sidx, matched = corr_sample_indices(orig, ctgt, cw, PTS, F_, P)
    xyz_s = torch.gather(xyz, 1, sidx.unsqueeze(-1).expand(-1, -1, 3))
    xyz_stab = stabilize(xyz_s, matched)
    raw = xyz_s.cpu().numpy(); stab = xyz_stab.cpu().numpy()
    print(f'  sample {idx} class {label}  drift_max={np.linalg.norm(raw-stab,axis=-1).max():.3f}')

    all_pts = np.concatenate([raw.reshape(-1, 3), stab.reshape(-1, 3)], 0)
    mn = all_pts.min(0); mx = all_pts.max(0)

    ax_raw = fig.add_subplot(len(SAMPLES), 3, 3*row + 1, projection='3d')
    ax_stab = fig.add_subplot(len(SAMPLES), 3, 3*row + 2, projection='3d')
    ax_drift = fig.add_subplot(len(SAMPLES), 3, 3*row + 3)
    for t in range(F_):
        color = cmap(t / max(F_-1, 1))
        ax_raw.scatter(raw[t, :, 0], raw[t, :, 1], raw[t, :, 2], c=[color], s=2, alpha=0.5)
        ax_stab.scatter(stab[t, :, 0], stab[t, :, 1], stab[t, :, 2], c=[color], s=2, alpha=0.5)
    ax_raw.set_title(f'idx {idx} class {label} — RAW')
    ax_stab.set_title(f'idx {idx} class {label} — STABILIZED')
    for ax in (ax_raw, ax_stab):
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])

    drift_mean = np.linalg.norm(raw - stab, axis=-1).mean(-1)
    drift_max  = np.linalg.norm(raw - stab, axis=-1).max(-1)
    raw_mean   = np.linalg.norm(raw - raw[0:1], axis=-1).mean(-1)  # raw movement vs frame 0
    ax_drift.plot(drift_mean, label='stab shift mean', color='C0')
    ax_drift.plot(drift_max, label='stab shift max', color='C0', linestyle='--', alpha=0.5)
    ax_drift.plot(raw_mean, label='raw |xyz[t]-xyz[0]| mean', color='C3')
    ax_drift.set_xlabel('frame'); ax_drift.set_ylabel('distance')
    ax_drift.set_title(f'drift profile idx {idx}')
    ax_drift.legend(fontsize=7); ax_drift.grid(True, alpha=0.3)

plt.suptitle('Rigid stabilization — 6 test samples (raw vs stabilized, drift profile)', y=1.001, fontsize=14)
plt.tight_layout()
plt.savefig('/tmp/viz_stab_batch.png', dpi=110, bbox_inches='tight')
print(f"saved /tmp/viz_stab_batch.png")
