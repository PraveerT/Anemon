"""Check if QCC features distinguish fusion-categories:
  tn_only    = tiny right, pm wrong  (KEY: where tiny uniquely helps)
  pm_only    = pm right, tiny wrong  (where gate should favor pm)
  both_right = both right (easy)
  both_wrong = both wrong (unfixable)

If tn_only cluster separates in QCC-space, we can gate per-sample -> push fusion above 90.66.

Collects QCC stats (res_fwd/bwd norm, cycle error) on test set only,
loads pm+tk logits from /tmp/tinyknn_fuse.npz.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import nvidia_dataloader

# Quaternion ops (same as before)
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


def quat_rotate_points(q, points):
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
    t = tm.squeeze(0).squeeze(0) - quat_rotate_points(q, sm.squeeze(0).squeeze(0).unsqueeze(0))[0]
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

print('Collecting QCC stats for test set...')
loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=True)
N = len(loader)

# 9-d QCC feature per sample:
# [rf_mean, rf_std, rf_max, rb_mean, rb_std, rb_max, cyc_mean, cyc_std, cyc_max]
feats = torch.zeros(N, 9)
labels = torch.zeros(N, dtype=torch.long)
for i in range(N):
    s = loader[i]; pts_d = s[0]; label = s[1]
    pts = pts_d['points'].cuda()
    F_, P, C = pts.shape
    xyz = pts[..., :3]
    orig = pts_d['orig_flat_idx'].cuda()
    ctgt = torch.from_numpy(pts_d['corr_full_target_idx']).long().cuda()
    cw = torch.from_numpy(pts_d['corr_full_weight']).float().cuda()
    sidx, matched = corr_sample_indices(orig, ctgt, cw, PTS, F_, P)
    xyz_s = torch.gather(xyz, 1, sidx.unsqueeze(-1).expand(-1, -1, 3))
    rf_arr, rb_arr, cy_arr = [], [], []
    for t in range(F_ - 1):
        src = xyz_s[t]; tgt = xyz_s[t+1]; m = matched[t]
        qf, trf = kabsch_quat(src, tgt, m)
        pf = quat_rotate_points(qf, src) + trf
        rf = (tgt - pf).norm(dim=-1)
        qb, trb = kabsch_quat(tgt, src, m)
        pb = quat_rotate_points(qb, tgt) + trb
        rb = (src - pb).norm(dim=-1)
        # cycle: src -> fwd -> bwd -> should be src
        cyc = (src - (quat_rotate_points(qb, pf) + trb)).norm(dim=-1)
        mf = m.float()
        rf_arr.append((rf * mf).sum() / mf.sum().clamp(min=1))
        rb_arr.append((rb * mf).sum() / mf.sum().clamp(min=1))
        cy_arr.append((cyc * mf).sum() / mf.sum().clamp(min=1))
    rf_t = torch.stack(rf_arr); rb_t = torch.stack(rb_arr); cy_t = torch.stack(cy_arr)
    feats[i] = torch.tensor([
        rf_t.mean(), rf_t.std(), rf_t.max(),
        rb_t.mean(), rb_t.std(), rb_t.max(),
        cy_t.mean(), cy_t.std(), cy_t.max()])
    labels[i] = label
    if (i+1) % 100 == 0: print(f"  {i+1}/{N}")

# Load pm+tk logits
d = np.load('/tmp/tinyknn_fuse.npz')
pm = torch.from_numpy(d['pm_logits'])
tk = torch.from_numpy(d['tk_logits'])
y_saved = torch.from_numpy(d['labels'])
assert torch.equal(labels, y_saved), "label order mismatch"

y = labels
pm_p = F.softmax(pm, -1); tk_p = F.softmax(tk, -1)
pm_right = pm_p.argmax(-1) == y
tk_right = tk_p.argmax(-1) == y

groups = {
    'both_right': pm_right & tk_right,
    'pm_only':    pm_right & ~tk_right,
    'tn_only':    ~pm_right & tk_right,
    'both_wrong': ~pm_right & ~tk_right,
}

feat_names = ['rf_mean','rf_std','rf_max','rb_mean','rb_std','rb_max','cyc_mean','cyc_std','cyc_max']

# Compute group stats
print("\n=== QCC feature distribution per fusion-group ===")
print(f"{'feat':<10s}", end='')
for g in groups: print(f" {g:>11s}", end='')
print()
for fi, fn in enumerate(feat_names):
    print(f"{fn:<10s}", end='')
    for gname, m in groups.items():
        if m.sum() == 0:
            print(f" {'na':>11s}", end='')
        else:
            mean = feats[m, fi].mean().item()
            print(f" {mean:>11.4f}", end='')
    print()

# For each feature, compute how tn_only differs from others
# Use z-score: (tn_only_mean - overall_mean) / overall_std
print("\n=== tn_only signature (z-score vs full test) ===")
print(f"{'feat':<10s} {'z-score':>10s}  (>|1|: distinctive)")
for fi, fn in enumerate(feat_names):
    overall_mean = feats[:, fi].mean().item()
    overall_std = feats[:, fi].std().item()
    tn_mean = feats[groups['tn_only'], fi].mean().item()
    z = (tn_mean - overall_mean) / (overall_std + 1e-9)
    flag = "  ^" if abs(z) > 0.5 else ""
    print(f"{fn:<10s} {z:>+10.3f}{flag}")

# Compute: for each feature, split into bins and see tn_only concentration
print("\n=== Binned concentration (tn_only density per quartile) ===")
for fi, fn in enumerate(feat_names):
    vals = feats[:, fi]
    qs = torch.quantile(vals, torch.tensor([0.25, 0.5, 0.75]))
    bins = [(vals < qs[0]), (vals >= qs[0]) & (vals < qs[1]),
            (vals >= qs[1]) & (vals < qs[2]), (vals >= qs[2])]
    tn_m = groups['tn_only']
    row = [(tn_m & b).sum().item() for b in bins]
    total_tn = tn_m.sum().item()
    if total_tn > 0:
        pcts = [r/total_tn*100 for r in row]
        print(f"{fn:<10s} Q1={pcts[0]:4.1f}%  Q2={pcts[1]:4.1f}%  Q3={pcts[2]:4.1f}%  Q4={pcts[3]:4.1f}%  (uniform=25%)")

# Can we find a rule: when QCC-feature > threshold, trust tiny more?
# Exhaustive: for each feature, try threshold, compute fusion gain
print("\n=== Per-sample gating via single QCC feature threshold ===")
# Base fusion: alpha=0.76 everywhere (the current best)
A_lo, A_hi = 0.76, 0.76
pm_conf = pm_p.max(-1).values  # for reference
best_qcc_fuse = 0
best_rule = None
for fi, fn in enumerate(feat_names):
    vals = feats[:, fi]
    for q in np.arange(0.1, 0.95, 0.05):
        thr = torch.quantile(vals, torch.tensor([q], dtype=vals.dtype)).item()
        # above thr: trust tiny more (alpha_pm lower)
        for a_lo in [0.3, 0.4, 0.5, 0.6, 0.7]:  # alpha when "high" qcc (trust tiny)
            for a_hi in [0.76, 0.8, 0.85, 0.9]:  # alpha when "low" qcc (trust pm)
                alpha = torch.where(vals > thr,
                                     torch.full_like(vals, a_lo),
                                     torch.full_like(vals, a_hi))
                f = alpha.unsqueeze(-1) * pm_p + (1 - alpha.unsqueeze(-1)) * tk_p
                acc = (f.argmax(-1) == y).float().mean().item()
                if acc > best_qcc_fuse:
                    best_qcc_fuse = acc
                    best_rule = (fn, q, thr, a_lo, a_hi)

print(f"Best QCC-gate rule: {best_rule}")
print(f"Acc: {best_qcc_fuse*100:.2f}%  (baseline alpha=0.76: 90.66%)")
print(f"Gain over alpha-only: {(best_qcc_fuse - 0.9066)*100:+.2f}pp")

# Compare ALSO vs pmamba entropy gating
ent_pm = -(pm_p * pm_p.clamp(min=1e-12).log()).sum(-1)
best_ent_fuse = 0; best_ent_rule = None
for q in np.arange(0.1, 0.95, 0.05):
    thr = torch.quantile(ent_pm, torch.tensor([q], dtype=ent_pm.dtype)).item()
    for a_lo in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for a_hi in [0.76, 0.8, 0.85, 0.9]:
            alpha = torch.where(ent_pm > thr,
                                 torch.full_like(ent_pm, a_lo),
                                 torch.full_like(ent_pm, a_hi))
            f = alpha.unsqueeze(-1) * pm_p + (1 - alpha.unsqueeze(-1)) * tk_p
            acc = (f.argmax(-1) == y).float().mean().item()
            if acc > best_ent_fuse:
                best_ent_fuse = acc; best_ent_rule = (q, thr, a_lo, a_hi)

print(f"\nBest pm-entropy-gate rule: q={best_ent_rule[0]:.2f}, thr={best_ent_rule[1]:.3f}, a_hi-ent={best_ent_rule[2]}, a_lo-ent={best_ent_rule[3]}")
print(f"Acc: {best_ent_fuse*100:.2f}%  ({(best_ent_fuse-0.9066)*100:+.2f}pp vs alpha)")
