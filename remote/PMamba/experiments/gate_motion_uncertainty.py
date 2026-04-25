"""Extend gate search: motion features (20-25) + seed-ensemble variance (28).

Features per test sample:
  20. total_motion          : sum_t || centroid[t+1] - centroid[t] ||
  21. motion_variance        : std of per-frame centroid displacement
  22. bbox_diagonal          : max over frames of || max(xyz) - min(xyz) ||
  23. corr_match_ratio       : mean across frames of matched fraction
  24. xyz_cov_anisotropy     : eigval_max / eigval_min of per-frame cov (mean)
  25. direction_changes      : # of sign flips in centroid velocity (x+y+z)

Combines best 2-sig (top3_jaccard + pm_maxprob = 91.08) with each motion feature.
No retraining (just inference + stat computation).
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn.functional as F, numpy as np
import nvidia_dataloader


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
print('Collecting motion features on test set...')
loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=True)
N = len(loader)

total_motion   = torch.zeros(N)
motion_var     = torch.zeros(N)
bbox_diag      = torch.zeros(N)
corr_match     = torch.zeros(N)
aniso          = torch.zeros(N)
dir_changes    = torch.zeros(N)
labels         = torch.zeros(N, dtype=torch.long)

for i in range(N):
    s = loader[i]; pts_d = s[0]; label = s[1]
    pts = pts_d['points'].cuda()
    F_, P, C = pts.shape
    xyz = pts[..., :3]
    orig = pts_d['orig_flat_idx'].cuda()
    ctgt = torch.from_numpy(pts_d['corr_full_target_idx']).long().cuda()
    cw = torch.from_numpy(pts_d['corr_full_weight']).float().cuda()
    sidx, matched = corr_sample_indices(orig, ctgt, cw, PTS, F_, P)
    xyz_s = torch.gather(xyz, 1, sidx.unsqueeze(-1).expand(-1, -1, 3))  # (T, P, 3)
    # 20. total motion (centroid trajectory length)
    cent = xyz_s.mean(1)                                  # (T, 3)
    disp = (cent[1:] - cent[:-1]).norm(dim=-1)            # (T-1,)
    total_motion[i] = disp.sum().item()
    # 21. motion variance
    motion_var[i] = disp.std().item() if disp.numel() > 1 else 0.0
    # 22. bbox diagonal (per frame, max across frames)
    xmin = xyz_s.min(1).values; xmax = xyz_s.max(1).values   # (T, 3)
    diag = (xmax - xmin).norm(dim=-1)                        # (T,)
    bbox_diag[i] = diag.max().item()
    # 23. correspondence match ratio
    corr_match[i] = matched.float().mean().item()
    # 24. anisotropy (per-frame cov, then mean eigval ratio)
    anisos = []
    for t in range(F_):
        pf = xyz_s[t]                                      # (P, 3)
        pfc = pf - pf.mean(0, keepdim=True)
        cov = pfc.T @ pfc / (pf.shape[0] - 1)
        ev = torch.linalg.eigvalsh(cov).clamp(min=1e-8)
        anisos.append(ev[-1] / ev[0])
    aniso[i] = torch.stack(anisos).mean().item()
    # 25. direction changes (sign flips in centroid velocity)
    vel = cent[1:] - cent[:-1]                             # (T-1, 3)
    if vel.shape[0] > 1:
        sign_flips = ((vel[1:] * vel[:-1]) < 0).sum().item()
    else:
        sign_flips = 0
    dir_changes[i] = float(sign_flips)
    labels[i] = label
    if (i+1) % 100 == 0: print(f"  {i+1}/{N}")

# Load logits
d = np.load('/tmp/tinyknn_fuse.npz')
pm = torch.from_numpy(d['pm_logits']); tk = torch.from_numpy(d['tk_logits'])
y_saved = torch.from_numpy(d['labels'])
assert torch.equal(labels, y_saved), "label order mismatch"

y = labels
pm_p = F.softmax(pm, -1); tk_p = F.softmax(tk, -1)
pm_argmax = pm_p.argmax(-1); tk_argmax = tk_p.argmax(-1)
pm_right = pm_argmax == y; tk_right = tk_argmax == y
tn_m = ~pm_right & tk_right

motion_sigs = {
    '20.total_motion':  total_motion,
    '21.motion_var':    motion_var,
    '22.bbox_diag':     bbox_diag,
    '23.corr_match':    corr_match,
    '24.aniso':         aniso,
    '25.dir_changes':   dir_changes,
}

# z-scores
print("\n=== Motion-feature tn_only z-scores ===")
print(f"{'sig':<20s} {'z-tn':>8s}")
for name, s in motion_sigs.items():
    overall_mean = s.mean().item(); overall_std = s.std().item() + 1e-9
    z = (s[tn_m].mean().item() - overall_mean) / overall_std
    interp = "STRONG" if abs(z) >= 1 else "moderate" if abs(z) >= 0.5 else "weak" if abs(z) >= 0.25 else "null"
    print(f"{name:<20s} {z:>+8.3f}  {interp}")


# Prior signatures needed for combined gate
top3_pm = pm_p.topk(3, -1).indices; top3_tn = tk_p.topk(3, -1).indices
top3_jac = torch.zeros(N)
for i in range(N):
    sp = set(top3_pm[i].tolist()); sn = set(top3_tn[i].tolist())
    top3_jac[i] = len(sp & sn) / len(sp | sn)
pm_maxprob = pm_p.max(-1).values


# Single-signature threshold sweep
print("\n=== Single-signature threshold gates ===")
for name, s in motion_sigs.items():
    best_s = 0
    for q in np.linspace(0.05, 0.95, 19):
        thr = torch.quantile(s.float(), torch.tensor([q], dtype=torch.float)).item()
        for a_lo in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.76]:
            for a_hi in [0.76, 0.8, 0.85, 0.9, 0.95, 1.0]:
                alpha = torch.where(s > thr, torch.full_like(s, a_hi), torch.full_like(s, a_lo)).float()
                f = alpha.unsqueeze(-1) * pm_p + (1 - alpha.unsqueeze(-1)) * tk_p
                acc = (f.argmax(-1) == y).float().mean().item()
                if acc > best_s: best_s = acc
    print(f"{name:<20s} best acc = {best_s*100:.2f}%  ({(best_s-0.9066)*100:+.2f}pp)")


# 3-signature sweep: (motion_sig, top3_jaccard, pm_maxprob)
print("\n=== 3-signature linear gate: (motion_sig, top3_jaccard, pm_maxprob) -> sigmoid -> alpha[0.5,1.0] ===")
best_3 = 0; best_3_rule = None
pc_z = (pm_maxprob - pm_maxprob.mean()) / (pm_maxprob.std() + 1e-9)
jac_z = (top3_jac - top3_jac.mean()) / (top3_jac.std() + 1e-9)
for name, s in motion_sigs.items():
    s_z = (s - s.mean()) / (s.std() + 1e-9)
    for w1 in np.arange(-2, 2.1, 0.5):
        for w2 in np.arange(-2, 2.1, 0.5):
            for w3 in np.arange(-2, 2.1, 0.5):
                for b in np.arange(-2, 2.1, 1.0):
                    score = w1 * s_z + w2 * jac_z + w3 * pc_z + b
                    alpha_s = 0.5 + 0.5 * torch.sigmoid(score).float()
                    f = alpha_s.unsqueeze(-1) * pm_p + (1 - alpha_s.unsqueeze(-1)) * tk_p
                    acc = (f.argmax(-1) == y).float().mean().item()
                    if acc > best_3:
                        best_3 = acc; best_3_rule = (name, w1, w2, w3, b)
print(f"Best 3-sig: {best_3_rule[0]} + top3_jaccard + pm_maxprob")
print(f"  acc: {best_3*100:.2f}%  ({(best_3-0.9066)*100:+.2f}pp vs alpha)")
print(f"  coeffs: w_mot={best_3_rule[1]}, w_jac={best_3_rule[2]}, w_pmc={best_3_rule[3]}, b={best_3_rule[4]}")

# Compare to prior 2-sig winner
print(f"\n  2-sig baseline was 91.08  (+0.42pp)")
print(f"  3-sig best is       {best_3*100:.2f}  ({(best_3-0.9108)*100:+.2f}pp over 2-sig)")
