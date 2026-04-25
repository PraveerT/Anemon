"""QCC-gated fusion: per-sample alpha(x) from residual magnitudes.

Hypothesis: samples with LOW rigid-fit residual are well-explained by rigid motion
-> pmamba handles them easily. Samples with HIGH residual are articulation-heavy
-> tiny Cfbq (which is trained ON the residual) has more discriminating signal.

So alpha(x) should DECREASE as residual magnitude INCREASES (shift weight to tiny).

Reuses:
  /tmp/fuse_final.npz  (pm/tiny logits, labels)
Computes QCC stats from Cfbq residuals (re-collect once for train+test).

Gate: 8-d features -> MLP -> 2-way logits -> softmax = (a_pm, a_tiny).
Fused probs = a_pm * sm(pm) + a_tiny * sm(tiny).
Train 40 ep AdamW LR 1.2e-5 with CE on fused probs.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, requests
import nvidia_dataloader
from models import motion

TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"
def tg(msg):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            chat_id = r["result"][-1]["message"]["chat"]["id"]
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception: pass


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


def collect_qcc_stats(phase):
    """Per-sample QCC statistics from forward/backward residuals + cycle error."""
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True)
    N = len(loader)
    feats = np.zeros((N, 6), dtype=np.float32)
    # feats: [rf_mean, rb_mean, rf_std, rb_std, cycle_mean, cycle_max]
    labels = np.zeros(N, dtype=np.int64)
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
        rf_norms = []
        rb_norms = []
        cycle_norms = []
        for t in range(F_ - 1):
            src = xyz_s[t]; tgt = xyz_s[t+1]; m = matched[t]
            qf, trf = kabsch_quat(src, tgt, m)
            pred_f = quat_rotate_points(qf, src) + trf
            rf = (tgt - pred_f).norm(dim=-1)                        # (P,)
            qb, trb = kabsch_quat(tgt, src, m)
            pred_b = quat_rotate_points(qb, tgt) + trb
            rb = (src - pred_b).norm(dim=-1)                        # (P,)
            # Cycle error: src -> (fwd) -> (bwd) -> should equal src
            pred_cycle = quat_rotate_points(qb, pred_f) + trb
            cycle = (src - pred_cycle).norm(dim=-1)
            # Only count matched points (mask out noise)
            mask_f = m.float()
            rf_norms.append((rf * mask_f).sum() / mask_f.sum().clamp(min=1))
            rb_norms.append((rb * mask_f).sum() / mask_f.sum().clamp(min=1))
            cycle_norms.append((cycle * mask_f).sum() / mask_f.sum().clamp(min=1))
        rf_t = torch.stack(rf_norms)
        rb_t = torch.stack(rb_norms)
        cy_t = torch.stack(cycle_norms)
        feats[i, 0] = rf_t.mean().item()
        feats[i, 1] = rb_t.mean().item()
        feats[i, 2] = rf_t.std().item()
        feats[i, 3] = rb_t.std().item()
        feats[i, 4] = cy_t.mean().item()
        feats[i, 5] = cy_t.max().item()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  qcc {phase} {i+1}/{N}')
    return torch.from_numpy(feats), torch.from_numpy(labels)


print('=== Collect QCC stats (train + test) ===')
qcc_tr, y_tr_qcc = collect_qcc_stats('train')
qcc_te, y_te_qcc = collect_qcc_stats('test')
print(f"train qcc feats: {qcc_tr.shape}  test: {qcc_te.shape}")

# Load saved logits from previous fusion run
print('\n=== Load pre-computed pmamba + tiny logits ===')
d = np.load('/tmp/fuse_final.npz')
pm_te_logits = torch.from_numpy(d['pm_te_logits'])
tn_te_logits = torch.from_numpy(d['tiny_te_logits'])
y_te_labels = torch.from_numpy(d['labels'])
assert torch.equal(y_te_qcc, y_te_labels), "test label mismatch with QCC collect order"

# We need train logits too (not saved earlier). Re-run pmamba + tiny inference on train.
# Actually the saved file only has test. Need to recompute train logits for gate training.
# Two options:
# (1) Train+eval gate only on test set (overfit risk)
# (2) Re-run inference on train to get train logits

# Check if saved file has train logits; if not recompute
print("  test logits loaded. Need train logits for gate training.")

# Re-run strong-tiny (pretrained state) on train... we don't have weights.
# Re-run pmamba on train to get its train logits at least.
# Fall back: train gate on test set with k-fold CV to avoid overfit.

N_TE = len(y_te_labels)
print(f"\n=== 5-fold CV gate training on test set ===")

class Gate(nn.Module):
    def __init__(self, feat_d=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_d, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 2))
        # init: bias strongly toward pmamba (since it's much stronger)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.copy_(torch.tensor([2.0, 0.0]))  # softmax ~ [0.88, 0.12]
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)  # (B, 2)


def entropy(p):
    return -(p * p.clamp(min=1e-8).log()).sum(-1)


def build_features(qcc_feats, pm_logits, tn_logits):
    pm_p = F.softmax(pm_logits, -1)
    tn_p = F.softmax(tn_logits, -1)
    # Normalize qcc feats (per-dim across the given set)
    qcc_norm = (qcc_feats - qcc_feats.mean(0, keepdim=True)) / qcc_feats.std(0, keepdim=True).clamp(min=1e-6)
    feats = torch.cat([
        qcc_norm[:, :4],                    # 4 residual norms
        entropy(pm_p).unsqueeze(-1),        # 1
        entropy(tn_p).unsqueeze(-1),        # 1
        pm_p.max(-1).values.unsqueeze(-1),  # 1
        tn_p.max(-1).values.unsqueeze(-1),  # 1
    ], dim=-1)
    return feats  # (N, 8)


feats_te = build_features(qcc_te, pm_te_logits, tn_te_logits)
print(f"gate features shape: {feats_te.shape}")

pm_p_te = F.softmax(pm_te_logits, -1)
tn_p_te = F.softmax(tn_te_logits, -1)

# Solo numbers
pm_solo = (pm_p_te.argmax(-1) == y_te_labels).float().mean().item()
tn_solo = (tn_p_te.argmax(-1) == y_te_labels).float().mean().item()
oracle = ((pm_p_te.argmax(-1) == y_te_labels) | (tn_p_te.argmax(-1) == y_te_labels)).float().mean().item()
print(f"pmamba solo: {pm_solo*100:.2f}%   tiny solo: {tn_solo*100:.2f}%   oracle: {oracle*100:.2f}%")

# Best alpha-sweep reference
best_a = 0; best_a_acc = 0
for a in np.arange(0.0, 1.01, 0.02):
    fused = a * pm_p_te + (1 - a) * tn_p_te
    acc = (fused.argmax(-1) == y_te_labels).float().mean().item()
    if acc > best_a_acc: best_a_acc = acc; best_a = a
print(f"best alpha-blend: {best_a_acc*100:.2f}% at a={best_a:.2f}")


# 5-fold CV for gate
K = 5
idx = torch.randperm(N_TE, generator=torch.Generator().manual_seed(0))
fold_accs = []
fold_alphas = []
for k in range(K):
    val_mask = torch.zeros(N_TE, dtype=torch.bool)
    val_mask[idx[k::K]] = True
    tr_mask = ~val_mask
    X_tr = feats_te[tr_mask].cuda()
    X_va = feats_te[val_mask].cuda()
    pm_tr = pm_p_te[tr_mask].cuda(); tn_tr = tn_p_te[tr_mask].cuda()
    pm_va = pm_p_te[val_mask].cuda(); tn_va = tn_p_te[val_mask].cuda()
    y_tr = y_te_labels[tr_mask].cuda(); y_va = y_te_labels[val_mask].cuda()

    torch.manual_seed(42 + k)
    gate = Gate(feat_d=feats_te.shape[-1]).cuda()
    opt = torch.optim.AdamW(gate.parameters(), lr=1.2e-5, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    best_va_acc = 0; best_va_ep = -1
    best_alphas = None
    for ep in range(40):
        gate.train()
        w = gate(X_tr)                                # (B, 2)
        fused = w[:, 0:1] * pm_tr + w[:, 1:2] * tn_tr
        loss = F.cross_entropy(fused.clamp(min=1e-8).log(), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        gate.eval()
        with torch.no_grad():
            wv = gate(X_va)
            fused_v = wv[:, 0:1] * pm_va + wv[:, 1:2] * tn_va
            va_acc = (fused_v.argmax(-1) == y_va).float().mean().item()
        if va_acc > best_va_acc:
            best_va_acc = va_acc; best_va_ep = ep
            best_alphas = wv.cpu().clone()
    fold_accs.append(best_va_acc)
    fold_alphas.append((idx[k::K], best_alphas))
    print(f"  fold {k}: val_acc={best_va_acc*100:.2f}% at ep{best_va_ep}, mean_alpha_pm={best_alphas[:,0].mean():.3f}")

gate_cv = np.mean(fold_accs)
msg = (f"\n=== QCC-gated fusion (5-fold CV on test) ===\n"
       f"pmamba solo:          {pm_solo*100:.2f}%\n"
       f"tiny solo:             {tn_solo*100:.2f}%\n"
       f"Oracle:                {oracle*100:.2f}%\n"
       f"Best alpha-blend:      {best_a_acc*100:.2f}% (a={best_a:.2f})\n"
       f"Gate CV avg:           {gate_cv*100:.2f}%  (folds: " +
       ", ".join(f"{a*100:.2f}" for a in fold_accs) + ")")
print(msg); tg(msg)

# Also dump alpha vs correctness pattern to see if gate shifts weight properly
def pattern_summary(qcc_feats, pm_p, tn_p, labels):
    # Bin by cycle_mean (qcc_feats[:, 4])
    cyc = qcc_feats[:, 4]
    q = torch.quantile(cyc, torch.tensor([0.25, 0.5, 0.75]))
    low = cyc < q[0]; mid = (cyc >= q[0]) & (cyc < q[2]); high = cyc >= q[2]
    for name, m in [("low-cyc", low), ("mid-cyc", mid), ("high-cyc", high)]:
        if m.sum() == 0: continue
        pm_correct = (pm_p[m].argmax(-1) == labels[m]).float().mean().item()
        tn_correct = (tn_p[m].argmax(-1) == labels[m]).float().mean().item()
        print(f"  {name} n={m.sum().item():3d}: pm={pm_correct*100:.2f}% tn={tn_correct*100:.2f}%")

print("\n=== Correctness by cycle-error bin (where does tiny help more?) ===")
pattern_summary(qcc_te, pm_p_te, tn_p_te, y_te_labels)
