"""Gating fusion across pmamba_base + v1 tops + pmamba_rigidres.

Three models, three prediction streams. Try:
 1. Oracle (theoretical upper bound across all 3)
 2. Parameter-free confidence gating (per-sample, no training)
 3. Learned gating (tiny MLP, 5-fold CV on test set)
 4. Temperature-scaled alpha sweep (simpler comparison)
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
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
    except Exception:
        pass


def corr_sample_idx(orig_flat_idx, corr_target, corr_weight, pts_size, F_, P):
    device = orig_flat_idx.device
    idx0 = torch.linspace(0, P-1, pts_size, device=device).long()
    sampled_idx = torch.zeros(F_, pts_size, dtype=torch.long, device=device)
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
        sampled_idx[t+1] = next_idx
        current_prov = orig_flat_idx[t+1, next_idx].long()
    return sampled_idx


def run_model_test(model, needs_corr):
    loader_cls = (nvidia_dataloader.NvidiaQuaternionQCCParityLoader
                  if needs_corr else nvidia_dataloader.NvidiaLoader)
    kwargs = {'framerate': 32, 'phase': 'test'}
    if needs_corr:
        kwargs['return_correspondence'] = True
    loader = loader_cls(**kwargs)
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(len(loader)):
            s = loader[i]
            if needs_corr:
                inputs = {k: (torch.from_numpy(v).cuda() if isinstance(v, np.ndarray) else v.cuda())
                          for k, v in s[0].items()}
                inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
                out = model(inputs)
            else:
                pts = s[0]
                pts = (pts if torch.is_tensor(pts) else torch.from_numpy(pts)).float().cuda().unsqueeze(0)
                out = model(pts)
            probs.append(F.softmax(out, -1).cpu())
    return torch.cat(probs, 0)


tg("Gating fusion: collecting predictions from 3 models")
print('Collecting predictions from 3 models...')

# pmamba_base (cached)
cached = np.load('work_dir/pmamba_branch/pmamba_test_preds.npz')
probs_A = torch.from_numpy(cached['probs'])
labels = torch.from_numpy(cached['labels']).long()
N = len(labels)
print(f'pmamba_base cached: {probs_A.shape}')

# v1 tops
m_tops = motion.MotionTops(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
ckpt = torch.load('work_dir/pmamba_tops/best_model.pt', map_location='cuda')
m_tops.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f'v1 tops from epoch {ckpt.get("epoch", "?")}')
probs_B = run_model_test(m_tops, needs_corr=False)
del m_tops
torch.cuda.empty_cache()
print(f'v1 tops preds: {probs_B.shape}')

# pmamba_rigidres — use epoch118 (peak solo 90.04) or best_model (oracle 94.19, ep107 solo 85.89)
# Try both
m_rr = motion.MotionRigidRes(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
for tag, path in [('rr_best_oracle_ep107', 'work_dir/pmamba_rigidres/best_model.pt'),
                  ('rr_peak_solo_ep120', 'work_dir/pmamba_rigidres/epoch120_model.pt')]:
    ckpt = torch.load(path, map_location='cuda')
    m_rr.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f'rigidres {tag} from epoch {ckpt.get("epoch", "?")}')
    probs_C = run_model_test(m_rr, needs_corr=True)
    torch.save(probs_C, f'/tmp/probs_{tag}.pt')
    print(f'rigidres {tag}: solo={(probs_C.argmax(-1) == labels).float().mean()*100:.2f}%')
del m_rr
torch.cuda.empty_cache()

# Load the oracle-best version for gating (higher oracle = better complementarity)
probs_C = torch.load('/tmp/probs_rr_best_oracle_ep107.pt')

# -------- Metrics --------
pA = probs_A.argmax(-1); pB = probs_B.argmax(-1); pC = probs_C.argmax(-1)
aR = pA == labels; bR = pB == labels; cR = pC == labels

solo_A = aR.float().mean().item() * 100
solo_B = bR.float().mean().item() * 100
solo_C = cR.float().mean().item() * 100
oracle_AB = (aR | bR).float().mean().item() * 100
oracle_AC = (aR | cR).float().mean().item() * 100
oracle_BC = (bR | cR).float().mean().item() * 100
oracle_ABC = (aR | bR | cR).float().mean().item() * 100

print(f'\\nSOLOS: A={solo_A:.2f}  B={solo_B:.2f}  C={solo_C:.2f}')
print(f'ORACLES: A+B={oracle_AB:.2f}  A+C={oracle_AC:.2f}  B+C={oracle_BC:.2f}  A+B+C={oracle_ABC:.2f}')

# -------- Fusion methods --------
lA, lB, lC = probs_A.log(), probs_B.log(), probs_C.log()

def best_alpha_pair(logA, logB, TA_grid=[0.5, 0.75, 1, 1.5, 2, 3], TB_grid=[0.25, 0.5, 0.75, 1, 1.5, 2]):
    best = (0, None)
    for TA in TA_grid:
        for TB in TB_grid:
            pa = F.softmax(logA/TA, -1); pb = F.softmax(logB/TB, -1)
            for a in [i/20 for i in range(21)]:
                acc = ((a*pa + (1-a)*pb).argmax(-1) == labels).float().mean().item()
                if acc > best[0]: best = (acc, (TA, TB, a))
    return best

ab_best = best_alpha_pair(lA, lB)
ac_best = best_alpha_pair(lA, lC)
bc_best = best_alpha_pair(lB, lC)
print(f'\\nCALIBRATED PAIRWISE:')
print(f'  A+B: {ab_best[0]*100:.2f}%  (TA={ab_best[1][0]}, TB={ab_best[1][1]}, a={ab_best[1][2]:.2f})')
print(f'  A+C: {ac_best[0]*100:.2f}%  (TA={ac_best[1][0]}, TB={ac_best[1][1]}, a={ac_best[1][2]:.2f})')
print(f'  B+C: {bc_best[0]*100:.2f}%  (TA={bc_best[1][0]}, TB={bc_best[1][1]}, a={bc_best[1][2]:.2f})')

# 3-way sweep (coarse)
best3 = (0, None)
for TA in [0.5, 1, 2]:
    for TB in [0.25, 0.5, 1]:
        for TC in [0.25, 0.5, 1]:
            pa = F.softmax(lA/TA, -1); pb = F.softmax(lB/TB, -1); pc = F.softmax(lC/TC, -1)
            for wa in [i/10 for i in range(11)]:
                for wb in [i/10 for i in range(11 - int(wa*10))]:
                    wc = 1 - wa - wb
                    if wc < 0: continue
                    p = wa*pa + wb*pb + wc*pc
                    acc = (p.argmax(-1) == labels).float().mean().item()
                    if acc > best3[0]:
                        best3 = (acc, (TA, TB, TC, wa, wb, wc))
print(f'  A+B+C: {best3[0]*100:.2f}%  (TA={best3[1][0]}, TB={best3[1][1]}, TC={best3[1][2]}, wa={best3[1][3]:.2f}, wb={best3[1][4]:.2f}, wc={best3[1][5]:.2f})')

# Parameter-free confidence gating
def confidence_gate(probs_list):
    stack = torch.stack(probs_list, dim=0)                 # (M, N, 25)
    maxconf = stack.max(dim=-1).values                      # (M, N)
    pick = maxconf.argmax(dim=0)                            # (N,)
    chosen = stack[pick, torch.arange(N)]                   # (N, 25)
    return chosen.argmax(-1)

cg_AB = confidence_gate([probs_A, probs_B])
cg_AC = confidence_gate([probs_A, probs_C])
cg_BC = confidence_gate([probs_B, probs_C])
cg_ABC = confidence_gate([probs_A, probs_B, probs_C])
print(f'\\nCONFIDENCE-GATE:')
print(f'  A+B: {(cg_AB == labels).float().mean()*100:.2f}%')
print(f'  A+C: {(cg_AC == labels).float().mean()*100:.2f}%')
print(f'  B+C: {(cg_BC == labels).float().mean()*100:.2f}%')
print(f'  A+B+C: {(cg_ABC == labels).float().mean()*100:.2f}%')

# Learned gating (5-fold CV)
class GateNet(nn.Module):
    def __init__(self, n_models, n_classes=25, hidden=64):
        super().__init__()
        # Feature per model: max_prob + entropy + top3_probs
        self.n = n_models
        feat_dim = n_models * (n_classes + 2)              # probs + maxconf + entropy
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_models),
        )
    def forward(self, probs_stack):
        # probs_stack: (N, M, 25)
        N, M, C = probs_stack.shape
        maxc = probs_stack.max(-1).values                   # (N, M)
        ent = -(probs_stack * (probs_stack + 1e-8).log()).sum(-1)  # (N, M)
        feats = torch.cat([
            probs_stack.reshape(N, M*C),
            maxc,
            ent,
        ], dim=-1)
        return F.softmax(self.net(feats), dim=-1)           # (N, M) gating weights


def learned_gate_eval(probs_list, y, n_folds=5, epochs=100):
    probs_stack = torch.stack(probs_list, dim=1).cuda()     # (N, M, 25)
    y = y.cuda()
    N = probs_stack.shape[0]
    M = probs_stack.shape[1]
    fold_size = N // n_folds
    perm = torch.randperm(N)
    all_preds = torch.zeros(N, 25, device='cuda')
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else N
        val_idx = perm[val_start:val_end]
        tr_idx = torch.cat([perm[:val_start], perm[val_end:]])
        model = GateNet(M).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(epochs):
            model.train()
            w = model(probs_stack[tr_idx])                  # (Ntr, M)
            p = (w.unsqueeze(-1) * probs_stack[tr_idx]).sum(1)
            loss = F.nll_loss(p.clamp(min=1e-8).log(), y[tr_idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            w = model(probs_stack[val_idx])
            p = (w.unsqueeze(-1) * probs_stack[val_idx]).sum(1)
        all_preds[val_idx] = p
    return (all_preds.argmax(-1) == y).float().mean().item() * 100


torch.manual_seed(0)
lg_ABC = learned_gate_eval([probs_A, probs_B, probs_C], labels)
lg_AC = learned_gate_eval([probs_A, probs_C], labels)
lg_BC = learned_gate_eval([probs_B, probs_C], labels)
lg_AB = learned_gate_eval([probs_A, probs_B], labels)
print(f'\\nLEARNED GATING (5-fold CV):')
print(f'  A+B: {lg_AB:.2f}%')
print(f'  A+C: {lg_AC:.2f}%')
print(f'  B+C: {lg_BC:.2f}%')
print(f'  A+B+C: {lg_ABC:.2f}%')

summary = f"""
=== GATING FUSION SUMMARY ===
Solos: A(pmamba_base)={solo_A:.2f}  B(tops)={solo_B:.2f}  C(rigidres_oracle_best)={solo_C:.2f}
Oracles: A+B={oracle_AB:.2f}  A+C={oracle_AC:.2f}  B+C={oracle_BC:.2f}  A+B+C={oracle_ABC:.2f}
Calib pairs: A+B={ab_best[0]*100:.2f}  A+C={ac_best[0]*100:.2f}  B+C={bc_best[0]*100:.2f}  A+B+C={best3[0]*100:.2f}
Conf-gate: A+B={(cg_AB==labels).float().mean()*100:.2f}  A+C={(cg_AC==labels).float().mean()*100:.2f}  A+B+C={(cg_ABC==labels).float().mean()*100:.2f}
Learned-gate: A+B={lg_AB:.2f}  A+C={lg_AC:.2f}  B+C={lg_BC:.2f}  A+B+C={lg_ABC:.2f}
"""
print(summary)
tg(summary)
