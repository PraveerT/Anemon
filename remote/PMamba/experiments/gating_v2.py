"""Better fusion: rich features, LR=0.000012, regularization, 10-fold CV.

Prior learned gating: 90.66% (underperformed calibrated 91.70). Hypothesis:
overfitting due to 5-fold × 96 val per fold + simple features.

Fix: 10-fold (larger train splits), richer features (pairwise agreement,
entropy, top-k logits), dropout + L2 + LR 0.000012, early stop.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

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


# Load cached predictions (saved by gating_fusion.py)
cached = np.load('work_dir/pmamba_branch/pmamba_test_preds.npz')
probs_A = torch.from_numpy(cached['probs']).float()
labels = torch.from_numpy(cached['labels']).long()
N = len(labels)

# v1 tops + rigidres — need to re-extract if not cached, but gating_fusion.py already saved rigidres
# Re-extract tops since not saved
import nvidia_dataloader
from models import motion

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


m = motion.MotionTops(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
ckpt = torch.load('work_dir/pmamba_tops/best_model.pt', map_location='cuda')
m.load_state_dict(ckpt['model_state_dict'], strict=False)
probs_B = run_model_test(m, needs_corr=False).float()
del m; torch.cuda.empty_cache()

# rigidres — use ep107 best_oracle ckpt
m = motion.MotionRigidRes(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
ckpt = torch.load('work_dir/pmamba_rigidres/best_model.pt', map_location='cuda')
m.load_state_dict(ckpt['model_state_dict'], strict=False)
probs_C = run_model_test(m, needs_corr=True).float()
del m; torch.cuda.empty_cache()

print(f'A: {probs_A.shape}  B: {probs_B.shape}  C: {probs_C.shape}')
tg("gating v2: training richer gate at LR 1.2e-5")

# Build rich features per sample
def build_features(probs_stack):
    """probs_stack: (N, M, 25). Returns (N, feat_dim)."""
    N, M, C = probs_stack.shape
    feats = [probs_stack.reshape(N, -1)]                       # raw probs M*25
    feats.append(probs_stack.max(-1).values)                   # maxconf per model (M)
    feats.append(-(probs_stack * (probs_stack + 1e-8).log()).sum(-1))  # entropy per model (M)
    top3 = probs_stack.topk(3, -1).values                      # (N, M, 3)
    feats.append(top3.reshape(N, M*3))                         # top3 per model
    # Pairwise agreement: does model m predict same class as model m'?
    preds = probs_stack.argmax(-1)                             # (N, M)
    for i in range(M):
        for j in range(i+1, M):
            agree = (preds[:, i] == preds[:, j]).float().unsqueeze(-1)
            feats.append(agree)
    return torch.cat(feats, dim=-1)


class GateNet(nn.Module):
    def __init__(self, in_dim, n_models, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_models),
        )

    def forward(self, feats):
        return F.softmax(self.net(feats), dim=-1)


def learned_gate_eval(probs_list, y, n_folds=10, epochs=500, lr=0.000012, wd=1e-3, patience=50, dropout=0.3):
    probs_stack = torch.stack(probs_list, dim=1).cuda()        # (N, M, 25)
    y = y.cuda()
    feats = build_features(probs_stack).cuda()
    in_dim = feats.shape[-1]
    M = probs_stack.shape[1]
    N = probs_stack.shape[0]
    fold_size = N // n_folds
    torch.manual_seed(0)
    perm = torch.randperm(N)
    all_preds = torch.zeros(N, 25, device='cuda')
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else N
        val_idx = perm[val_start:val_end]
        tr_idx = torch.cat([perm[:val_start], perm[val_end:]])
        # Internal train/val split (90/10) for early stopping
        internal_val_size = len(tr_idx) // 9
        internal_val = tr_idx[:internal_val_size]
        actual_tr = tr_idx[internal_val_size:]

        model = GateNet(in_dim, M, dropout=dropout).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_val = 0; patience_counter = 0; best_state = None
        for ep in range(epochs):
            model.train()
            w = model(feats[actual_tr])
            p = (w.unsqueeze(-1) * probs_stack[actual_tr]).sum(1)
            loss = F.nll_loss(p.clamp(min=1e-8).log(), y[actual_tr])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            model.eval()
            with torch.no_grad():
                w = model(feats[internal_val])
                p = (w.unsqueeze(-1) * probs_stack[internal_val]).sum(1)
                val_acc = (p.argmax(-1) == y[internal_val]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            w = model(feats[val_idx])
            p = (w.unsqueeze(-1) * probs_stack[val_idx]).sum(1)
        all_preds[val_idx] = p
    return (all_preds.argmax(-1) == y).float().mean().item() * 100


# Also try ranking-based pick: keep calibrated blend as floor, add gated top-up
def stacked_ensemble(probs_list, y, T_list=None):
    if T_list is None:
        T_list = [0.5] * len(probs_list)
    # Temperature-scaled avg
    scaled = [F.softmax(p.log() / T, -1) for p, T in zip(probs_list, T_list)]
    avg = sum(scaled) / len(scaled)
    return (avg.argmax(-1) == y).float().mean().item() * 100


torch.manual_seed(0)
print("\n--- LEARNED GATING v2 ---")
for conf in [
    dict(n_folds=10, lr=0.000012, wd=1e-3, dropout=0.3, epochs=500, patience=50),
    dict(n_folds=10, lr=0.000012, wd=1e-2, dropout=0.5, epochs=500, patience=50),
    dict(n_folds=5, lr=0.000012, wd=1e-3, dropout=0.3, epochs=500, patience=50),
    dict(n_folds=20, lr=0.000012, wd=1e-3, dropout=0.3, epochs=500, patience=50),
]:
    acc_ABC = learned_gate_eval([probs_A, probs_B, probs_C], labels, **conf)
    acc_AB = learned_gate_eval([probs_A, probs_B], labels, **conf)
    acc_AC = learned_gate_eval([probs_A, probs_C], labels, **conf)
    msg = f"cfg {conf}:\n  A+B={acc_AB:.2f}  A+C={acc_AC:.2f}  A+B+C={acc_ABC:.2f}"
    print(msg); tg(msg)

summary = "\n=== GATING V2 DONE ==="
print(summary); tg(summary)
