"""V0 baseline cache.

Skip retraining V0 (Cfbq alone, TinyKNN k=16, 120 ep) when (phase_a_seed,
train_seed) hits a cached result. Saves model state_dict, test logits, best
accuracy, and best epoch.

Usage in a variant script:
    from qcc_cache import get_or_train_v0, train_one_120ep

    v0_acc, v0_ep, v0_lg, v0_st = get_or_train_v0(
        phase_a_seed=0, train_seed=1,
        train_data=(C_tr, y_tr), test_data=(C_te, y_te),
        TinyV0_class=TinyV0,
    )

The cache stores a SHA-256-like fingerprint of the train tensor to detect
silent corruption between runs (e.g., if Phase A code changed and produced
different tensors despite the same seed).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


CACHE_ROOT = '/notebooks/PMamba/experiments/work_dir/qcc_branch/v0_cache'
os.makedirs(CACHE_ROOT, exist_ok=True)


def _fingerprint(tensor):
    """Cheap fingerprint to detect silent tensor changes between scripts."""
    flat = tensor.reshape(-1).float()
    n = flat.numel()
    samples = torch.linspace(0, n - 1, 64).long().clamp(max=n - 1)
    parts = flat[samples]
    digest_int = int((parts * 1e6).abs().sum().item()) % (1 << 31)
    return f"{tensor.shape}_{digest_int:08x}"


def cache_path(phase_a_seed, train_seed, fp):
    return f"{CACHE_ROOT}/v0_pa{phase_a_seed}_tr{train_seed}_{fp}.pt"


def train_one_120ep(model, X_tr, y_tr, X_te, y_te, train_seed,
                    epochs=120, bs=16, lr=2e-3):
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / 5 if ep < 5
        else 0.5 * (1 + math.cos(math.pi * (ep - 5) / max(1, epochs - 5)))
    )
    Xtr = X_tr.cuda(); ytr_c = y_tr.cuda()
    Xte = X_te.cuda(); yte_c = y_te.cuda()
    best, best_ep, best_logits, best_state = 0.0, -1, None, None
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(train_seed * 1000 + ep)
        perm = torch.randperm(len(X_tr), generator=g)
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i+bs]
            xb = Xtr[idx]; yb = ytr_c[idx]
            B_, T_, P_, _ = xb.shape
            xb = xb * (torch.rand(B_, 1, P_, 1, device=xb.device) > 0.10).float()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(X_te), 32):
                lg.append(model(Xte[i:i+32]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == yte_c).float().mean().item()
        if acc > best:
            best = acc; best_ep = ep; best_logits = lg.clone()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  V0 ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})")
    return best, best_ep, best_logits, best_state


def get_or_train_v0(phase_a_seed, train_seed,
                    train_data, test_data,
                    TinyV0_class, in_ch=10, k=16,
                    epochs=120, bs=16, lr=2e-3,
                    force=False):
    """Returns (acc, ep, logits, state_dict). Trains and caches if missing.

    train_data, test_data: tuples (X, y) of cuda-able tensors.
    """
    X_tr, y_tr = train_data
    X_te, y_te = test_data
    fp = _fingerprint(X_tr) + "_" + _fingerprint(X_te)
    path = cache_path(phase_a_seed, train_seed, fp)
    if not force and os.path.exists(path):
        print(f"V0 cache hit: {path}")
        ckpt = torch.load(path, map_location='cpu')
        return ckpt['acc'], ckpt['ep'], ckpt['logits'], ckpt['state_dict']
    print(f"V0 cache miss; training (target: {path})")
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    model = TinyV0_class(in_ch=in_ch, k=k).cuda()
    acc, ep, logits, state = train_one_120ep(
        model, X_tr, y_tr, X_te, y_te, train_seed, epochs=epochs, bs=bs, lr=lr
    )
    torch.save({
        'acc': acc, 'ep': ep, 'logits': logits, 'state_dict': state,
        'phase_a_seed': phase_a_seed, 'train_seed': train_seed,
        'fingerprint': fp,
    }, path)
    return acc, ep, logits, state
