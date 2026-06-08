"""Skew-TCC on MSRAction3D (cross-subject). Shared per-frame PointNet encoder;
three param-matched temporal heads differing ONLY in the temporal descriptor:
  base    : symmetric temporal pool (mean+max over T) -> order-blind
  sym_sh  : + SHARED-projector SYMMETRIC lagged cross-cov descriptor (control)
  skew_sh : + SHARED-projector ANTISYMMETRIC lagged cross-cov (Skew-TCC)
Same encoder/classifier sizes; only the descriptor matrix-part differs. Reports
cross-subject test acc, 3 seeds. Does antisymmetry beat the symmetric control on
REAL actions? (synthetic already showed skew_sh>>sym_sh when signal is antisym.)
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = 'cuda'
d = np.load('/notebooks/wt9191/experiments/msr_pc.npz')
X, A, S = d['X'], d['action'], d['subject']
T, N = X.shape[1], X.shape[2]
tr = np.isin(S, [1, 3, 5, 7, 9]); te = ~tr
Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(A[tr])
Xte, Yte = torch.tensor(X[te]), torch.tensor(A[te])
print('train', Xtr.shape, 'test', Xte.shape, 'T', T, 'N', N, flush=True)


class Net(nn.Module):
    def __init__(self, mode='base', D=128, r=16, lags=(1, 2, 3)):
        super().__init__()
        self.mode = mode; self.lags = lags
        self.is_skew = mode.startswith('skew')
        self.enc = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128),
                                 nn.ReLU(), nn.Linear(128, D))
        din = 2 * D
        if mode != 'base':
            self.W = nn.Linear(D, r, bias=False)                # SHARED projector
            self.ti = torch.tril_indices(r, r, -1 if self.is_skew else 0)
            din += self.ti.shape[1] * len(lags)
        self.cls = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(256, 20))

    def desc(self, f):                                          # f (B,T,D)
        z = (f - f.mean(1, keepdim=True)) / (f.std(1, keepdim=True) + 1e-5)
        U = self.W(z)
        outs = []
        for dd in self.lags:
            u, v = U[:, :T - dd], U[:, dd:]
            C = torch.einsum('bti,btj->bij', u, v) / (T - dd)
            M = (C - C.transpose(1, 2)) * .5 if self.is_skew else (C + C.transpose(1, 2)) * .5
            outs.append(M[:, self.ti[0], self.ti[1]])
        return torch.cat(outs, 1)

    def forward(self, x):                                       # x (B,T,N,3)
        f = self.enc(x).max(2)[0]                               # per-frame feat (B,T,D)
        pooled = torch.cat([f.mean(1), f.max(1)[0]], 1)
        if self.mode != 'base':
            pooled = torch.cat([pooled, self.desc(f)], 1)
        return self.cls(pooled)


def aug(x):                                                    # jitter + scale
    return (x + torch.randn_like(x) * 0.02) * (0.9 + 0.2 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))


def run(mode, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(mode).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 120)
    xtr, ytr = Xtr.to(dev), Ytr.to(dev); xte, yte = Xte.to(dev), Yte.to(dev)
    best = 0
    for ep in range(120):
        m.train(); perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr), 16):
            b = perm[i:i + 16]
            opt.zero_grad(); loss = F.cross_entropy(m(aug(xtr[b])), ytr[b]); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            best = max(best, (m(xte).argmax(1) == yte).float().mean().item() * 100)
    return best


for mode in ['base', 'sym_sh', 'skew_sh']:
    accs = [run(mode, s) for s in range(3)]
    print('%-8s cross-subject acc: %s  mean %.1f' %
          (mode, ['%.1f' % a for a in accs], float(np.mean(accs))), flush=True)
print('DONE', flush=True)
