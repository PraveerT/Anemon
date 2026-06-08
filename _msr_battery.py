"""Find the real lever on MSRAction3D (cross-subject, 3-seed). Baseline is a
vanilla per-frame PointNet + symmetric temporal pool. Test the two things it
lacks: LOCAL spatial structure (EdgeConv kNN) and TEMPORAL order (temporal conv).
  pn_pool   : baseline (vanilla pointnet + mean/max pool)         ~84
  pn_tconv  : + temporal order
  edge_pool : + local spatial structure
  edge_tconv: + both
Whichever moves 84 up reproducibly is where a contribution can live.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = 'cuda'
d = np.load('/notebooks/wt9191/experiments/msr_pc.npz')
X, A, S = d['X'], d['action'], d['subject']
T, N = X.shape[1], X.shape[2]
tr = np.isin(S, [1, 3, 5, 7, 9])
Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(A[tr])
Xte, Yte = torch.tensor(X[~tr]), torch.tensor(A[~tr])
print('train', Xtr.shape, 'test', Xte.shape, flush=True)
K = 16


def knn_edge(x, mlp):                # x (BT, N, 3) -> (BT, N, D) edgeconv-maxpool
    BT = x.shape[0]
    dist = torch.cdist(x, x)                       # (BT,N,N)
    idx = dist.topk(K, largest=False, dim=-1).indices   # (BT,N,K)
    nb = torch.gather(x.unsqueeze(2).expand(-1, -1, K, -1), 1,
                      idx.unsqueeze(-1).expand(-1, -1, -1, 3))
    feat = torch.cat([x.unsqueeze(2).expand(-1, -1, K, -1), nb - x.unsqueeze(2)], -1)  # (BT,N,K,6)
    return mlp(feat).max(2)[0]                     # (BT,N,D)


def ln_mlp(dims):                        # Linear+LayerNorm+ReLU stack (stable)
    L = []
    for i in range(len(dims) - 1):
        L += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]), nn.ReLU()]
    return nn.Sequential(*L)


class Net(nn.Module):
    def __init__(self, enc='pn', temporal='pool', D=128):
        super().__init__()
        self.enc, self.temporal = enc, temporal
        if enc == 'pn':
            self.point = ln_mlp([3, 64, 128, D])
        else:
            self.edge = ln_mlp([6, 64, 64])
            self.point = ln_mlp([64, 128, D])
        if temporal == 'tconv':
            self.tc = nn.Sequential(nn.Conv1d(D, D, 3, padding=1), nn.BatchNorm1d(D), nn.ReLU(),
                                    nn.Conv1d(D, D, 3, padding=1), nn.BatchNorm1d(D), nn.ReLU())
        self.cls = nn.Sequential(nn.Linear(2 * D, 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 20))

    def forward(self, x):                # x (B,T,N,3)
        B = x.shape[0]
        xf = x.reshape(B * T, N, 3)
        if self.enc == 'pn':
            pf = self.point(xf).max(1)[0]            # (BT,D)
        else:
            pf = self.point(knn_edge(xf, self.edge)).max(1)[0]
        f = pf.reshape(B, T, -1)                      # (B,T,D)
        if self.temporal == 'tconv':
            f = f + self.tc(f.transpose(1, 2)).transpose(1, 2)   # residual
        return self.cls(torch.cat([f.mean(1), f.max(1)[0]], 1))


def aug(x):
    return (x + torch.randn_like(x) * 0.02) * (0.9 + 0.2 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))


def run(enc, temporal, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(enc, temporal).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 120)
    xtr, ytr = Xtr.to(dev), Ytr.to(dev); xte, yte = Xte.to(dev), Yte.to(dev)
    hist = []
    for ep in range(120):
        m.train(); perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr), 16):
            b = perm[i:i + 16]
            opt.zero_grad(); F.cross_entropy(m(aug(xtr[b])), ytr[b]).backward(); opt.step()
        sch.step(); m.eval()
        with torch.no_grad():
            hist.append((m(xte).argmax(1) == yte).float().mean().item() * 100)
    return hist[-1], float(np.mean(hist[-10:]))     # honest: final, last-10 mean (no test-selection)


for enc, temporal, name in [('pn', 'pool', 'pn_pool'), ('pn', 'tconv', 'pn_tconv'),
                            ('edge', 'tconv', 'edge_tconv')]:
    res = [run(enc, temporal, s) for s in range(3)]
    fin = [r[0] for r in res]; l10 = [r[1] for r in res]
    print('%-11s final %s (m%.1f)  last10 %s (m%.1f)'
          % (name, ['%.1f' % a for a in fin], float(np.mean(fin)),
             ['%.1f' % a for a in l10], float(np.mean(l10))), flush=True)
print('DONE', flush=True)
