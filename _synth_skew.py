"""Mechanism test for Skew-TCC on synthetic data built for it.
Classes = rotation {z,x} x {CW,CCW}. The per-frame covariance MEAN/STD over time
(symmetric pooling) reveals the AXIS but is blind to DIRECTION (the time-multiset of
frames is identical for CW/CCW). The antisymmetric lagged cross-covariance (Skew-TCC)
is order-dependent -> captures direction. Param-matched nets:
  baseline   : symmetric pool (mean+std of per-frame cov) -> MLP        (axis only)
  +sym       : baseline ++ SYMMETRIC lagged cross-cov descriptor (control)
  +skew      : baseline ++ ANTISYMMETRIC lagged cross-cov (Skew-TCC)
Expect: baseline ~= +sym ~= 50% (axis), +skew ~= 100% (axis+direction).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0); np.random.seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
P, T, NC = 64, 16, 4
AXES = [(0, 1), (1, 2)]          # rotation planes: xy (about z), yz (about x)
STEP = 0.35                      # rad/frame


def rot(plane, ang):
    R = torch.eye(3)
    i, j = plane
    c, s = np.cos(ang), np.sin(ang)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def gen(n_per):
    # direction class = EXACT time-reversal of forward -> symmetric part provably
    # identical (sym_sh MUST be blind to direction); only antisym flips.
    X, Y = [], []
    for cls in range(NC):
        plane = AXES[cls // 2]
        rev = (cls % 2 == 1)
        for _ in range(n_per):
            base = (torch.randn(P, 3) * torch.tensor([3.0, 1.0, 0.5]))
            base = base @ rot((0, 2), float(np.random.rand() * 6.28))
            seq = torch.stack([base @ rot(plane, STEP * t).T for t in range(T)])
            if rev:
                seq = torch.flip(seq, [0])               # exact reversal
            X.append(seq); Y.append(cls)
    return torch.stack(X), torch.tensor(Y)


def raw_cov(x):                  # x (B,T,P,3) -> cov (B,T,9) RAW (no time-standardize)
    xc = x - x.mean(2, keepdim=True)
    cov = torch.einsum('btpi,btpj->btij', xc, xc) / (P - 1)
    return cov.reshape(x.shape[0], T, 9)


def per_frame_cov(x):            # standardized over time, for the descriptor
    z = raw_cov(x)
    return (z - z.mean(1, keepdim=True)) / (z.std(1, keepdim=True) + 1e-5)


class Net(nn.Module):
    # mode in {base, sym, skew, sym_sh, skew_sh}; *_sh => SHARED projector Wu=Wv
    def __init__(self, mode='base', r=8, lags=(1, 2)):
        super().__init__()
        self.mode = mode; self.r = r; self.lags = lags
        self.shared = mode.endswith('_sh')
        self.is_skew = mode.startswith('skew')
        din = 18                                   # mean+std of z (9+9)
        if mode != 'base':
            self.Wu = nn.Linear(9, r, bias=False)
            self.Wv = self.Wu if self.shared else nn.Linear(9, r, bias=False)
            # skew -> strictly-lower tri (antisym); sym -> lower tri incl diag
            self.ti = torch.tril_indices(r, r, -1 if self.is_skew else 0)
            din += self.ti.shape[1] * len(lags)
        self.mlp = nn.Sequential(nn.Linear(din, 64), nn.GELU(), nn.Linear(64, NC))

    def desc(self, z):
        U, V = self.Wu(z), self.Wv(z)
        outs = []
        for d in self.lags:
            u, v = U[:, :T - d], V[:, d:]
            C = torch.einsum('bti,btj->bij', u, v) / (T - d)
            M = (C - C.transpose(1, 2)) * .5 if self.is_skew else (C + C.transpose(1, 2)) * .5
            outs.append(M[:, self.ti[0], self.ti[1]])
        return torch.cat(outs, 1)

    def forward(self, x):
        raw = raw_cov(x)                            # fair baseline: RAW cov stats
        f = torch.cat([raw.mean(1), raw.std(1)], 1)   # symmetric pool (gets axis)
        if self.mode != 'base':
            f = torch.cat([f, self.desc(per_frame_cov(x))], 1)
        return self.mlp(f)


Xtr, Ytr = gen(250); Xte, Yte = gen(80)
Xtr, Ytr, Xte, Yte = [t.to(dev) for t in (Xtr, Ytr, Xte, Yte)]
print('train', Xtr.shape, 'test', Xte.shape, flush=True)


def run(mode):
    torch.manual_seed(0)
    m = Net(mode).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
    best = 0
    for ep in range(200):
        m.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), 64):
            b = perm[i:i + 64]
            opt.zero_grad(); loss = F.cross_entropy(m(Xtr[b]), Ytr[b]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc = (m(Xte).argmax(1) == Yte).float().mean().item() * 100
        best = max(best, acc)
    np_ = sum(p.numel() for p in m.parameters())
    return best, np_


for mode in ['base', 'sym_sh', 'skew_sh', 'sym', 'skew']:
    acc, npar = run(mode)
    print('%-7s best test acc %5.1f%%   params %d' % (mode, acc, npar), flush=True)
print('(shared-projector Wu=Wv: sym_sh = clean time-symmetric control; '
      'skew_sh isolates antisymmetry)', flush=True)
print('DONE', flush=True)
