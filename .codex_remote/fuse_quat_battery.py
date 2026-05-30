"""Quaternion non-commutativity fusion battery (logit-level, honest protocol).

Fuse {CN-XXL, RGB-fgcrop, FG83} log-softmax vectors. All heads are RESIDUAL from
CN-XXL (logits = cnxxl + scale*correction, scale init ~0) so they start at the
CN-XXL baseline and learn corrections. Heads:
  linear  : correction = W[concat(rgb,fg)]                 (no quaternion)
  mlp     : correction = MLP(concat(rgb,fg))               (nonlinear, no quat)
  quat    : Hamilton-product interaction qprod(qprod(cn,rgb),fg)  (non-commutative)
  quatcomm: quat + commutator gate (qprod(cn,a)-qprod(a,cn)) -> when to trust aux

Protocol: fit on 70% of TRAIN, select epoch on 30% holdout (by NLL), eval TEST once.
Multi-seed. Honest caveat: holdout is train-derived (CN-XXL ~100% on it) so selection
is weak; test is never tuned on. 444/482 = 92.116, cnxxl alone = 440.
"""
import numpy as np, re, torch, torch.nn as nn, torch.nn.functional as F

W = '/notebooks/Anemon/experiments/work_dir/'
P = {'cn': 'cn_xxl_quat_head', 'rgb': 'rgb_fgcrop_r2p1d', 'fg': 'depth_small_r2_fg83_restored_20260528_033028'}


def sig(s):
    s = str(s); m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', s)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}' if m else s


def load(p, split, ref=None):
    d = np.load(f'{W}{p}/{split}_logits.npz', allow_pickle=True)
    lg = d['logits'].astype(np.float32); y = d['labels']; sg = np.array([sig(s) for s in d['sigs']])
    if ref is not None:
        by = {s: i for i, s in enumerate(sg)}; idx = np.array([by[s] for s in ref]); lg, y = lg[idx], y[idx]
    return lg, y, sg


def lsm(z):
    z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


# train + test, aligned within each split
cn_tr, y_tr, ref_tr = load(P['cn'], 'train')
rg_tr, _, _ = load(P['rgb'], 'train', ref_tr); fg_tr, _, _ = load(P['fg'], 'train', ref_tr)
cn_te, y_te, ref_te = load(P['cn'], 'test')
rg_te, _, _ = load(P['rgb'], 'test', ref_te); fg_te, _, _ = load(P['fg'], 'test', ref_te)
to = lambda a: torch.tensor(lsm(a))
CNtr, RGtr, FGtr = to(cn_tr), to(rg_tr), to(fg_tr)
CNte, RGte, FGte = to(cn_te), to(rg_te), to(fg_te)
Ytr = torch.tensor(y_tr); Yte = torch.tensor(y_te)
print(f'cnxxl solo test {(CNte.argmax(1)==Yte).sum().item()}/{len(Yte)}  (444=92.116)')


def qprod(a, b):  # a,b: (...,4K) laid [r|i|j|k]
    K = a.shape[-1] // 4
    ar, ai, aj, ak = a.split(K, -1); br, bi, bj, bk = b.split(K, -1)
    return torch.cat([ar*br-ai*bi-aj*bj-ak*bk, ar*bi+ai*br+aj*bk-ak*bj,
                      ar*bj-ai*bk+aj*br+ak*bi, ar*bk+ai*bj-aj*bi+ak*br], -1)


class Fuse(nn.Module):
    def __init__(self, kind, K=16, hid=64, ncls=25):
        super().__init__()
        self.kind = kind; D = 4 * K
        self.pc = nn.Linear(ncls, D); self.pr = nn.Linear(ncls, D); self.pf = nn.Linear(ncls, D)
        if kind == 'linear':
            self.head = nn.Linear(2 * ncls, ncls)
        elif kind == 'mlp':
            self.head = nn.Sequential(nn.Linear(2 * ncls, hid), nn.GELU(), nn.Dropout(0.3), nn.Linear(hid, ncls))
        elif kind == 'quat':
            self.head = nn.Sequential(nn.Linear(D, hid), nn.GELU(), nn.Dropout(0.3), nn.Linear(hid, ncls))
        elif kind == 'quatcomm':
            self.head = nn.Sequential(nn.Linear(2 * D, hid), nn.GELU(), nn.Dropout(0.3), nn.Linear(hid, ncls))
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, cn, rg, fg):
        if self.kind in ('linear', 'mlp'):
            corr = self.head(torch.cat([rg, fg], -1))
        else:
            zc, zr, zf = self.pc(cn), self.pr(rg), self.pf(fg)
            inter = qprod(qprod(zc, zr), zf)              # ordered, non-commutative
            if self.kind == 'quatcomm':
                comm = qprod(zc, zr) - qprod(zr, zc)      # commutator (disagreement gate)
                corr = self.head(torch.cat([inter, comm], -1))
            else:
                corr = self.head(inter)
        return cn + self.scale * corr


def run(kind, seeds=range(5)):
    accs = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        n = len(Ytr); perm = np.random.permutation(n); nf = int(0.7 * n)
        fit, hol = perm[:nf], perm[nf:]
        m = Fuse(kind); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-3)
        best_nll, best_te = 1e9, 0
        for ep in range(300):
            m.train(); opt.zero_grad()
            o = m(CNtr[fit], RGtr[fit], FGtr[fit]); loss = F.cross_entropy(o, Ytr[fit])
            loss.backward(); opt.step()
            if ep % 5 == 0:
                m.eval()
                with torch.no_grad():
                    oh = m(CNtr[hol], RGtr[hol], FGtr[hol]); nll = F.cross_entropy(oh, Ytr[hol]).item()
                    if nll < best_nll:
                        best_nll = nll
                        ot = m(CNte, RGte, FGte); best_te = (ot.argmax(1) == Yte).sum().item()
        accs.append(best_te)
    return accs


for kind in ['linear', 'mlp', 'quat', 'quatcomm']:
    a = run(kind)
    print(f'{kind:9s} test correct/482 over seeds: {a}  mean={np.mean(a):.1f}  max={max(a)}  >=444:{sum(x>=444 for x in a)}/5')
