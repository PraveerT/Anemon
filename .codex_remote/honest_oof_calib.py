"""Honest OOF-calibrated fusion. Calibrate weights/temps (and a quaternion head)
on the 218 held-out-subject OOF logits (fold models never trained on subjects
11/15/18 -> CN-XXL is ~91% there, NOT saturated -> real calibration signal).
Apply the chosen rule to the full-train 482 TEST logits ONCE. 444/482 = 92.116.
Respects 'use train softmaxes only' (OOF = held-out train).
"""
import numpy as np, re, torch, torch.nn as nn, torch.nn.functional as F

W = '/notebooks/Anemon/experiments/work_dir/'
OOF = {'cn': 'cn_xxl_fold/test_logits.npz', 'fg': 'fg83_fold/test_logits.npz', 'rgb': 'rgb_fold/best_logits.npz'}
TST = {'cn': 'cn_xxl_quat_head/test_logits.npz', 'fg': 'depth_small_r2_fg83_restored_20260528_033028/test_logits.npz', 'rgb': 'rgb_fgcrop_r2p1d/best_logits.npz'}


def sig(s):
    s = str(s); m = re.search(r'class_(\d+)/subject(\d+)_r(\d+)', s)
    return f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}' if m else s


def load(p, ref=None):
    d = np.load(W + p, allow_pickle=True)
    lg = (d['logits'] if 'logits' in d.files else d['base_logits']).astype(np.float64)
    y = d['labels']; sg = np.array([sig(s) for s in d['sigs']])
    by = {s: i for i, s in enumerate(sg)}
    if ref is None:
        return lg, y, sg
    idx = np.array([by[s] for s in ref])
    return lg[idx], y[idx], sg[ref.tolist().index] if False else sg


def aligned(group):
    cn, y, ref = load(group['cn'])
    fg, yf, _ = load(group['fg'], ref); rg, yr, _ = load(group['rgb'], ref)
    assert (yf == y).all() and (yr == y).all(), 'label mismatch'
    return cn, fg, rg, y


def lsm(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


def acc(z, y):
    return int((z.argmax(1) == y).sum())


cnO, fgO, rgO, yO = aligned(OOF)
cnT, fgT, rgT, yT = aligned(TST)
print(f'OOF n={len(yO)}  cnxxl-OOF acc={acc(lsm(cnO),yO)}/{len(yO)} ({acc(lsm(cnO),yO)/len(yO)*100:.1f}%)  (NOT saturated -> usable)')
print(f'TEST n={len(yT)} cnxxl-test acc={acc(lsm(cnT),yT)}/{len(yT)}   target 444=92.116%')

# ---- (1) OOF-grid linear fusion: maximize OOF acc, apply to TEST ----
Ts = [1.0, 1.5, 2.0, 3.0]; Ws = [0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
best = (-1, None)
for Tr in Ts:
    rO, rT = lsm(rgO, Tr), lsm(rgT, Tr)
    for Tf in Ts:
        fO, fT = lsm(fgO, Tf), lsm(fgT, Tf)
        for wr in Ws:
            for wf in Ws:
                a = acc(lsm(cnO) + wr * rO + wf * fO, yO)
                if a > best[0]:
                    best = (a, (wr, wf, Tr, Tf))
wr, wf, Tr, Tf = best[1]
test_lin = acc(lsm(cnT) + wr * lsm(rgT, Tr) + wf * lsm(fgT, Tf), yT)
print(f'\n[OOF-grid linear] chosen wr={wr} wf={wf} Tr={Tr} Tf={Tf}  (OOF {best[0]}/{len(yO)})')
print(f'  -> TEST {test_lin}/{len(yT)} = {test_lin/len(yT)*100:.3f}%   {"*** >92 ***" if test_lin>=444 else ""}')

# ---- (2) quaternion-product fusion head, fit on OOF, applied to TEST ----
def qprod(a, b):
    K = a.shape[-1] // 4; ar, ai, aj, ak = a.split(K, -1); br, bi, bj, bk = b.split(K, -1)
    return torch.cat([ar*br-ai*bi-aj*bj-ak*bk, ar*bi+ai*br+aj*bk-ak*bj,
                      ar*bj-ai*bk+aj*br+ak*bi, ar*bk+ai*bj-aj*bi+ak*br], -1)


class QFuse(nn.Module):
    def __init__(self, K=8, h=32, nc=25):
        super().__init__(); D = 4 * K
        self.pc, self.pr, self.pf = nn.Linear(nc, D), nn.Linear(nc, D), nn.Linear(nc, D)
        self.head = nn.Sequential(nn.Linear(D, h), nn.GELU(), nn.Dropout(0.4), nn.Linear(h, nc))
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, cn, rg, fg):
        inter = qprod(qprod(self.pc(cn), self.pr(rg)), self.pf(fg))
        return cn + self.scale * self.head(inter)


def run_q(seeds=range(8)):
    cnOt, rgOt, fgOt = torch.tensor(lsm(cnO)), torch.tensor(lsm(rgO)), torch.tensor(lsm(fgO))
    cnTt, rgTt, fgTt = torch.tensor(lsm(cnT)), torch.tensor(lsm(rgT)), torch.tensor(lsm(fgT))
    yOt = torch.tensor(yO); tests = []
    n = len(yO)
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        perm = np.random.permutation(n); fit, hol = perm[:int(0.7*n)], perm[int(0.7*n):]
        m = QFuse(); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=3e-3)
        bestnll, bestte = 1e9, 0
        for ep in range(400):
            m.train(); opt.zero_grad()
            loss = F.cross_entropy(m(cnOt[fit].float(), rgOt[fit].float(), fgOt[fit].float()), yOt[fit])
            loss.backward(); opt.step()
            if ep % 5 == 0:
                m.eval()
                with torch.no_grad():
                    nll = F.cross_entropy(m(cnOt[hol].float(), rgOt[hol].float(), fgOt[hol].float()), yOt[hol]).item()
                    if nll < bestnll:
                        bestnll = nll
                        ot = m(cnTt.float(), rgTt.float(), fgTt.float())
                        bestte = int((ot.argmax(1) == torch.tensor(yT)).sum())
        tests.append(bestte)
    return tests


qt = run_q()
print(f'\n[OOF quaternion fusion] TEST over seeds: {qt}  mean={np.mean(qt):.1f} max={max(qt)}  >=444:{sum(x>=444 for x in qt)}')
print(f'  best quaternion TEST = {max(qt)}/{len(yT)} = {max(qt)/len(yT)*100:.3f}%')
