"""Strong mesh-GNN (v5) — balanced. v3's fast-learning core (4-layer multi-scale EdgeConv,
mean aggregation, 1-layer BiGRU with LayerNorm'd input + grad-clip) + global normalization +
LIGHT augmentation done per-BATCH on GPU (fast: chirality-safe in-plane rotation, scale,
jitter) + light DropEdge + moderate dropout. Swappable edge set: mesh triangulation vs kNN
(only connectivity differs). Honest: epoch picked on a stratified val fold of TRAIN.
work_dir/mesh_gnn_<et>/log.txt is fetcher-formatted. Baseline ref: 91.08 / 91.29; PST-T 73.4."""
import os, gc, time, numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree
os.chdir('/notebooks/Manta/experiments')
from mesh_ds import mesh_faces

DEV = 'cuda'
K = 6
WIDTH = 128
LAYERS = 4
GRU_H = 128
LR = 1e-3
EPOCHS = int(os.environ.get('EP', '80'))
BS = 8
DROPEDGE = 0.05
EDGE_TYPES = os.environ.get('ET', 'mesh,knn').split(',')
hard = np.load('hard_idx.npy')


def load_split(split):
    return torch.load(f'mesh_cache_{split}.pt')


trraw = load_split('train')
allv = np.concatenate([c['x'].astype(np.float32).reshape(-1, 3) for c in trraw if len(c['x'])], 0)
GMEAN = allv.mean(0).astype(np.float32); GSCALE = float(allv.std())
print('global norm: mean %s scale %.2f' % (np.round(GMEAN, 1).tolist(), GSCALE), flush=True)


def build(clips):
    out = []
    for c in clips:
        xi = c['x'].astype(np.int64); fptr = c['fptr']; y = int(c['y'])
        if len(xi) == 0:
            out.append(None); continue
        xf = (xi.astype(np.float32) - GMEAN) / (GSCALE + 1e-6)
        fid = np.zeros(len(xi), np.int64); me, ke = [], []
        for t in range(32):
            a, b = fptr[t], fptr[t + 1]; n = b - a; fid[a:b] = t
            sl = np.arange(a, b); self_e = np.stack([sl, sl], 0)
            if n >= 4:
                f = mesh_faces(xi[a:b])
                if len(f):
                    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
                    e = np.unique(np.sort(e, 1), axis=0) + a
                    me.append(np.concatenate([e.T, e[:, ::-1].T, self_e], 1))
                else:
                    me.append(self_e)
            elif n >= 1:
                me.append(self_e)
            if n >= 2:
                kk = min(K, n - 1)
                _, idx = cKDTree(xf[a:b]).query(xf[a:b], k=kk + 1)
                src = np.repeat(np.arange(n), kk); dst = idx[:, 1:].reshape(-1)
                e = np.stack([src + a, dst + a], 0)
                ke.append(np.concatenate([e, e[::-1], self_e], 1))
            elif n >= 1:
                ke.append(self_e)
        out.append(dict(
            xf=torch.from_numpy(xf), fid=torch.from_numpy(fid),
            mesh_ei=torch.from_numpy(np.concatenate(me, 1)).long() if me else torch.zeros((2, 0), dtype=torch.long),
            knn_ei=torch.from_numpy(np.concatenate(ke, 1)).long() if ke else torch.zeros((2, 0), dtype=torch.long),
            y=y))
    return out


class EdgeConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * cin, cout), nn.BatchNorm1d(cout), nn.GELU())

    def forward(self, h, ei):
        i, j = ei[0], ei[1]
        msg = self.mlp(torch.cat([h[i], h[j] - h[i]], 1))
        cnt = torch.zeros(h.shape[0], device=h.device).index_add_(0, i, torch.ones(i.shape[0], device=h.device))
        out = torch.zeros(h.shape[0], msg.shape[1], device=h.device).index_add_(0, i, msg)
        return out / cnt.clamp(min=1).unsqueeze(1)


def frame_pool(H, fid, NF):
    D = H.shape[1]
    cnt = torch.zeros(NF, device=H.device).index_add_(0, fid, torch.ones(fid.shape[0], device=H.device))
    mean = torch.zeros(NF, D, device=H.device).index_add_(0, fid, H) / cnt.clamp(min=1).unsqueeze(1)
    mx = torch.full((NF, D), -1e9, device=H.device).scatter_reduce(0, fid.unsqueeze(1).expand(-1, D), H, 'amax', include_self=True)
    mx = torch.where(cnt.unsqueeze(1) > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], 1)


class StrongMeshGNN(nn.Module):
    def __init__(self, w=WIDTH, nclass=25, layers=LAYERS):
        super().__init__()
        self.embed = nn.Linear(3, w)
        self.convs = nn.ModuleList([EdgeConv(w, w) for _ in range(layers)])
        self.frame_proj = nn.Sequential(nn.Linear(2 * layers * w, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(256, GRU_H, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * GRU_H), nn.Linear(2 * GRU_H, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, nclass))

    def forward(self, x, ei, fid, NF):
        h = self.embed(x); feats = []
        for c in self.convs:
            h = h + c(h, ei); feats.append(h)
        H = torch.cat(feats, 1)
        f = self.frame_proj(frame_pool(H, fid, NF)).view(-1, 32, 256)
        g, _ = self.gru(f)
        return self.head(g.mean(1))


def collate(batch, et):
    xs, eis, fids, ys, noff, foff = [], [], [], [], 0, 0
    key = 'mesh_ei' if et == 'mesh' else 'knn_ei'
    for c in batch:
        xs.append(c['xf']); eis.append(c[key] + noff); fids.append(c['fid'] + foff)
        ys.append(c['y']); noff += c['xf'].shape[0]; foff += 32
    return (torch.cat(xs).to(DEV), torch.cat(eis, 1).to(DEV), torch.cat(fids).to(DEV),
            len(batch) * 32, torch.tensor(ys).to(DEV))


def augment(X, EI):
    """Light per-batch aug on GPU: in-plane rotation (det +1), scale, jitter, DropEdge."""
    th = (torch.rand((), device=X.device) - 0.5) * 0.34            # ~+-10 deg
    ct, st = torch.cos(th), torch.sin(th)
    Xr = X.clone()
    Xr[:, 0] = ct * X[:, 0] - st * X[:, 1]
    Xr[:, 1] = st * X[:, 0] + ct * X[:, 1]
    Xr = Xr * (0.9 + 0.2 * torch.rand((), device=X.device)) + torch.randn_like(Xr) * 0.01
    if DROPEDGE > 0 and EI.shape[1] > 0:
        EI = EI[:, torch.rand(EI.shape[1], device=EI.device) > DROPEDGE]
    return Xr, EI


def run(et, tr, te, vidx, tidx):
    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    net = StrongMeshGNN().to(DEV); opt = torch.optim.Adam(net.parameters(), LR, weight_decay=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Yte = np.array([te[i]['y'] for i in range(len(te))]); Yv = np.array([tr[i]['y'] for i in vidx])
    rundir = f'work_dir/mesh_gnn_{et}'; os.makedirs(rundir, exist_ok=True)
    lf = open(f'{rundir}/log.txt', 'w')

    def logln(s):
        lf.write(s + '\n'); lf.flush()

    def ev(ds, idx):
        net.eval(); P = []
        with torch.no_grad():
            for s in range(0, len(idx), 24):
                b = [ds[i] for i in idx[s:s + 24]]
                x, ei, fid, NF, y = collate(b, et)
                P.append(net(x, ei, fid, NF).cpu().numpy())
        return np.concatenate(P)

    best = (-1, 0, None)
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(tidx); tot, nb = 0.0, 0
        for s in range(0, len(perm), BS):
            b = [tr[i] for i in perm[s:s + BS]]
            x, ei, fid, NF, y = collate(b, et); x, ei = augment(x, ei); opt.zero_grad()
            loss = nn.functional.cross_entropy(net(x, ei, fid, NF), y, label_smoothing=0.1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1
        sch.step()
        va = (ev(tr, vidx).argmax(1) == Yv).mean()
        lt = ev(te, list(range(len(te)))); ta = (lt.argmax(1) == Yte).mean()
        top5 = np.mean([Yte[i] in lt[i].argsort()[-5:] for i in range(len(Yte))])
        if va > best[0]:
            best = (va, ep, lt)
        logln('Training epoch: %d' % ep); logln('Mean training loss: %.4f' % (tot / max(1, nb)))
        logln('Mean training acc: %.4f' % va); logln('lr: %.6f' % sch.get_last_lr()[0])
        logln('Epoch %d, Test, Evaluation: prec1 %.2f, prec5 %.2f' % (ep, 100 * ta, 100 * top5))
        if ep % 5 == 0 or ep == EPOCHS - 1:
            bt = (best[2].argmax(1) == Yte).mean()
            print('[%s] ep%3d loss %.3f val %.3f test %.4f (%d/482) best %.4f hard %d/54'
                  % (et, ep, tot / max(1, nb), va, ta, (lt.argmax(1) == Yte).sum(), bt, (lt.argmax(1)[hard] == Yte[hard]).sum()), flush=True)
    lf.close()
    va, ep, lt = best; ta = (lt.argmax(1) == Yte).mean()
    print('[%s] BEST-VAL ep%d: val %.3f -> TEST %.4f (%d/482)  hard %d/54'
          % (et, ep, va, ta, (lt.argmax(1) == Yte).sum(), (lt.argmax(1)[hard] == Yte[hard]).sum()), flush=True)
    np.save(f'_mesh_logits_{et}.npy', lt)
    del net, opt; gc.collect(); torch.cuda.empty_cache()
    return ta


t0 = time.time()
print('building edges (global-normalized)...', flush=True)
TR = build(trraw); TE = build(load_split('test')); TR = [c for c in TR if c is not None]
print('train %d test %d | built in %.0fs' % (len(TR), len(TE), time.time() - t0), flush=True)

ys = np.array([c['y'] for c in TR]); rng = np.random.default_rng(0); vidx = []
for cls in range(25):
    ids = np.where(ys == cls)[0]; rng.shuffle(ids); vidx += ids[:max(1, int(0.15 * len(ids)))].tolist()
vidx = sorted(vidx); tidx = sorted(set(range(len(TR))) - set(vidx))
print('val %d train %d' % (len(vidx), len(tidx)), flush=True)

res = {}
for et in EDGE_TYPES:
    print('\n===== EDGE TYPE: %s =====' % et, flush=True)
    res[et] = run(et, TR, TE, vidx, tidx)
print('\n=== SUMMARY (baseline ref: 91.08 / 91.29; PST-Transformer 73.4) ===', flush=True)
for et in EDGE_TYPES:
    print('  %-5s test %.4f' % (et, res[et]), flush=True)
print('DONE', flush=True)
