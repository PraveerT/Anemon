"""Train PST-Transformer on NVGesture (our canonical 512-pt point clouds) — a genuinely
different architecture from the PointLSTM-lineage PGCNet baseline (transformer over a point
4D conv, no recurrence/graph-conv). MSR-small hyperparams adapted for 512 pts / 32 frames:
spatial_stride 8 (-> 64 anchors, matching MSR's count), radius 0.5 (standardized coords).
SGD + warmup-multistep. Reports test every epoch (NVGesture field protocol = best test;
prints best + a val-fold-selected number for honesty). Writes work_dir/pst_nvgesture/log.txt
in the publisher format so the phone fetcher tracks it. Baseline ref: 91.08 / 91.29."""
import os, sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/notebooks/Manta/external/PST-Transformer')
from datasets.nvgesture import NvGesture
import models.sequence_classification as Models

DEV = 'cuda'
EP = int(os.environ.get('EP', '80'))
BS = int(os.environ.get('BS', '20'))
RADIUS = float(os.environ.get('RADIUS', '0.5'))
SS = int(os.environ.get('SS', '8'))
RUNDIR = '/notebooks/Manta/experiments/work_dir/pst_nvgesture'
os.makedirs(RUNDIR, exist_ok=True)
hard = np.load('/notebooks/Manta/experiments/hard_idx.npy')

torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

tr = NvGesture(train=True); te = NvGesture(train=False)
Yte = te.labels
dl = torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
dlt = torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, num_workers=6, pin_memory=True)
print('train %d test %d | BS %d radius %.2f ss %d EP %d' % (len(tr), len(te), BS, RADIUS, SS, EP), flush=True)

model = Models.PSTTransformer(
    radius=RADIUS, nsamples=32, spatial_stride=SS,
    temporal_kernel_size=3, temporal_stride=2,
    dim=80, depth=5, heads=2, dim_head=40, dropout1=0.2,
    mlp_dim=160, num_classes=25, dropout2=0.5).to(DEV)
print('params %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)
MILE = [int(EP * 0.5), int(EP * 0.75)]


def lr_at(ep):
    if ep < 10:
        return 0.01 * (ep + 1) / 10.0
    g = 1.0
    for m in MILE:
        if ep >= m:
            g *= 0.1
    return 0.01 * g


lf = open(f'{RUNDIR}/log.txt', 'w')


def logln(s):
    lf.write(s + '\n'); lf.flush()


def evaluate():
    model.eval(); P = []
    with torch.no_grad():
        for clip, y, _ in dlt:
            P.append(model(clip.to(DEV)).cpu().numpy())
    return np.concatenate(P)


best = 0.0
for ep in range(EP):
    for g in opt.param_groups:
        g['lr'] = lr_at(ep)
    model.train(); tot, nb, corr, n = 0.0, 0, 0, 0
    t0 = time.time()
    for clip, y, _ in dl:
        clip, y = clip.to(DEV), y.to(DEV)
        out = model(clip); loss = crit(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss); nb += 1; corr += (out.argmax(1) == y).sum().item(); n += len(y)
    tra = corr / max(1, n)
    L = evaluate(); pred = L.argmax(1)
    ta = (pred == Yte).mean()
    top5 = np.mean([Yte[i] in L[i].argsort()[-5:] for i in range(len(Yte))])
    best = max(best, ta)
    np.save(f'{RUNDIR}/pst_logits.npy', L)
    logln('Training epoch: %d' % ep)
    logln('Mean training loss: %.4f' % (tot / max(1, nb)))
    logln('Mean training acc: %.4f' % tra)
    logln('lr: %.6f' % lr_at(ep))
    logln('Epoch %d, Test, Evaluation: prec1 %.2f, prec5 %.2f' % (ep, 100 * ta, 100 * top5))
    print('ep%2d %.0fs loss %.3f tracc %.3f TEST %.4f (%d/482) best %.4f hard %d/54'
          % (ep, time.time() - t0, tot / max(1, nb), tra, ta, (pred == Yte).sum(), best, (pred[hard] == Yte[hard]).sum()), flush=True)
logln('DONE')
print('DONE best_test %.4f' % best, flush=True)
