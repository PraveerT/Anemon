"""Does the model map mirror(class i) -> a consistent twin(i)? If so it learned
reflection as a clean class-swapping involution (handedness as a group action),
NOT a collapse. Builds the mirror-confusion, derives the empirical twin map,
tests involution + re-scores against twins."""
import os, sys
os.chdir('/notebooks/Manta/experiments'); sys.path.insert(0, '.')
os.environ['DENSE'] = '0'
import torch
import torch.utils.data as D
from models.pgcnet_pruned import PGCNetPruned
from nvidia_dataloader import NvidiaLoader

ma = dict(framesize=32, knn=[32, 24, 48, 24], lxl_bidirectional=True, lxl_dropout=0.3,
          lxl_hidden_dim=64, lxl_mlp_dim=128, lxl_num_layers=1, lxl_residual_scale=0.7,
          multi_scale_num_scales=5, num_classes=25, pts_size=172, th_hidden=128, th_layers=1)
m = PGCNetPruned(**ma).cuda().eval()
ck = torch.load('work_dir/pgcnet_lean1m_det/best_model.pt', map_location='cpu')
m.load_state_dict({k.replace('module.', ''): v for k, v in ck.get('model_state_dict', ck).items()}, strict=False)

ds = NvidiaLoader(framerate=32, phase='test')
dl = D.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

C = 25


def mirror_u(x):
    x = x.clone(); u = x[..., 0]
    x[..., 0] = 2 * u.mean(dim=(1, 2), keepdim=True) - u
    return x


def collect(tf):
    P, Y = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            P.append(m(tf(x.cuda())).argmax(1).cpu()); Y.append(y)
    return torch.cat(P), torch.cat(Y)


pn, Y = collect(lambda x: x)
pm, _ = collect(mirror_u)

# mirror confusion M[i,j] = count(true i -> predict j under mirror)
M = torch.zeros(C, C)
for i, j in zip(Y.tolist(), pm.tolist()):
    M[i, j] += 1
counts = M.sum(1).clamp(min=1)
twin = M.argmax(1)                                  # empirical twin(i)
conc = (M.max(1).values / counts)                   # fraction of mirror(i) -> twin(i)

# involution check: twin(twin(i)) == i ?
invol = sum(int(twin[twin[i]] == i) for i in range(C) if counts[i] > 0)
nonself = sum(int(twin[i] != i) for i in range(C) if counts[i] > 0)
present = [i for i in range(C) if counts[i] > 0]

# re-score: mirror prediction == twin(true label)?
twin_correct = sum(int(pm[k] == twin[Y[k]]) for k in range(len(Y)))
orig_correct = int((pm == Y).sum())

out = []
out.append(f"normal acc {int((pn==Y).sum())}/{len(Y)} = {100*int((pn==Y).sum())/len(Y):.2f}%")
out.append(f"mirror vs ORIGINAL labels {orig_correct}/{len(Y)} = {100*orig_correct/len(Y):.2f}%")
out.append(f"mirror vs TWIN labels     {twin_correct}/{len(Y)} = {100*twin_correct/len(Y):.2f}%")
out.append(f"classes present {len(present)} | twin is involution for {invol}/{len(present)} | non-self-twin {nonself}/{len(present)}")
out.append("per-class twin(i) [conc]:")
for i in present:
    out.append(f"  {i:2d} -> {int(twin[i]):2d}  conc {conc[i]:.2f}  (n={int(counts[i])})")
open('work_dir/twin_test.txt', 'w').write('\n'.join(out) + '\nDONE\n')
print('\n'.join(out))
