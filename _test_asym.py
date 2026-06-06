"""Re-verify the antisymmetry claim on the LEAN deterministic model.
Does accuracy collapse under spatial mirror / time-reversal of test inputs?
Input (B,T,N,4) = u,v,depth,frame."""
import os, sys
os.chdir('/notebooks/Manta/experiments'); sys.path.insert(0, '.')
os.environ['DENSE'] = '0'
import torch
import torch.utils.data as D
from models.pgcnet_pruned import PGCNetPruned
from nvidia_dataloader import NvidiaLoader

CKPT = 'work_dir/pgcnet_lean1m_det/best_model.pt'
ma = dict(framesize=32, knn=[32, 24, 48, 24], lxl_bidirectional=True, lxl_dropout=0.3,
          lxl_hidden_dim=64, lxl_mlp_dim=128, lxl_num_layers=1, lxl_residual_scale=0.7,
          multi_scale_num_scales=5, num_classes=25, pts_size=172, th_hidden=128, th_layers=1)
m = PGCNetPruned(**ma).cuda().eval()
ck = torch.load(CKPT, map_location='cpu')
sd = ck.get('model_state_dict', ck)
r = m.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)

ds = NvidiaLoader(framerate=32, phase='test')
dl = D.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)


def ident(x):
    return x


def mirror_u(x):
    x = x.clone(); u = x[..., 0]
    x[..., 0] = 2 * u.mean(dim=(1, 2), keepdim=True) - u
    return x


def mirror_v(x):
    x = x.clone(); v = x[..., 1]
    x[..., 1] = 2 * v.mean(dim=(1, 2), keepdim=True) - v
    return x


def treverse(x):
    x = x.clone()
    x[..., :3] = torch.flip(x[..., :3], dims=[1])
    return x


def run(tf):
    cor = tot = 0; preds = []; labs = []
    with torch.no_grad():
        for x, y, _ in dl:
            p = m(tf(x.cuda())).argmax(1).cpu()
            preds.append(p); labs.append(y)
            cor += int((p == y).sum()); tot += y.numel()
    return cor, tot, torch.cat(preds), torch.cat(labs)


out = [f"ckpt {CKPT} | missing {len(r.missing_keys)} unexpected {len(r.unexpected_keys)}"]
bc, bt, bp, lab = run(ident)
base_correct = (bp == lab)
out.append(f"normal     {bc}/{bt} = {100*bc/bt:.2f}%")
for name, tf in [('mirror_u', mirror_u), ('mirror_v', mirror_v), ('treverse', treverse)]:
    c, t, p, _ = run(tf)
    changed = int((p != bp).sum())
    # of the samples the model got RIGHT normally, how many does it still get right?
    kept = int((p[base_correct] == lab[base_correct]).sum())
    out.append(f"{name:10s} {c}/{t} = {100*c/t:.2f}%   preds_changed {changed}/{t}   of_{int(base_correct.sum())}_correct_kept {kept}")

open('work_dir/asym_test.txt', 'w').write('\n'.join(out) + '\nDONE\n')
print('\n'.join(out))
