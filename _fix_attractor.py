"""Attractor diagnosis + post-hoc logit-adjustment fix (train-calibrated, honest).
Dataset is balanced, so attractors = the MODEL over-predicting some classes.
Estimate per-class over-prediction on TRAIN, subtract from TEST logits (tau=1,
no test tuning). Report clean vs adjusted test acc + the attractor structure."""
import os, sys
os.chdir('/notebooks/Manta/experiments'); sys.path.insert(0, '.')
os.environ['DENSE'] = '0'
import torch
import torch.nn.functional as F
import torch.utils.data as D
from models.pgcnet_pruned import PGCNetPruned
from nvidia_dataloader import NvidiaLoader

ma = dict(framesize=32, knn=[32, 24, 48, 24], lxl_bidirectional=True, lxl_dropout=0.3,
          lxl_hidden_dim=64, lxl_mlp_dim=128, lxl_num_layers=1, lxl_residual_scale=0.7,
          multi_scale_num_scales=5, num_classes=25, pts_size=172, th_hidden=128, th_layers=1)
m = PGCNetPruned(**ma).cuda().eval()
ck = torch.load('work_dir/pgcnet_lean1m_det/best_model.pt', map_location='cpu')
m.load_state_dict({k.replace('module.', ''): v for k, v in ck.get('model_state_dict', ck).items()}, strict=False)
C = 25


def logits_of(phase):
    ds = NvidiaLoader(framerate=32, phase=phase)
    dl = D.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    L, Y = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            L.append(m(x.cuda()).cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y)


trL, trY = logits_of('train')
teL, teY = logits_of('test')

acc = lambda L, Y: 100.0 * int((L.argmax(1) == Y).sum()) / len(Y)
out = [f"clean: train {acc(trL,trY):.2f}%  test {acc(teL,teY):.2f}%"]

# attractor structure on TEST (balanced 25 classes, ~19-20 each)
te_pred = torch.bincount(teL.argmax(1), minlength=C)
te_true = torch.bincount(teY, minlength=C)
over = (te_pred - te_true)
worst_attr = over.argsort(descending=True)[:4].tolist()
worst_vic = over.argsort()[:4].tolist()
out.append("attractors (over-predicted on test): " + ", ".join(f"c{c}(+{int(over[c])})" for c in worst_attr))
out.append("victims    (under-predicted on test): " + ", ".join(f"c{c}({int(over[c])})" for c in worst_vic))

# offset estimated on TRAIN (avg predicted prob per class), applied to TEST at tau=1
tr_avgprob = F.softmax(trL, dim=1).mean(0)              # (C,)
offset = torch.log(tr_avgprob + 1e-8)
te_adj = teL - offset
out.append(f"post-hoc logit-adj (train-calib, tau=1): test {acc(te_adj,teY):.2f}%  (delta {acc(te_adj,teY)-acc(teL,teY):+.2f})")

# argmax-frequency variant
tr_freq = torch.bincount(trL.argmax(1), minlength=C).float() / len(trY)
off2 = torch.log(tr_freq + 1e-8)
out.append(f"post-hoc logit-adj (argmax-freq, tau=1):  test {acc(teL-off2,teY):.2f}%  (delta {acc(teL-off2,teY)-acc(teL,teY):+.2f})")

open('work_dir/fix_attr.txt', 'w').write('\n'.join(out) + '\nDONE\n')
print('\n'.join(out))
