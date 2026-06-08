"""Where does the floor->91.91 basin escape live? Diff the 91.91 checkpoint
against epoch101 of the SAME resume run (one epoch in == ~floor state it resumed
from). Per-parameter and per-group relative weight change shows whether the escape
is concentrated (a few layers -> reproducible perturbation) or diffuse (hard basin).
CPU only.
"""
import torch, collections

A = torch.load('/notebooks/Manta/experiments/work_dir/pgcnet_skew_resume100/best_model.pt', map_location='cpu')
B = torch.load('/notebooks/Manta/experiments/work_dir/pgcnet_skew_resume100/epoch101_model.pt', map_location='cpu')
A = A.get('model_state_dict', A) if isinstance(A, dict) and 'model_state_dict' in A else A
B = B.get('model_state_dict', B) if isinstance(B, dict) and 'model_state_dict' in B else B

keys = [k for k in A if k in B and A[k].shape == B[k].shape and A[k].is_floating_point()]
rows = []
for k in keys:
    wa, wb = A[k].float(), B[k].float()
    d = (wa - wb).norm().item()
    n = wb.norm().item()
    rows.append((k, d, d / (n + 1e-9), wa.numel()))

print('total params compared:', len(rows))
gtot = collections.defaultdict(lambda: [0.0, 0.0, 0])   # group -> [sum d^2, sum n^2, count]
for k, d, rel, n in rows:
    g = k.split('.')[0]
    gtot[g][0] += d * d; gtot[g][1] += rel; gtot[g][2] += 1
print('\n== per-group: group  |Δ|(L2)  mean-rel-change  nparams ==')
for g in sorted(gtot, key=lambda x: -gtot[x][0]):
    s = gtot[g]
    print('  %-16s %9.3f   %7.4f   %d' % (g, s[0] ** 0.5, s[1] / s[2], s[2]))

print('\n== top-15 params by RELATIVE change ==')
for k, d, rel, n in sorted(rows, key=lambda r: -r[2])[:15]:
    print('  %6.3f rel   |Δ|=%7.3f   %s (%d)' % (rel, d, k, n))

import math
allrel = [rel for _, _, rel, _ in rows]
print('\noverall mean rel change %.4f   median %.4f' %
      (sum(allrel) / len(allrel), sorted(allrel)[len(allrel) // 2]))
print('DONE')
