import os, sys, time
os.chdir('/notebooks/Manta/experiments'); sys.path.insert(0, '.')
import torch
from models.motion_bdn_q import BDeltaQTemporalEncoder as EncN
from models.motion_bdn_q_vec import BDeltaQTemporalEncoder as EncV

torch.manual_seed(0)
X = torch.randn(8, 128, 32, 172).cuda()


def mk(E):
    return E(128, 128, 256, num_layers=2, num_heads=4, n_q=4, n_v=8,
             buffer_size=4, bidirectional=True).cuda().train()


def bench(m, it=5):
    m.zero_grad(); y = m(X); y.sum().backward(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(it):
        m.zero_grad(); y = m(X); y.sum().backward(); torch.cuda.synchronize()
    return (time.time() - t0) / it


out = []
tn = bench(mk(EncN)); out.append(f"naive            {tn:.3f}s")
ev = mk(EncV); tv = bench(ev); out.append(f"vec              {tv:.3f}s  {tn/tv:.2f}x")

for mode in ['default', 'reduce-overhead']:
    try:
        ev2 = mk(EncV)
        kw = {} if mode == 'default' else {'mode': mode}
        ec = torch.compile(ev2, dynamic=False, **kw)
        t0 = time.time()
        ec(X).sum().backward(); torch.cuda.synchronize()
        warm = time.time() - t0
        tc = bench(ec)
        out.append(f"vec+compile({mode}) {tc:.3f}s  {tn/tc:.2f}x  (warmup {warm:.0f}s)")
    except Exception as e:
        out.append(f"vec+compile({mode}) FAILED: {repr(e)[:120]}")

open('work_dir/bench_bdnq.txt', 'w').write('\n'.join(out) + '\nDONE\n')
print('\n'.join(out))
