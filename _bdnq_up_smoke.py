"""Smoke-test BDN-Q upstream + RD with real NVGesture data."""
import sys, os, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_bdnq_upstream import MotionBDNQUpstreamRD
from models.motion_bdn_q import BDeltaQBlock

# Count BDN-Q calls
_records = []
_orig = BDeltaQBlock.forward
def _patched(self, x):
    _records.append((x.shape[0], x.shape[1], self.W, max(0, x.shape[1] - self.W)))
    return _orig(self, x)
BDeltaQBlock.forward = _patched

ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
batch = next(iter(loader))
x = batch[0].cuda().float()
print(f'input shape: {tuple(x.shape)}')

model_args = dict(
    pts_size=256, num_classes=25, knn=[32, 24, 48, 24], topk=8,
    multi_scale_num_scales=5,
)
model = MotionBDNQUpstreamRD(**model_args).cuda().eval()
print(f'Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

with torch.no_grad():
    out = model(x)
logits = out[0] if isinstance(out, tuple) else out
print(f'Output: {tuple(logits.shape)}')

print()
print(f'BDeltaQBlock calls (upstream BDN-Q): {len(_records)}')
if _records:
    Ts = sorted(set(r[1] for r in _records))
    ejs = [r[3] for r in _records]
    print(f'  seq_len={Ts}, ejections/call mean={sum(ejs) // len(ejs)}')
