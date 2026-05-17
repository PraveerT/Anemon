"""Smoke-test BDN-Q N-axis: forward + backward, count ejections."""
import sys, torch
sys.path.insert(0, '/notebooks/PMamba')
sys.path.insert(0, '/notebooks/PMamba/experiments')

from models.motion_bdn_q import MotionBDeltaQ, BDeltaQBlock

_records = []
_orig = BDeltaQBlock.forward
def _patched(self, x):
    B, T_seq, _ = x.shape
    _records.append((T_seq, self.W, max(0, T_seq - self.W)))
    return _orig(self, x)
BDeltaQBlock.forward = _patched

model_args = dict(
    pts_size=96, num_classes=25, knn=[32, 24, 48, 24], topk=8,
    multi_scale_num_scales=5,
    bdnq_hidden_dim=128, bdnq_num_layers=2, bdnq_num_heads=4,
    bdnq_n_q=4, bdnq_n_v=8, bdnq_buffer_size=1, bdnq_dropout=0.3,
    bdnq_bidirectional=True, bdnq_scan_axis='N',
)
model = MotionBDeltaQ(**model_args).cuda().eval()
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, scan_axis=N')

x = torch.randn(2, 3, 32, 256, device='cuda')
print(f'Input: {tuple(x.shape)}')
out = model(x)
loss = (out[0] if isinstance(out, tuple) else out).float().sum()
loss.backward()
print('Forward + backward OK.')

print()
print(f'BDeltaQBlock calls: {len(_records)}')
if _records:
    seqs = [r[0] for r in _records]
    Ws = [r[1] for r in _records]
    ejs = [r[2] for r in _records]
    print(f'  seq_len={sorted(set(seqs))} W={sorted(set(Ws))} ejections/call mean={sum(ejs)/len(ejs):.1f}')
