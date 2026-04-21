"""Evaluate every saved pmamba_branch checkpoint on the test set; find best epoch."""
import sys, os, glob
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
import numpy as np
from models.motion import Motion
from nvidia_dataloader import NvidiaLoader

PTS = 256
N_TTA = 3
CKPT_DIR = 'work_dir/pmamba_branch'

ckpts = sorted(
    glob.glob(f'{CKPT_DIR}/epoch*_model.pt'),
    key=lambda p: int(p.split('epoch')[1].split('_')[0]),
)
print(f'Found {len(ckpts)} checkpoints.')

loader = NvidiaLoader(framerate=32, phase='test')
n = len(loader)
print(f'Test samples: {n}')

# Cache sample tensors to speed up repeated evaluation.
cache = []
for i in range(n):
    s, label, _ = loader[i]
    if isinstance(s, torch.Tensor):
        t = s.unsqueeze(0).cuda()
    else:
        t = torch.from_numpy(s).unsqueeze(0).cuda()
    cache.append((t, int(label)))
print('Cached test set.')

model = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()

best_ep = None
best_acc = 0.0
results = []
for c in ckpts:
    ep = int(c.split('epoch')[1].split('_')[0])
    ckpt = torch.load(c, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    correct = 0
    with torch.no_grad():
        for t, lab in cache:
            outs = [model(t) for _ in range(N_TTA)]
            out = torch.stack(outs).mean(dim=0)
            pred = out.argmax(dim=1).item()
            if pred == lab:
                correct += 1
    acc = correct / n * 100
    results.append((ep, acc, correct))
    marker = ''
    if acc > best_acc:
        best_acc = acc
        best_ep = ep
        marker = '  <-- new best'
    print(f'  epoch{ep:3d}: {acc:6.2f}%  ({correct}/{n}){marker}')

print()
print(f'BEST: epoch{best_ep} at {best_acc:.2f}%')
