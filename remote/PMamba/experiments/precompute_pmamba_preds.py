"""Cache PMamba@epoch110 softmax probabilities for the test set.

Saves to work_dir/pmamba_branch/pmamba_test_preds.npy of shape (N, 25).
Runs once; main.py's oracle hook loads this cache to compute per-epoch oracle
and fusion without re-running PMamba.
"""
import os, sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
import numpy as np
import torch

from models.motion import Motion
from nvidia_dataloader import NvidiaLoader

PTS = 256
N_TTA = 3
OUT = 'work_dir/pmamba_branch/pmamba_test_preds.npy'

m = Motion(num_classes=25, pts_size=PTS, knn=[32, 24, 48, 24], topk=8).cuda()
cp = torch.load('work_dir/pmamba_branch/epoch110_model.pt', map_location='cpu')
m.load_state_dict(cp.get('model_state_dict', cp), strict=False); m.eval()

L = NvidiaLoader(framerate=32, phase='test')
n = len(L)
probs = np.zeros((n, 25), dtype=np.float32)
labels = np.zeros(n, dtype=np.int64)
with torch.no_grad():
    for i in range(n):
        s, lab, _ = L[i]
        x = s.unsqueeze(0).cuda() if isinstance(s, torch.Tensor) else torch.from_numpy(s).unsqueeze(0).cuda()
        out = torch.stack([m(x) for _ in range(N_TTA)]).mean(0)
        probs[i] = torch.softmax(out, dim=1).cpu().numpy()[0]
        labels[i] = int(lab)
        if i % 100 == 0: print(f"  {i}/{n}")

np.savez(OUT.replace('.npy', '.npz'), probs=probs, labels=labels)
np.save(OUT, probs)
print(f"saved {OUT} and .npz; correct={(probs.argmax(1) == labels).sum()}/{n}")
