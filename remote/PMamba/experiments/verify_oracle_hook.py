"""Verify the oracle hook now aligns correctly after shuffle=False fix.

Runs the same logic as main.py's eval + hook:
- DepthVideoLoader (no shuffle)
- Accumulate model softmax probs
- Load cached PMamba probs
- Compute oracle + best-fusion exactly like _maybe_compute_oracle
"""
import sys, os
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
import numpy as np, torch
from torch.utils.data import DataLoader

from depth_branch.model import DepthCNNLSTM
from depth_branch.dataloader import DepthVideoLoader

dm = DepthCNNLSTM(num_classes=25, in_channels=4, rigidity_dim=0,
                  rigidity_aux_dim=0, clip_reweight_beta=1.0).cuda()
cd = torch.load('work_dir/depth_branch/v9_clean/best_model.pt', map_location='cpu')
dm.load_state_dict(cd.get('model_state_dict', cd), strict=False); dm.eval()

dpl = DepthVideoLoader(framerate=32, phase='test', img_size=112,
                       use_tops=True, use_rigidity=True, rigidity_norm_scale=1.0)
loader = DataLoader(dpl, batch_size=4, shuffle=False, drop_last=False, num_workers=0)

probs = []; labels = []
with torch.no_grad():
    for (dt, rt), lab, _ in loader:
        dt = dt.cuda(); rt = rt.cuda()
        outs = torch.stack([dm((dt, rt)) for _ in range(3)]).mean(0)
        probs.append(torch.softmax(outs, dim=1).cpu().numpy())
        labels.append(lab.numpy())
probs = np.concatenate(probs, 0); labels = np.concatenate(labels, 0)

cache = np.load('work_dir/pmamba_branch/pmamba_test_preds.npz')
pm_probs = cache['probs']; pm_labels = cache['labels']
print(f"probs align: {(labels == pm_labels).all()} (vs labels-array match)")

pm_correct = pm_probs.argmax(1) == pm_labels
md_correct = probs.argmax(1) == pm_labels
oracle = (pm_correct | md_correct).mean() * 100
print(f"PMamba: {pm_correct.mean()*100:.2f} | model: {md_correct.mean()*100:.2f} | oracle: {oracle:.2f}")
print(f"only-model: {int((~pm_correct & md_correct).sum())}  both-wrong: {int((~pm_correct & ~md_correct).sum())}")
best_a, best_acc = 1.0, pm_correct.mean()*100
for ai in range(0, 105, 5):
    a = ai/100
    fp = (a*pm_probs + (1-a)*probs).argmax(1)
    acc = (fp == pm_labels).mean() * 100
    if acc > best_acc: best_acc = acc; best_a = a
print(f"Best fusion: a={best_a:.2f} -> {best_acc:.2f} (+{best_acc - pm_correct.mean()*100:.2f})")
