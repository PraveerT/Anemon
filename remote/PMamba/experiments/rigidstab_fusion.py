"""Fusion analysis: pmamba_rigidstab best_model + pmamba_base ep110.

best saved at oracle=93.98% (prec1=56.22%). High oracle vs base 89.83 suggests
strong complementary errors; question is whether linear routing can recover any.

Outputs: solo / oracle / alpha-blend / error correlation / per-class breakdown.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn.functional as F, numpy as np
import nvidia_dataloader
from models import motion

CKPT_STAB = '/notebooks/PMamba/experiments/work_dir/pmamba_rigidstab/best_model.pt'
CKPT_BASE = '/notebooks/PMamba/experiments/work_dir/pmamba_branch/epoch110_model.pt'

print('Loading rigidstab')
stab = motion.MotionRigidStabilize(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
sd = torch.load(CKPT_STAB, map_location='cuda')
stab.load_state_dict(sd['model_state_dict'], strict=False)
stab.eval()

print('Loading base')
base = motion.Motion(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
sd2 = torch.load(CKPT_BASE, map_location='cuda')
base.load_state_dict(sd2['model_state_dict'], strict=False)
base.eval()

loader_stab = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=True, assignment_mode='hungarian')
loader_base = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')

N = len(loader_stab)
assert N == len(loader_base), f"len mismatch: {N} vs {len(loader_base)}"
stab_logits = torch.zeros(N, 25)
base_logits = torch.zeros(N, 25)
labels = torch.zeros(N, dtype=torch.long)

print(f'Running inference on {N} test samples...')
with torch.no_grad():
    for i in range(N):
        # rigidstab inference
        s = loader_stab[i]; pts_d = s[0]; lab = s[1]
        inputs = {}
        for k, v in pts_d.items():
            if isinstance(v, np.ndarray): v = torch.from_numpy(v)
            inputs[k] = v.cuda().unsqueeze(0)
        out_s = stab(inputs)
        stab_logits[i] = out_s[0].cpu()
        labels[i] = int(lab)
        # base inference
        pts, lab2, _ = loader_base[i]
        assert int(lab2) == int(lab), f"label mismatch at i={i}"
        pts_t = (pts if torch.is_tensor(pts) else torch.from_numpy(pts)).float().cuda().unsqueeze(0)
        out_b = base(pts_t)
        base_logits[i] = out_b[0].cpu()
        if (i+1) % 100 == 0: print(f'  {i+1}/{N}')

stab_p = F.softmax(stab_logits, -1)
base_p = F.softmax(base_logits, -1)
stab_right = stab_p.argmax(-1) == labels
base_right = base_p.argmax(-1) == labels

stab_solo = stab_right.float().mean().item()
base_solo = base_right.float().mean().item()
oracle = (base_right | stab_right).float().mean().item()

# Alpha sweep
best_a = 0; best_a_acc = 0
for a in np.arange(0.0, 1.001, 0.01):
    f = a * base_p + (1 - a) * stab_p
    acc = (f.argmax(-1) == labels).float().mean().item()
    if acc > best_a_acc: best_a_acc = acc; best_a = a

# Error correlation
ep = (~base_right).float(); et = (~stab_right).float()
cov = ((ep - ep.mean()) * (et - et.mean())).mean().item()
r = cov / (ep.std().item() * et.std().item() + 1e-9)

both_r = (base_right & stab_right).sum().item()
base_only = (base_right & ~stab_right).sum().item()
stab_only = (~base_right & stab_right).sum().item()
both_w = (~base_right & ~stab_right).sum().item()

print(f"\n=== rigidstab best + pmamba_base ep110 fusion ===")
print(f"base solo:           {base_solo*100:.2f}%")
print(f"stab solo:           {stab_solo*100:.2f}%")
print(f"oracle:              {oracle*100:.2f}%  headroom +{(oracle-base_solo)*100:+.2f}pp")
print(f"alpha-blend:         {best_a_acc*100:.2f}% at a={best_a:.2f}  gain={(best_a_acc-base_solo)*100:+.2f}pp")
print(f"error correlation r: {r:.3f}")
print(f"\nBreakdown vs base:")
print(f"  both right    = {both_r}")
print(f"  base_only     = {base_only}")
print(f"  stab_only     = {stab_only}  (recovery ceiling if routed)")
print(f"  both_wrong    = {both_w}")
print(f"\nReference points (from QCC_SUBTRACTION.md):")
print(f"  TinyKNN 82.57:    fuse 90.66 (+0.83), r=0.44, qcc_only=16, oracle=93.15")
print(f"  rigidres 90.04:   fuse 94.19 oracle (all-time high), +0.21pp solo")

np.savez('/tmp/rigidstab_fuse.npz',
         stab_logits=stab_logits.numpy(),
         base_logits=base_logits.numpy(),
         labels=labels.numpy(),
         stab_solo=stab_solo, base_solo=base_solo, oracle=oracle,
         fuse=best_a_acc, fuse_a=best_a, r=r,
         base_only=base_only, stab_only=stab_only, both_wrong=both_w)
print("saved /tmp/rigidstab_fuse.npz")
