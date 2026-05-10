import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion import Motion

ckpt = 'work_dir/pmamba_skeleton/best_model.pt'
ds = nvidia_dataloader.NvidiaSkeletonLoader(framerate=32, phase='test')
loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
model = Motion(num_classes=25, pts_size=21, knn=[8, 6, 8, 6], topk=4, multi_scale_num_scales=3).cuda()
state = torch.load(ckpt, map_location='cpu')['model_state_dict']
res = model.load_state_dict(state, strict=False)
print(f'missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for i, batch in enumerate(loader):
        x, y = batch[0].cuda().float(), batch[1]
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy() if hasattr(y, 'numpy') else np.array(y))
P = np.concatenate(all_probs); L = np.concatenate(all_labels)
out = 'dump_probs_runs/net72_skeleton_best.npz'
np.savez(out, probs=P, labels=L)
print(f'shape={P.shape} test_acc={(P.argmax(1)==L).mean()*100:.2f}% -> {out}')
