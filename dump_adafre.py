import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_adafre import MotionAdaFre

ckpt_path = 'work_dir/pmamba_adafre/best_model.pt'
ds = nvidia_dataloader.NvidiaDTWLoader(framerate=32, phase='test')
loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
model = MotionAdaFre(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8,
                    multi_scale_num_scales=5, T_for_adafre=32).cuda()
state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
res = model.load_state_dict(state, strict=False)
print(f'missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
if res.missing_keys[:3]: print('  miss:', res.missing_keys[:3])
if res.unexpected_keys[:3]: print('  unexp:', res.unexpected_keys[:3])
# Use a batch>=2 dummy for lazy init (BN requires bs>1 in train mode)
loader_bs2 = DataLoader(ds, batch_size=2, num_workers=0, shuffle=False)
with torch.no_grad():
    for i, batch in enumerate(loader_bs2):
        x = batch[0].cuda().float()
        _ = model(x)
        break
# Now AdaFreBlock is initialized — reload weights including its params
res2 = model.load_state_dict(state, strict=False)
print(f'after init reload: missing={len(res2.missing_keys)} unexpected={len(res2.unexpected_keys)}')
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
out = 'dump_probs_runs/adafre_best.npz'
np.savez(out, probs=P, labels=L)
print(f'shape={P.shape} test_acc={(P.argmax(1)==L).mean()*100:.2f}% -> {out}')
