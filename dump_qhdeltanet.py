import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_qhdeltanet import MotionQHDeltaNet

ckpt_path = 'work_dir/pmamba_baseline_qhdeltanet/best_model.pt'
ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
model = MotionQHDeltaNet(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8,
                          multi_scale_num_scales=5, qh_hidden_dim=128, qh_num_layers=2,
                          qh_num_heads=4, qh_n_q=4, qh_n_v=8, qh_dropout=0.3,
                          qh_bidirectional=True).cuda()
state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
res = model.load_state_dict(state, strict=False)
print(f'missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for i, batch in enumerate(loader):
        x, y = batch[0].cuda().float(), batch[1]
        logits = model(x)
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(y.numpy() if hasattr(y, 'numpy') else np.array(y))
P = np.concatenate(all_probs); L = np.concatenate(all_labels)
out = 'dump_probs_runs/qhdeltanet_best.npz'
np.savez(out, probs=P, labels=L)
print(f'shape={P.shape} test_acc={(P.argmax(1)==L).mean()*100:.2f}% -> {out}')
