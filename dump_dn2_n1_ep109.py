import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_deltanet_v2 import MotionDeltaNetV2

ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
m = MotionDeltaNetV2(num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
                     multi_scale_num_scales=5, dn_hidden_dim=256, dn_num_layers=2,
                     dn_num_heads=4, dn_head_dim=64, dn_expand_v=2, dn_dropout=0.3,
                     dn_bidirectional=True).cuda()
state = torch.load('work_dir/pmamba_baseline_deltanet_v2/epoch109_model.pt', map_location='cpu')['model_state_dict']
res = m.load_state_dict(state, strict=False)
print(f'missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
m.eval()
pl, ll = [], []
with torch.no_grad():
    for b in loader:
        x, y = b[0].cuda().float(), b[1]
        pl.append(torch.softmax(m(x), -1).cpu().numpy())
        ll.append(y.numpy() if hasattr(y, 'numpy') else np.array(y))
P = np.concatenate(pl); L = np.concatenate(ll)
out = 'dump_probs_runs/deltanet_v2_n1_ep109.npz'
np.savez(out, probs=P, labels=L)
print(f'DN2(N1)_ep109 solo = {(P.argmax(1)==L).mean()*100:.2f}% -> {out}')
