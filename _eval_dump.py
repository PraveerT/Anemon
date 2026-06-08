import os, sys
sys.path.insert(0, '.')
import numpy as np, torch
from torch.utils.data import DataLoader
from nvidia_dataloader import NvidiaLoader
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead

torch.manual_seed(0); np.random.seed(0)
dev = 'cuda'
margs = dict(num_classes=25, pts_size=172, topk=8, knn=[32, 24, 48, 24],
             multi_scale_num_scales=5, lxl_hidden_dim=256, lxl_mlp_dim=512,
             lxl_num_layers=4, lxl_dropout=0.3, lxl_bidirectional=True,
             lxl_residual_scale=0.7)
model = MotionCleanestLinXLQuatHead(**margs).to(dev)
ck = torch.load('/notebooks/Manta/experiments/work_dir/pgcnet_quat_head/best_model.pt',
                map_location=dev)
sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
missing, unexpected = model.load_state_dict(sd, strict=False)
print('missing', len(missing), 'unexpected', len(unexpected), flush=True)
model.eval(); model.pts_size = 172

ds = NvidiaLoader(framerate=32, phase='test', datatype='depth')
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
trues, preds, samp = [], [], []
with torch.no_grad():
    for x, y, line in dl:
        x = x.to(dev)
        out = model(x)
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(np.asarray(y))
        pts = x.permute(0, 3, 1, 2)                 # B,4,T,512
        N = pts.shape[3]; ss = min(172, N)
        idx = torch.linspace(0, N - 1, ss, device=x.device).long()
        sp = pts[:, :3, :, idx].permute(0, 2, 3, 1).cpu().numpy()   # B,T,172,3
        samp.append(sp)
trues = np.concatenate(trues); preds = np.concatenate(preds); samp = np.concatenate(samp)
acc = (trues == preds).mean() * 100
print('ACC %.2f  n=%d  wrong=%d' % (acc, len(trues), int((trues != preds).sum())), flush=True)
np.savez('/notebooks/wt9191/experiments/quat9108_dump.npz',
         trues=trues, preds=preds, samp=samp)
print('SAVED', samp.shape, flush=True)
