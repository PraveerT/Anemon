"""Dump TRAIN softmaxes for all train-best ckpts. Used for honest fusion weight optimization."""
import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_realdeltanet import MotionRealDeltaNet
from models.motion_bilateralrd import MotionBilateralRD
from models.motion_attrd import MotionAttRD

def dump(model, loader_cls, ckpt_path, out_path, name):
    ds = loader_cls(framerate=32, phase='train')
    loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    model.load_state_dict(state, strict=False); model.eval()
    pl, ll = [], []
    with torch.no_grad():
        for b in loader:
            x, y = b[0].cuda().float(), b[1]
            pl.append(torch.softmax(model(x), -1).cpu().numpy())
            ll.append(y.numpy() if hasattr(y, 'numpy') else np.array(y))
    P = np.concatenate(pl); L = np.concatenate(ll)
    np.savez(out_path, probs=P, labels=L)
    print(f'  {name} TRAIN solo = {(P.argmax(1)==L).mean()*100:.2f}% ({P.shape}) -> {out_path}')

# RD(N1) ep118
m = MotionRealDeltaNet(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8,
                       multi_scale_num_scales=5, rd_hidden_dim=128, rd_num_layers=2,
                       rd_num_heads=4, rd_n_q=4, rd_n_v=8, rd_dropout=0.3,
                       rd_bidirectional=True).cuda()
dump(m, nvidia_dataloader.NvidiaLoader,
     'work_dir/pmamba_baseline_realdeltanet/epoch118_model.pt',
     'dump_probs_runs/realdeltanet_ep118_TRAIN.npz', 'RD(N1)_ep118')
del m; torch.cuda.empty_cache()

# RD(N2) ep118
m = MotionRealDeltaNet(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8,
                       multi_scale_num_scales=5, rd_hidden_dim=128, rd_num_layers=2,
                       rd_num_heads=4, rd_n_q=4, rd_n_v=8, rd_dropout=0.3,
                       rd_bidirectional=True).cuda()
dump(m, nvidia_dataloader.NvidiaDTWLoader,
     'work_dir/pmamba_dtw_realdeltanet/epoch118_model.pt',
     'dump_probs_runs/realdeltanet_n2_ep118_TRAIN.npz', 'RD(N2)_ep118')
del m; torch.cuda.empty_cache()

# BRD ep112
m = MotionBilateralRD(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8,
                      multi_scale_num_scales=5, brd_hidden_dim=128, brd_num_layers=2,
                      brd_num_heads=4, brd_n_q=4, brd_n_v=8, brd_dropout=0.3,
                      brd_t_bidirectional=True, brd_fuse='sum').cuda()
dump(m, nvidia_dataloader.NvidiaLoader,
     'work_dir/pmamba_baseline_bilateralrd/epoch112_model.pt',
     'dump_probs_runs/brd_ep112_TRAIN.npz', 'BRD_ep112')
del m; torch.cuda.empty_cache()

# AttRD ep120
m = MotionAttRD(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8,
                multi_scale_num_scales=5, ar_hidden_dim=128, ar_num_layers=2,
                ar_num_heads=4, ar_n_q=4, ar_n_v=8, ar_d_read=32, ar_dropout=0.3,
                ar_bidirectional=True).cuda()
dump(m, nvidia_dataloader.NvidiaLoader,
     'work_dir/pmamba_baseline_attrd/epoch120_model.pt',
     'dump_probs_runs/attrd_ep120_TRAIN.npz', 'AttRD_ep120')
print('DONE')
