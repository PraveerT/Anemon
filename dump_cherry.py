"""Dump test-best (cherry-pick) checkpoints for BRD and AttRD."""
import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')
from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_bilateralrd import MotionBilateralRD
from models.motion_attrd import MotionAttRD

def dump(model, loader, ckpt_path, out_path, model_name):
    state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'{model_name}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].cuda().float(), batch[1]
            logits = model(x)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_labels.append(y.numpy() if hasattr(y, 'numpy') else np.array(y))
    P = np.concatenate(all_probs); L = np.concatenate(all_labels)
    np.savez(out_path, probs=P, labels=L)
    print(f'  test_acc={(P.argmax(1)==L).mean()*100:.2f}% -> {out_path}')

ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)

mb = MotionBilateralRD(num_classes=25, pts_size=256, knn=[32,24,48,24],
                       topk=8, multi_scale_num_scales=5,
                       brd_hidden_dim=128, brd_num_layers=2, brd_num_heads=4,
                       brd_n_q=4, brd_n_v=8, brd_dropout=0.3,
                       brd_t_bidirectional=True, brd_fuse='sum').cuda()
dump(mb, loader, 'work_dir/pmamba_baseline_bilateralrd/epoch119_model.pt',
     'dump_probs_runs/brd_ep119.npz', 'BRD_ep119')
del mb; torch.cuda.empty_cache()

ma = MotionAttRD(num_classes=25, pts_size=256, knn=[32,24,48,24],
                 topk=8, multi_scale_num_scales=5,
                 ar_hidden_dim=128, ar_num_layers=2, ar_num_heads=4,
                 ar_n_q=4, ar_n_v=8, ar_d_read=32, ar_dropout=0.3,
                 ar_bidirectional=True).cuda()
dump(ma, loader, 'work_dir/pmamba_baseline_attrd/epoch117_model.pt',
     'dump_probs_runs/attrd_ep117.npz', 'AttRD_ep117')
