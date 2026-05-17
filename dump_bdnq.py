"""Dump BDN-Q test-set softmax for honest fusion.

Picks the train-best epoch (highest 'Mean training acc' in the log), then
runs the corresponding checkpoint on the NVGesture test set and writes a
softmax + labels .npz under dump_probs_runs/.
"""
import os, re, glob, sys, argparse, numpy as np, torch
sys.path.insert(0, '/notebooks/PMamba/experiments')
os.chdir('/notebooks/PMamba/experiments')

from torch.utils.data import DataLoader
import nvidia_dataloader
from models.motion_bdn_q import MotionBDeltaQ

WORKDIR = 'work_dir/pmamba_baseline_bdnq'
LOG = 'work_dir/bdnq_train.log'
OUT = 'dump_probs_runs/bdnq_train_best.npz'


def find_train_best(log_path):
    """Return (epoch, train_acc) of the highest mean train acc."""
    best = (None, -1.0)
    cur_ep = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Training epoch:\s+(\d+)', line)
            if m: cur_ep = int(m.group(1))
            m = re.search(r'Mean training acc:\s+(\d+\.\d+)', line)
            if m and cur_ep is not None:
                ta = float(m.group(1))
                # cur_ep at this line is the START of NEXT epoch; logged train acc is
                # for previous epoch. So actual epoch = cur_ep - 1.
                ep = cur_ep - 1
                if ta > best[1]:
                    best = (ep, ta)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ep', type=int, default=None, help='override epoch')
    p.add_argument('--out', default=OUT)
    args = p.parse_args()

    if args.ep is None:
        ep, ta = find_train_best(LOG)
        print(f'[train-best] ep{ep} train_acc={ta:.2f}%')
    else:
        ep = args.ep
        print(f'[forced] ep{ep}')

    # Closest available checkpoint
    ckpts = sorted(glob.glob(f'{WORKDIR}/epoch*_model.pt'),
                   key=lambda p: int(re.search(r'epoch(\d+)_', p).group(1)))
    ckpt_epochs = [int(re.search(r'epoch(\d+)_', p).group(1)) for p in ckpts]
    if ep not in ckpt_epochs:
        # snap to nearest available checkpoint
        nearest = min(ckpt_epochs, key=lambda e: abs(e - ep))
        print(f'[snap] ep{ep} not saved, using nearest ep{nearest}')
        ep = nearest
    ckpt_path = f'{WORKDIR}/epoch{ep}_model.pt'
    print(f'[ckpt] {ckpt_path}')

    ds = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
    loader = DataLoader(ds, batch_size=1, num_workers=4, shuffle=False)
    model = MotionBDeltaQ(
        num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8,
        multi_scale_num_scales=5,
        bdnq_hidden_dim=128, bdnq_num_layers=2, bdnq_num_heads=4,
        bdnq_n_q=4, bdnq_n_v=8, bdnq_buffer_size=1, bdnq_dropout=0.3,
        bdnq_bidirectional=True,
    ).cuda()
    state = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    res = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].cuda().float()
            y = batch[1]
            logits = model(x)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_labels.append(y.numpy() if hasattr(y, 'numpy') else np.array(y))
    P = np.concatenate(all_probs)
    L = np.concatenate(all_labels).astype(np.int64).ravel()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, probs=P, labels=L, epoch=ep)
    acc = (P.argmax(1) == L).mean() * 100
    print(f'[dump] shape={P.shape} test_acc(ep{ep})={acc:.2f}% -> {args.out}')


if __name__ == '__main__':
    main()
