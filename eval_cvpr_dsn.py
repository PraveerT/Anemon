"""Eval CVPR I3DWTrans NV-K (depth) on our 482-sample test set, dump softmax."""
import sys, os, numpy as np, torch
sys.path.insert(0, '/notebooks/MotionRGBD')
sys.path.insert(0, '/notebooks/MotionRGBD/lib')
sys.path.insert(0, '/notebooks/MotionRGBD/utils')
sys.path.insert(0, '/notebooks/MotionRGBD/tools')
os.chdir('/notebooks/MotionRGBD')

from types import SimpleNamespace
from torch.utils.data import DataLoader
from lib.datasets.NvGesture import NvData
from lib.model.DSN import DSNNet

# Build args matching config/NvGesture.yml + Network common
args = SimpleNamespace(
    # data
    data='/notebooks/cvpr_data',
    splits='/notebooks/cvpr_data/dataset_splits',
    dataset='NvGesture',
    type='K',                         # K = depth
    Network='I3DWTrans',
    num_classes=25,
    sample_duration=64,
    sample_size=224,
    batch_size=4,
    test_batch_size=2,
    num_workers=4,
    nprocs=1,
    local_rank=0,
    dist=False,
    # transform
    flip=0.0,
    rotated=0.0,
    angle='(0, 0)',
    Blur=False,
    resize='(256, 256)',
    crop_size=224,
    low_frames=16,
    media_frames=32,
    high_frames=48,
    # model
    w=4,
    temper=0.4,
    recoupling=True,
    knn_attention=0.7,
    sharpness=False,                  # disable sharpness for eval (no temp_schedule needed)
    temp=[0.04, 0.07],
    frp=True,
    SEHeads=1,
    N=6,
    grad_clip=5.0,
    SYNC_BN=0,
    epoch=0,
    epochs=100,
    init_epochs=0,
    DEBUG=False,
    MultiLoss=True,
    pretrained=False,
)

# Build dataset (ground_truth = path to valid.txt)
ds = NvData(args, ground_truth=f'{args.splits}/valid.txt', modality='depth', phase='valid')
print(f'len test = {len(ds)}')
loader = DataLoader(ds, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

# Build model
model = DSNNet(args, num_classes=args.num_classes, pretrained=False).cuda()

# Load weights
ckpt = torch.load('/notebooks/PMamba/experiments/work_dir/pmamba_cvpr_official/ckpt.pt', map_location='cpu')
sd = ckpt['model']
# Remove possible 'module.' prefix
sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
res = model.load_state_dict(sd, strict=False)
print(f'loaded: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')
if res.missing_keys[:5]: print('  miss5:', res.missing_keys[:5])
if res.unexpected_keys[:5]: print('  unexp5:', res.unexpected_keys[:5])

model.eval()
all_probs, all_labels = [], []
import time
t0 = time.time()
with torch.no_grad():
    for i, batch in enumerate(loader):
        clip, garr, label, _ = batch
        clip = clip.cuda().float()
        garr = garr.cuda().float()
        # DSNNet returns (logits, distill, aux). logits = (x, xs, xm, xl) — x is merged
        logits_tuple, _, _ = model(clip, garr)
        x, xs, xm, xl = logits_tuple
        # store both logits and probs
        all_probs.append(torch.softmax(x, dim=-1).cpu().numpy())
        if not hasattr(model, '_logits_buf'): model._logits_buf = []
        model._logits_buf.append(x.cpu().numpy())
        all_labels.append(label.numpy() if hasattr(label, 'numpy') else np.array(label))
        if i % 20 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(loader)} elapsed={elapsed:.0f}s', flush=True)

P = np.concatenate(all_probs); L = np.concatenate(all_labels)
LG = np.concatenate(model._logits_buf)
out = '/notebooks/PMamba/experiments/dump_probs_runs/cvpr_dsn_K_depth.npz'
np.savez(out, probs=P, labels=L, logits=LG)
print(f'logits stats: mean abs={np.abs(LG).mean():.3f} max={LG.max():.3f} min={LG.min():.3f}')
print(f'\nshape={P.shape} test_acc={(P.argmax(1)==L).mean()*100:.2f}% -> {out}')
