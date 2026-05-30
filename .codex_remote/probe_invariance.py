"""Invariance fragility probe for CN-XXL on NVGesture.

Run CN-XXL on the 482 test clips under different TEST-TIME transforms of the
point cloud, in real (un-standardized) space. For each transform report:
  acc, fragility (baseline-correct -> wrong), fixes (baseline-wrong -> correct).
Geometric transforms (rot/scale/mirror/translate) un-standardize xyz, transform,
re-standardize. Also TTA: average softmax over K random small transforms.
Tells us which invariances the errors are fragile to -> honest augmentation/TTA lever.
"""
import sys, numpy as np, torch, yaml
sys.path.insert(0, '/notebooks/Anemon/experiments')
from utils import import_class

CFG = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/config.yaml'
CKPT = '/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head/best_model.pt'
arg = yaml.load(open(CFG), Loader=yaml.FullLoader)
dev = 'cuda'

# stats for un-standardize (xyz channels 0,1,2)
stats = np.load('/notebooks/Anemon/experiments/nvidia_dataset_stats.npy', allow_pickle=True).item()
MEAN = torch.tensor([stats['x_mean'], stats['y_mean'], stats['z_mean']], dtype=torch.float32, device=dev)
STD = torch.tensor([stats['x_std'], stats['y_std'], stats['z_std']], dtype=torch.float32, device=dev)

# model
M = import_class(arg['model'])(**arg['model_args']).to(dev).eval()
sd = torch.load(CKPT, map_location='cpu')
sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
sd = {k.replace('module.', ''): v for k, v in sd.items()}
M.load_state_dict(sd, strict=False)

# test data
DL = import_class(arg['dataloader'])
ds = DL(framerate=32, phase='test', **{k: v for k, v in arg.get('test_loader_args', {}).items() if k != 'phase' and k != 'framerate'})
X = torch.stack([ds[i][0] for i in range(len(ds))]).to(dev)   # (N,T,P,C)
Y = torch.tensor([ds[i][1] for i in range(len(ds))]).to(dev)
N = len(Y)
print(f'N={N}  input shape {tuple(X.shape)}')


def unstd(x):
    x = x.clone(); x[..., :3] = x[..., :3] * STD + MEAN; return x

def restd(x):
    x = x.clone(); x[..., :3] = (x[..., :3] - MEAN) / STD; return x


def rotmat(axis, rad):
    c, s = np.cos(rad), np.sin(rad)
    if axis == 'z': R = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    elif axis == 'y': R = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else: R = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return torch.tensor(R, dtype=torch.float32, device=dev)


def geo(x, fn):           # apply fn to real-space xyz
    xr = unstd(x); xyz = xr[..., :3]
    xc = xyz.mean((1, 2), keepdim=True)         # per-clip centroid
    xyz2 = fn(xyz - xc) + xc
    xr = xr.clone(); xr[..., :3] = xyz2
    return restd(xr)


def rot(x, axis, deg):
    R = rotmat(axis, np.radians(deg))
    return geo(x, lambda p: p @ R.T)

def scale(x, s):
    return geo(x, lambda p: p * s)

def mirror(x):
    return geo(x, lambda p: p * torch.tensor([-1., 1., 1.], device=dev))

def jitter(x, sig):
    xr = unstd(x); xr = xr.clone(); xr[..., :3] += torch.randn_like(xr[..., :3]) * sig; return restd(xr)

def treverse(x):
    return x.flip(1)


@torch.no_grad()
def infer(x):
    out = []
    for i in range(0, N, 16):
        out.append(M(x[i:i+16]).float())   # fp32 (quat head uses eigh, no fp16)
    return torch.cat(out)


base = infer(X); bp = base.argmax(1); bcorrect = bp == Y
err = ~bcorrect
print(f'baseline acc {bcorrect.sum().item()}/{N}  errors={err.sum().item()}')


def report(name, x):
    p = infer(x).argmax(1)
    acc = (p == Y).sum().item()
    frag = ((bcorrect) & (p != Y)).sum().item()       # was right, now wrong
    fix = ((err) & (p == Y)).sum().item()             # was wrong, now right
    flip = (p != bp).sum().item()
    print(f'  {name:16s} acc={acc:3d}  flips={flip:3d}  breaks={frag:3d}  fixes={fix:3d}')


print('\n=== single transforms ===')
for ax in ['z', 'y', 'x']:
    for d in [10, 20, 45]:
        report(f'rot{ax}+{d}', rot(X, ax, d))
for s in [0.85, 0.92, 1.1, 1.2]:
    report(f'scale{s}', scale(X, s))
report('mirror_x', mirror(X))
report('treverse', treverse(X))
for sg in [0.02, 0.05]:
    report(f'jitter{sg}', jitter(X, sg))


@torch.no_grad()
def tta(transforms):
    acc_sm = torch.zeros(N, base.shape[1], device=dev)
    for t in transforms:
        acc_sm += torch.softmax(infer(t), 1)
    p = acc_sm.argmax(1); return (p == Y).sum().item(), ((err) & (p == Y)).sum().item(), ((bcorrect) & (p != Y)).sum().item()


print('\n=== TTA (avg softmax incl. identity) ===')
small_rots = [X] + [rot(X, a, d) for a in ['z', 'y', 'x'] for d in [-12, 12]]
a, f, b = tta(small_rots); print(f'  rot-TTA(±12 xyz): acc={a} fixes={f} breaks={b}')
scales = [X, scale(X, 0.92), scale(X, 1.08)]
a, f, b = tta(scales); print(f'  scale-TTA(.92,1.08): acc={a} fixes={f} breaks={b}')
allt = small_rots + [scale(X, 0.9), scale(X, 1.1), mirror(X)]
a, f, b = tta(allt); print(f'  combined-TTA: acc={a} fixes={f} breaks={b}')
