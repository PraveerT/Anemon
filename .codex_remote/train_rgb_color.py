"""Strong color-RGB R(2+1)D for NVGesture (orthogonal non-DSN fusion partner).

Key boost: DEPTH-GUIDED FOREGROUND CROP. RGB and depth are camera-aligned, so we
take the hand bbox from the depth>0 mask (union over frames) and crop the RGB to
it -> hand-focused, background removed (the lever that lifted the depth model).
Kinetics-pretrained R(2+1)D-18. Dumps test+train logits (sigs) for honest fusion.
"""
import os, time, random, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

RGB = '/notebooks/cvpr_data/rgb'
DEPTH = '/notebooks/cvpr_data/depth'
SPLITS = os.environ.get('SPLITS', '/notebooks/cvpr_data/dataset_splits')
CACHE = os.environ.get('RGB_CACHE', '/notebooks/Anemon/dataset/Nvidia/Processed/rgb_fgcrop_cache')
WD = '/notebooks/Anemon/experiments/work_dir/rgb_fgcrop_r2p1d'  # overridden in main by --arch / --tag
# r2plus1d kinetics norm; swin3d/mvit use imagenet-ish norm (set per-arch in main)
KMEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
KSTD = np.array([0.22803, 0.22145, 0.216989], np.float32)


def read_split(p):
    out = []
    for ln in open(p):
        a = ln.split()
        if len(a) >= 3:
            out.append((a[0].strip('/'), int(a[1]), int(a[2])))
    return out


def fg_bbox(depth_frames, pad=0.18):
    m = np.zeros(depth_frames[0].shape, bool)
    for a in depth_frames:
        m |= a > 10
    ys, xs = np.where(m)
    h, w = depth_frames[0].shape
    if len(xs) == 0:
        return (0, 0, w, h)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = int(max(y1 - y0, x1 - x0) * pad) + 4
    return (max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p))


def build_cache(split, phase, size=128):
    os.makedirs(CACHE, exist_ok=True)
    base = os.path.join(CACHE, f'{phase}_s{size}')
    if os.path.exists(base + '.npy'):
        return (np.load(base + '.npy', mmap_mode='r'),
                np.load(base + '_lab.npy'), np.load(base + '_sig.npy', allow_pickle=True))
    recs = read_split(os.path.join(SPLITS, split))
    maxf = max(r[1] for r in recs)
    clips = np.zeros((len(recs), maxf, size, size, 3), np.uint8)
    labs = np.array([r[2] for r in recs], np.int64)
    sigs = np.array([r[0] for r in recs], dtype=object)
    t0 = time.time()
    for i, (rel, nf, _) in enumerate(recs):
        dd, rd = os.path.join(DEPTH, rel), os.path.join(RGB, rel)
        dep = [np.asarray(Image.open(os.path.join(dd, f'{t:06d}.jpg')).convert('L'), np.uint8) for t in range(nf)]
        bb = fg_bbox(dep)
        for t in range(nf):
            img = Image.open(os.path.join(rd, f'{t:06d}.jpg')).convert('RGB').crop(bb).resize((size, size), Image.BILINEAR)
            clips[i, t] = np.asarray(img, np.uint8)
        if (i + 1) % 200 == 0 or i + 1 == len(recs):
            print(f'[cache {phase}] {i+1}/{len(recs)} {time.time()-t0:.0f}s', flush=True)
    np.save(base + '.npy', clips); np.save(base + '_lab.npy', labs); np.save(base + '_sig.npy', sigs)
    return np.load(base + '.npy', mmap_mode='r'), labs, sigs


class DS(Dataset):
    def __init__(self, clips, labs, sigs, frames=40, crop=112, train=False, resize=0):
        self.c, self.l, self.s = clips, labs, sigs
        self.frames, self.crop, self.train, self.resize = frames, crop, train, resize
        self.cache = clips.shape[2]; self.maxf = clips.shape[1]

    def __len__(self):
        return len(self.l)

    def _tidx(self):
        if self.train:
            span = random.randint(max(self.frames, int(self.maxf * 0.6)), self.maxf)
            st = random.randint(0, self.maxf - span)
            idx = np.linspace(st, st + span - 1, self.frames)
            return np.clip(np.rint(idx + np.random.uniform(-0.4, 0.4, self.frames)), 0, self.maxf - 1).astype(np.int64)
        return np.linspace(0, self.maxf - 1, self.frames).round().astype(np.int64)

    def __getitem__(self, i):
        x = np.asarray(self.c[i, self._tidx()], np.float32) / 255.0   # (T,H,W,3)
        lim = self.cache - self.crop
        if self.train and lim > 0:
            y0, x0 = random.randint(0, lim), random.randint(0, lim)
        else:
            y0 = x0 = max(0, lim // 2)
        x = x[:, y0:y0 + self.crop, x0:x0 + self.crop, :]
        if self.train:
            if random.random() < 0.5:
                x = x[:, :, ::-1, :].copy()
            # light color jitter (brightness/contrast)
            x = np.clip(x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08), 0, 1)
        x = (x - KMEAN) / KSTD
        t = torch.from_numpy(x).permute(3, 0, 1, 2).float()      # (3,T,H,W)
        if self.resize and self.resize != t.shape[-1]:
            t = nn.functional.interpolate(t, size=(self.resize, self.resize), mode='bilinear', align_corners=False)
        return t, int(self.l[i]), str(self.s[i])


@torch.no_grad()
def evaluate(model, loader, dev, dump=None):
    model.eval(); L, Y, S = [], [], []
    for x, y, s in loader:
        with torch.autocast('cuda'):
            o = model(x.to(dev)) + model(torch.flip(x, dims=[4]).to(dev))
        L.append(o.float().cpu().numpy()); Y.append(y.numpy()); S += list(s)
    L = np.concatenate(L); Y = np.concatenate(Y)
    if dump:
        np.savez(dump, logits=L, labels=Y, sigs=np.array(S, dtype=object))
    return (L.argmax(1) == Y).mean() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--blr', type=float, default=4e-5)
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--arch', default='r2plus1d_18')
    ap.add_argument('--cache-size', type=int, default=128)
    ap.add_argument('--crop', type=int, default=112)
    ap.add_argument('--resize', type=int, default=0)
    a = ap.parse_args()
    global WD, KMEAN, KSTD
    WD = os.environ.get('WD', f'/notebooks/Anemon/experiments/work_dir/rgb_fgcrop_{a.arch}')
    if a.arch != 'r2plus1d_18':   # swin3d / mvit use imagenet-style norm
        KMEAN = np.array([0.485, 0.456, 0.406], np.float32)
        KSTD = np.array([0.229, 0.224, 0.225], np.float32)
    os.makedirs(WD, exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    tr = build_cache('train.txt', 'train', size=a.cache_size); va = build_cache('valid.txt', 'valid', size=a.cache_size)
    mk = lambda ds, sh: DataLoader(ds, batch_size=a.bs, shuffle=sh, num_workers=6, drop_last=sh, pin_memory=True, persistent_workers=True)
    dtr = mk(DS(*tr, frames=a.frames, crop=a.crop, train=True, resize=a.resize), True)
    dva = mk(DS(*va, frames=a.frames, crop=a.crop, train=False, resize=a.resize), False)
    dtr_e = mk(DS(*tr, frames=a.frames, crop=a.crop, train=False, resize=a.resize), False)
    dev = 'cuda'
    import torchvision.models.video as V
    if a.arch == 'r2plus1d_18':
        m = V.r2plus1d_18(weights=V.R2Plus1D_18_Weights.KINETICS400_V1)
        m.fc = nn.Linear(m.fc.in_features, 25); head = m.fc
    elif a.arch == 'swin3d_t':
        m = V.swin3d_t(weights=V.Swin3D_T_Weights.KINETICS400_V1)
        m.head = nn.Linear(m.head.in_features, 25); head = m.head
    elif a.arch == 'mvit_v2_s':
        m = V.mvit_v2_s(weights=V.MViT_V2_S_Weights.KINETICS400_V1)
        m.head[-1] = nn.Linear(m.head[-1].in_features, 25); head = m.head[-1]
    else:
        raise SystemExit('bad arch')
    m = m.to(dev)
    hids = {id(p) for p in head.parameters()}
    body = [p for p in m.parameters() if id(p) not in hids]
    head = list(head.parameters())
    opt = torch.optim.AdamW([{'params': body, 'lr': a.blr}, {'params': head, 'lr': a.lr}], weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler(); lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0
    for ep in range(a.epochs):
        m.train(); t0 = time.time(); tot = cor = seen = 0
        for x, y, _ in dtr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                o = m(x); loss = lossf(o, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * y.numel(); cor += (o.argmax(1) == y).sum().item(); seen += y.numel()
        sched.step()
        acc = evaluate(m, dva, dev, dump=os.path.join(WD, 'test_logits.npz'))
        msg = f'ep{ep:3d} tr_loss={tot/seen:.4f} tr_acc={cor/seen*100:.2f}% te_acc={acc:.2f}% best={best:.2f} dt={time.time()-t0:.0f}s'
        if acc > best:
            best = acc
            evaluate(m, dva, dev, dump=os.path.join(WD, 'best_logits.npz'))
            evaluate(m, dtr_e, dev, dump=os.path.join(WD, 'train_logits.npz'))
            torch.save(m.state_dict(), os.path.join(WD, 'best.pt')); msg += ' *'
        print(msg, flush=True); open(os.path.join(WD, 'run.log'), 'a').write(msg + '\n')
    print(f'DONE best={best:.2f}', flush=True)


if __name__ == '__main__':
    main()
