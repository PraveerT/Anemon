"""Render canonical-vs-input animations for many samples across classes.

Output: viz_out/canonical_sample_NN_classCC.gif for 12 test samples.
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, '.')

from models.motion_cleanest_ae import FrameEncoder, FrameDecoder
from nvidia_dataloader import NvidiaLoader


OUT_DIR = 'viz_out'
os.makedirs(OUT_DIR, exist_ok=True)

ckpt = torch.load('work_dir/ae_pretrain/ae_pretrain.pt', map_location='cpu')
cfg = ckpt['config']
K = cfg['K']
enc = FrameEncoder(feature_dim=cfg['feature_dim']).cuda()
enc.load_state_dict(ckpt['encoder']); enc.eval()
dec = FrameDecoder(
    feature_dim=cfg['feature_dim'], K=K,
    query_dim=cfg['query_dim'], heads=cfg['heads'],
).cuda()
dec.load_state_dict(ckpt['decoder']); dec.eval()

ds = NvidiaLoader(framerate=32, phase='test')
print(f'AE config: K={K}, feature_dim={cfg["feature_dim"]}, query_dim={cfg["query_dim"]}, heads={cfg["heads"]}')
print(f'AE pretrain score={ckpt["best_score"]:.4f} at ep{ckpt["best_epoch"]}')

# Pick 12 samples spanning different classes by scanning labels.
n_total = len(ds)
seen_classes = set()
chosen = []
for idx in range(n_total):
    lbl = int(ds[idx][1])
    if lbl in seen_classes:
        continue
    seen_classes.add(lbl)
    chosen.append((idx, lbl))
    if len(chosen) >= 12:
        break

print(f'chosen samples: {chosen}')

rng = np.random.RandomState(7)
idx_colors = rng.rand(K, 3) * 0.7 + 0.2

for s_i, (ds_idx, label) in enumerate(chosen):
    pts, _, _ = ds[ds_idx]
    xyz_full = pts[..., :3].unsqueeze(0).cuda()

    with torch.no_grad():
        feats = enc(xyz_full)
        canonical = dec(feats)

    inp_np = pts[..., :3].cpu().numpy()
    can_np = canonical[0].cpu().numpy()

    all_pts = np.concatenate([inp_np.reshape(-1, 3), can_np.reshape(-1, 3)], axis=0)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    pad = (mx - mn) * 0.05

    fig = plt.figure(figsize=(6, 3), dpi=72)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    def draw(t):
        for ax in (ax1, ax2):
            ax.cla()
            ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
            ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
            ax.set_zlim(mn[2] - pad[2], mx[2] + pad[2])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=15, azim=-60)

        ax1.scatter(inp_np[t, :, 0], inp_np[t, :, 1], inp_np[t, :, 2],
                    s=1.5, c='#2080d0', alpha=0.5)
        ax1.set_title(f'input N=512 | t={t}', fontsize=8)

        ax2.scatter(can_np[t, :, 0], can_np[t, :, 1], can_np[t, :, 2],
                    s=4, c=idx_colors, alpha=0.8)
        ax2.set_title(f'canonical K={K}', fontsize=8)

        fig.suptitle(f'sample {ds_idx} · class {label}', fontsize=9)
        return ()

    anim = FuncAnimation(fig, draw, frames=32, interval=80, blit=False)
    out = f'{OUT_DIR}/canonical_sample_{s_i:02d}_class{label:02d}.gif'
    anim.save(out, writer=PillowWriter(fps=8), dpi=60)
    plt.close(fig)
    size = os.path.getsize(out) / 1024
    print(f'wrote {out} ({size:.0f} KB)')

print('done')
