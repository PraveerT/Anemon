"""Show what the classifier actually sees at ep1 (pts_size=48).

Three panels per frame:
  1. raw input (N=512)
  2. v2 canonical (K=1024 from frozen AE)
  3. 48 points fed to the classifier (random subset of canonical, same RNG
     the model uses inside _sample_points)
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


OUT_DIR = 'viz_out_pts48'
os.makedirs(OUT_DIR, exist_ok=True)

ckpt = torch.load('work_dir/ae_pretrain_v2/ae_pretrain.pt', map_location='cpu')
cfg = ckpt['config']
K = cfg['K']
enc = FrameEncoder(feature_dim=cfg['feature_dim']).cuda()
enc.load_state_dict(ckpt['encoder']); enc.eval()
dec = FrameDecoder(
    feature_dim=cfg['feature_dim'], K=K,
    query_dim=cfg['query_dim'], heads=cfg['heads'],
    num_attn_blocks=cfg['num_attn_blocks'], ffn_mult=cfg['ffn_mult'],
).cuda()
dec.load_state_dict(ckpt['decoder']); dec.eval()

ds = NvidiaLoader(framerate=32, phase='test')
# Pick one sample per class for the first 6 classes
seen = set()
chosen = []
for idx in range(len(ds)):
    lbl = int(ds[idx][1])
    if lbl in seen:
        continue
    seen.add(lbl)
    chosen.append((idx, lbl))
    if len(chosen) >= 6:
        break

print(f'AE config K={K}, chamfer={ckpt["best_score"]:.4f}')
print(f'chosen: {chosen}')

rng = np.random.RandomState(7)
idx_colors = rng.rand(K, 3) * 0.7 + 0.2

# Bake step: take first 512 of K=1024 canonical (matches what
# CanonicalNvidiaLoader would store). Then per-frame random 48 of those
# 512 simulates _sample_points at ep1 (parent's randperm behaviour).
KEEP = 512
PTS48 = 48

for s_i, (ds_idx, label) in enumerate(chosen):
    pts, _, _ = ds[ds_idx]
    xyz_full = pts[..., :3].unsqueeze(0).cuda()

    with torch.no_grad():
        canonical = dec(enc(xyz_full))[0].cpu().numpy()          # (32, K, 3)
    inp_np = pts[..., :3].cpu().numpy()                            # (32, 512, 3)

    # Bake's first-512 slice
    canon_512 = canonical[:, :KEEP, :]                              # (32, 512, 3)
    col_512 = idx_colors[:KEEP]

    # Per-frame random 48 of 512 (one fixed RNG for reproducibility in viz)
    rng48 = np.random.RandomState(0)
    pick_idx = np.stack([rng48.permutation(KEEP)[:PTS48] for _ in range(32)])
    canon_48 = np.take_along_axis(canon_512, pick_idx[..., None], axis=1)
    col_48 = idx_colors[pick_idx]                                   # (32, 48, 3)

    all_pts = np.concatenate([inp_np.reshape(-1, 3), canonical.reshape(-1, 3)], axis=0)
    mn = all_pts.min(axis=0); mx = all_pts.max(axis=0)
    pad = (mx - mn) * 0.05

    fig = plt.figure(figsize=(9, 3), dpi=72)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    def draw(t):
        for ax in (ax1, ax2, ax3):
            ax.cla()
            ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
            ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
            ax.set_zlim(mn[2] - pad[2], mx[2] + pad[2])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=15, azim=-60)
        ax1.scatter(inp_np[t, :, 0], inp_np[t, :, 1], inp_np[t, :, 2],
                    s=1.5, c='#2080d0', alpha=0.5)
        ax1.set_title(f'input N=512', fontsize=8)
        ax2.scatter(canon_512[t, :, 0], canon_512[t, :, 1], canon_512[t, :, 2],
                    s=3, c=col_512, alpha=0.85)
        ax2.set_title(f'v2 canonical (first {KEEP} of K={K})', fontsize=8)
        ax3.scatter(canon_48[t, :, 0], canon_48[t, :, 1], canon_48[t, :, 2],
                    s=14, c=col_48[t], alpha=0.95)
        ax3.set_title(f'classifier sees at ep1 (random {PTS48})', fontsize=8)
        fig.suptitle(f'sample {ds_idx} · class {label} | t={t}', fontsize=9)
        return ()

    anim = FuncAnimation(fig, draw, frames=32, interval=80, blit=False)
    out = f'{OUT_DIR}/pts48_sample_{s_i:02d}_class{label:02d}.gif'
    anim.save(out, writer=PillowWriter(fps=8), dpi=60)
    plt.close(fig)
    size = os.path.getsize(out) / 1024
    print(f'wrote {out} ({size:.0f} KB)')

print('done')
