"""Render canonical-vs-input animations for the AE pretrain check.

For 3 test samples (different gesture classes), produce a GIF showing:
  [ input cloud ]  [ canonical cloud (colored by index k) ]
  T=32 frames at low resolution.

Coloring: each canonical index k gets a fixed color across all 32 frames.
If correspondence holds, index k visually tracks a hand region as the
gesture progresses; if it doesn't, colors wander randomly.

Output: viz_out/canonical_sample_{i}.gif (small files, easy to deploy).
"""
import os
import sys
import io

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
K = ckpt['config']['K']
enc = FrameEncoder(ckpt['config']['latent_dim']).cuda()
enc.load_state_dict(ckpt['encoder']); enc.eval()
dec = FrameDecoder(ckpt['config']['latent_dim'], K, ckpt['config']['decoder_hidden']).cuda()
dec.load_state_dict(ckpt['decoder']); dec.eval()

ds = NvidiaLoader(framerate=32, phase='test')
# Pick 3 samples from different classes
sample_indices = [0, 100, 250]
print(f'AE config: K={K}, latent={ckpt["config"]["latent_dim"]}, decoder_hidden={ckpt["config"]["decoder_hidden"]}')
print(f'AE pretrain score={ckpt["best_score"]:.4f} at ep{ckpt["best_epoch"]}')

# Persistent per-index color map (so canonical index k has the same color
# across every frame, every sample). Random but reproducible.
rng = np.random.RandomState(7)
idx_colors = rng.rand(K, 3) * 0.7 + 0.2   # avoid pure black/white

for s_i, ds_idx in enumerate(sample_indices):
    pts, label, path = ds[ds_idx]
    label = int(label)
    xyz_full = pts[..., :3].unsqueeze(0).cuda()                # (1, 32, 512, 3)

    with torch.no_grad():
        latent = enc(xyz_full)
        canonical = dec(latent)                                 # (1, 32, K, 3)

    inp_np = pts[..., :3].cpu().numpy()                         # (32, 512, 3)
    can_np = canonical[0].cpu().numpy()                         # (32, K, 3)

    # Shared axis limits across all frames so motion is visible
    all_pts = np.concatenate([inp_np.reshape(-1, 3), can_np.reshape(-1, 3)], axis=0)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    pad = (mx - mn) * 0.05

    fig = plt.figure(figsize=(7, 3.5), dpi=80)
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
                    s=2, c='#2080d0', alpha=0.6)
        ax1.set_title(f'input (N=512) | t={t}/{31}', fontsize=9)

        ax2.scatter(can_np[t, :, 0], can_np[t, :, 1], can_np[t, :, 2],
                    s=10, c=idx_colors, alpha=0.9)
        ax2.set_title(f'canonical (K={K}, color=index)', fontsize=9)

        fig.suptitle(f'sample {ds_idx} | class {label}', fontsize=10)
        return ()

    anim = FuncAnimation(fig, draw, frames=32, interval=80, blit=False)
    out = f'{OUT_DIR}/canonical_sample_{s_i}_class{label}.gif'
    anim.save(out, writer=PillowWriter(fps=8), dpi=70)
    plt.close(fig)
    size = os.path.getsize(out) / 1024
    print(f'wrote {out} ({size:.0f} KB)')

print('done')
