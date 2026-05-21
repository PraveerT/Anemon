"""Apply a frozen AE to every NVGesture sample and save the canonical
ordered point cloud to disk.

Output: dataset/Nvidia/Processed/canonical_{phase}.npy
  shape (num_samples, T, K, 4) where the 4th channel is per-frame
  normalised time (matches the original NvidiaLoader layout).

Run from experiments/:
    python3 -u bake_canonical.py --ckpt work_dir/ae_pretrain/ae_pretrain.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from models.motion_cleanest_ae import FrameEncoder, FrameDecoder
from nvidia_dataloader import NvidiaLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--out-dir', type=str, default='../dataset/Nvidia/Processed')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-worker', type=int, default=8)
    return p.parse_args()


def bake_phase(phase, model, K, batch_size, num_worker, out_path):
    dataset = NvidiaLoader(framerate=32, phase=phase)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_worker, drop_last=False,
    )
    print(f'[bake] {phase}: {len(dataset)} samples')

    out = np.empty((len(dataset), 32, K, 4), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.int64)
    cursor = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            inputs = data[0].cuda(non_blocking=True)             # (B, T, N, 4)
            lbl = data[1]
            xyz = inputs[..., :3]
            B, T, N, _ = xyz.shape

            point_feats = model['encoder'](xyz)
            canonical = model['decoder'](point_feats)             # (B, T, K, 3)

            t_idx = torch.arange(T, device=xyz.device, dtype=xyz.dtype)
            t_norm = (t_idx - t_idx.mean()) / t_idx.std().clamp(min=1e-6)
            t_channel = t_norm.view(1, T, 1, 1).expand(B, T, K, 1)
            payload = torch.cat([canonical, t_channel], dim=-1)   # (B, T, K, 4)

            out[cursor:cursor + B] = payload.cpu().numpy()
            labels[cursor:cursor + B] = lbl.numpy() if hasattr(lbl, 'numpy') else lbl
            cursor += B

            if batch_idx % 10 == 0:
                print(f'[bake] {phase}: {cursor}/{len(dataset)}')

    np.save(out_path, out)
    np.save(out_path.replace('.npy', '_labels.npy'), labels)
    print(f'[bake] wrote {out_path} shape={out.shape} dtype={out.dtype} '
          f'size={out.nbytes / 1024 / 1024:.1f} MB  dt={time.time() - t0:.1f}s')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['config']
    K = cfg['K']
    print(f'[bake] AE config: K={K}, feature_dim={cfg["feature_dim"]}, '
          f'query_dim={cfg["query_dim"]}, heads={cfg["heads"]}')
    print(f'[bake] AE pretrain score={ckpt["best_score"]:.4f} at ep{ckpt["best_epoch"]}')

    enc = FrameEncoder(feature_dim=cfg['feature_dim']).cuda()
    dec = FrameDecoder(
        feature_dim=cfg['feature_dim'], K=K,
        query_dim=cfg['query_dim'], heads=cfg['heads'],
    ).cuda()
    enc.load_state_dict(ckpt['encoder']); enc.eval()
    dec.load_state_dict(ckpt['decoder']); dec.eval()
    model = {'encoder': enc, 'decoder': dec}

    for phase in ('train', 'test'):
        out_path = os.path.join(args.out_dir, f'canonical_{phase}.npy')
        bake_phase(phase, model, K, args.batch_size, args.num_worker, out_path)


if __name__ == '__main__':
    main()
