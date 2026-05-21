"""Self-supervised pretrain of the v2 AE (no opacity gating).

Losses:
  chamfer (bidirectional)
  temporal smoothness
  density-weighted coverage
  repulsion (hinge-squared)
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models.motion_cleanest_ae import (
    FrameEncoder, FrameDecoder,
    chamfer_two_sided, density_weighted_coverage, repulsion_loss,
)
from nvidia_dataloader import NvidiaLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-worker', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--feature-dim', type=int, default=128)
    p.add_argument('--K', type=int, default=1024)
    p.add_argument('--query-dim', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--num-attn-blocks', type=int, default=2)
    p.add_argument('--ffn-mult', type=int, default=4)
    p.add_argument('--chamfer-weight', type=float, default=1.0)
    p.add_argument('--temporal-weight', type=float, default=0.5)
    p.add_argument('--density-weight', type=float, default=1.0)
    p.add_argument('--density-knn', type=int, default=5)
    p.add_argument('--repulsion-weight', type=float, default=1.0)
    p.add_argument('--repulsion-radius', type=float, default=0.05)
    p.add_argument('--out', type=str, default='work_dir/ae_pretrain/ae_pretrain.pt')
    p.add_argument('--seed', type=int, default=2)
    return p.parse_args()


class AE(nn.Module):
    def __init__(self, feature_dim, K, query_dim, heads, num_attn_blocks, ffn_mult):
        super().__init__()
        self.encoder = FrameEncoder(feature_dim=feature_dim)
        self.decoder = FrameDecoder(
            feature_dim=feature_dim, K=K, query_dim=query_dim,
            heads=heads, num_attn_blocks=num_attn_blocks, ffn_mult=ffn_mult,
        )
        self.K = K

    def forward(self, xyz):
        return self.decoder(self.encoder(xyz))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = torch.device('cuda')
    dataset = NvidiaLoader(framerate=32, phase='train')
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_worker, drop_last=True, pin_memory=True,
    )
    print(f'[ae-pretrain] dataset {len(dataset)} samples, {len(loader)} batches/epoch')

    model = AE(args.feature_dim, args.K, args.query_dim, args.heads,
               args.num_attn_blocks, args.ffn_mult).to(device)
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    print(f'[ae-pretrain] params encoder={enc_p} decoder={dec_p} total={enc_p + dec_p}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float('inf')
    history = []
    for epoch in range(args.epochs):
        model.train()
        ep = {k: 0.0 for k in ('ch_a', 'ch_b', 'temp', 'density', 'rep')}
        n_batches = 0
        t0 = time.time()
        bar = tqdm(loader, desc=f'ep{epoch + 1}', leave=False)
        for batch_idx, data in enumerate(bar):
            inputs = data[0]
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(device, non_blocking=True)
            xyz = inputs[..., :3]
            B, T, N, _ = xyz.shape
            canonical = model(xyz)
            BT = B * T
            can_flat = canonical.reshape(BT, args.K, 3)
            inp_flat = xyz.reshape(BT, N, 3)

            ch_a, ch_b = chamfer_two_sided(can_flat, inp_flat)
            chamfer = (ch_a + ch_b) / 2
            temporal = ((canonical[:, 1:] - canonical[:, :-1]) ** 2).mean()
            density = density_weighted_coverage(can_flat, inp_flat, knn=args.density_knn)
            rep = repulsion_loss(can_flat, radius=args.repulsion_radius)

            loss = (args.chamfer_weight * chamfer
                    + args.temporal_weight * temporal
                    + args.density_weight * density
                    + args.repulsion_weight * rep)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep['ch_a'] += float(ch_a)
            ep['ch_b'] += float(ch_b)
            ep['temp'] += float(temporal)
            ep['density'] += float(density)
            ep['rep'] += float(rep)
            n_batches += 1
            bar.set_postfix({k: f'{v / n_batches:.4f}' for k, v in ep.items()})

        sched.step()
        for k in ep:
            ep[k] /= n_batches
        elapsed = time.time() - t0
        lr = opt.param_groups[0]['lr']
        print(f'[ae-pretrain] ep {epoch + 1:3d}/{args.epochs} '
              f'cha={ep["ch_a"]:.4f} chb={ep["ch_b"]:.4f} '
              f'temp={ep["temp"]:.4f} dens={ep["density"]:.4f} '
              f'rep={ep["rep"]:.4f} lr={lr:.2e} dt={elapsed:.1f}s', flush=True)
        history.append({'epoch': epoch + 1, **ep, 'lr': lr})

        score = (ep['ch_a'] + ep['ch_b']) / 2
        if score < best:
            best = score
            torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'config': vars(args),
                'history': history,
                'best_score': best,
                'best_epoch': epoch + 1,
            }, args.out)
            print(f'[ae-pretrain] saved best {args.out} chamfer={score:.5f}', flush=True)

    print(f'[ae-pretrain] done. best chamfer={best:.5f}')


if __name__ == '__main__':
    main()
