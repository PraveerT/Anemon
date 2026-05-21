"""Self-supervised pretraining of the FrameEncoder + FrameDecoder.

Loads the NVGesture training data via the standard NvidiaLoader.
Trains the AE alone on:
  L = chamfer_weight * chamfer(canonical, input) + temporal_weight * smoothness
Saves encoder + decoder state_dict to a checkpoint that the classifier
training will load as a warm start.

Run from experiments/ directory:
    python3 -u pretrain_ae.py --epochs 30 --out work_dir/ae_pretrain.pt
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

from models.motion_cleanest_ae import FrameEncoder, FrameDecoder, _chamfer_distance
from nvidia_dataloader import NvidiaLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-worker', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--latent-dim', type=int, default=128)
    p.add_argument('--K', type=int, default=128)
    p.add_argument('--decoder-hidden', type=int, default=256)
    p.add_argument('--chamfer-weight', type=float, default=1.0)
    p.add_argument('--temporal-weight', type=float, default=0.5)
    p.add_argument('--out', type=str, default='work_dir/ae_pretrain/ae_pretrain.pt')
    p.add_argument('--seed', type=int, default=2)
    return p.parse_args()


class AE(nn.Module):
    def __init__(self, latent_dim, K, decoder_hidden):
        super().__init__()
        self.encoder = FrameEncoder(latent_dim=latent_dim)
        self.decoder = FrameDecoder(latent_dim=latent_dim, K=K, hidden=decoder_hidden)
        self.K = K

    def forward(self, xyz):
        latent = self.encoder(xyz)
        canonical = self.decoder(latent)
        return canonical


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

    model = AE(args.latent_dim, args.K, args.decoder_hidden).to(device)
    print(f'[ae-pretrain] params encoder={sum(p.numel() for p in model.encoder.parameters())}'
          f' decoder={sum(p.numel() for p in model.decoder.parameters())}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float('inf')
    history = []
    for epoch in range(args.epochs):
        model.train()
        ep_chamfer = 0.0
        ep_temporal = 0.0
        n_batches = 0
        t0 = time.time()
        bar = tqdm(loader, desc=f'ep{epoch + 1}', leave=False)
        for batch_idx, data in enumerate(bar):
            inputs = data[0]
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(device, non_blocking=True)        # (B, T, N, 4)
            xyz = inputs[..., :3]                                # (B, T, N, 3)
            B, T, N, _ = xyz.shape

            canonical = model(xyz)                                # (B, T, K, 3)
            BT = B * T
            chamfer = _chamfer_distance(
                canonical.reshape(BT, args.K, 3),
                xyz.reshape(BT, N, 3),
            )
            temporal = ((canonical[:, 1:] - canonical[:, :-1]) ** 2).mean()
            loss = args.chamfer_weight * chamfer + args.temporal_weight * temporal

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_chamfer += float(chamfer)
            ep_temporal += float(temporal)
            n_batches += 1
            bar.set_postfix({
                'ch': f'{float(chamfer):.4f}',
                'tm': f'{float(temporal):.4f}',
            })

        sched.step()
        ep_chamfer /= n_batches
        ep_temporal /= n_batches
        elapsed = time.time() - t0
        lr = opt.param_groups[0]['lr']
        print(f'[ae-pretrain] ep {epoch + 1:3d}/{args.epochs} '
              f'chamfer={ep_chamfer:.5f} temporal={ep_temporal:.5f} '
              f'lr={lr:.2e} dt={elapsed:.1f}s', flush=True)
        history.append({
            'epoch': epoch + 1,
            'chamfer': ep_chamfer,
            'temporal': ep_temporal,
            'lr': lr,
        })

        score = ep_chamfer + ep_temporal
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
            print(f'[ae-pretrain] saved best {args.out} score={score:.5f}', flush=True)

    print(f'[ae-pretrain] done. best={best:.5f} at ep{[h["epoch"] for h in history if h["chamfer"] + h["temporal"] == best][0]}')


if __name__ == '__main__':
    main()
