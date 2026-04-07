"""Snapshot ensemble of quaternion_branch checkpoints.

Loads N checkpoints, runs each on the test set, averages logits, computes
accuracy.  Free win if checkpoint variance is meaningful.
"""
import argparse
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.reqnn_motion import BearingQCCFeatureMotion
import nvidia_dataloader  # noqa: F401


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloader(cfg):
    cls_name = cfg['dataloader']
    module_name, class_name = cls_name.rsplit('.', 1)
    import importlib
    mod = importlib.import_module(module_name)
    loader_cls = getattr(mod, class_name)
    test_args = dict(cfg.get('test_loader_args', {}))
    test_args['phase'] = 'test'
    test_args['framerate'] = cfg.get('framesize', 32)
    test_args['pts_size'] = cfg.get('pts_size', 256)
    test_args.setdefault('return_correspondence', False)
    dataset = loader_cls(**test_args)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get('test_batch_size', 1),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def build_model(cfg, weights_path, pts_size):
    model_args = dict(cfg['model_args'])
    model_args['pts_size'] = pts_size
    model = BearingQCCFeatureMotion(**model_args)
    ckpt = torch.load(weights_path, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'  loaded {weights_path}: missing={len(missing)} unexpected={len(unexpected)}')
    model.cuda()
    model.eval()
    model.pts_size = pts_size
    return model


@torch.no_grad()
def collect_logits(model, loader):
    all_logits = []
    all_labels = []
    for data in loader:
        image = {k: v.cuda(non_blocking=True) for k, v in data[0].items()} if isinstance(data[0], dict) else data[0].cuda(non_blocking=True)
        label = data[1].cuda(non_blocking=True)
        out = model(image)
        all_logits.append(out.cpu())
        all_labels.append(label.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--pts-size', type=int, default=256)
    parser.add_argument('--mode', choices=['logits', 'softmax'], default='logits')
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f'Building dataloader...')
    loader = build_dataloader(cfg)
    print(f'Loaded {len(loader.dataset)} test samples')

    print(f'\nEvaluating {len(args.checkpoints)} checkpoints with pts_size={args.pts_size}...')
    accumulated = None
    labels = None
    for ckpt_path in args.checkpoints:
        model = build_model(cfg, ckpt_path, args.pts_size)
        logits, lbl = collect_logits(model, loader)
        if args.mode == 'softmax':
            logits = F.softmax(logits, dim=-1)
        if accumulated is None:
            accumulated = logits.clone()
            labels = lbl
        else:
            accumulated = accumulated + logits
        # Per-checkpoint accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == lbl).float().mean().item() * 100
        print(f'    individual acc: {acc:.2f}%')
        del model
        torch.cuda.empty_cache()

    accumulated = accumulated / len(args.checkpoints)
    preds = accumulated.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    acc = correct / total * 100
    print(f'\n=== ENSEMBLE ({len(args.checkpoints)} checkpoints, mode={args.mode}) ===')
    print(f'Total Correct: {correct}/{total}')
    print(f'Overall Accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
