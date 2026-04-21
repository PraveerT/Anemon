"""Random-sampling TTA evaluation.

Forces random point sampling at eval time and averages predictions across N
forward passes per sample.  Reduces sampling variance.
"""
import argparse
import sys
import os
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
        dataset, batch_size=cfg.get('test_batch_size', 1), shuffle=False,
        num_workers=4, pin_memory=True,
    )


def build_model(cfg, weights_path, pts_size):
    model_args = dict(cfg['model_args'])
    model_args['pts_size'] = pts_size
    model = BearingQCCFeatureMotion(**model_args)
    ckpt = torch.load(weights_path, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    model.pts_size = pts_size
    return model


@torch.no_grad()
def tta_eval(model, loader, n_tta, force_random):
    correct = 0
    total = 0
    # Monkey-patch _sample_point_indices to use random sampling for TTA
    if force_random:
        original = model._sample_point_indices

        def random_sample(point_count, device):
            sample_size = min(model.pts_size, point_count)
            if sample_size == point_count:
                return None
            return torch.randperm(point_count, device=device)[:sample_size]
        model._sample_point_indices = random_sample

    try:
        for data in loader:
            image = data[0].cuda(non_blocking=True) if torch.is_tensor(data[0]) else data[0]
            if isinstance(image, dict):
                image = {k: v.cuda(non_blocking=True) for k, v in image.items()}
            label = data[1].cuda(non_blocking=True)
            outs = []
            for _ in range(n_tta):
                out = model(image)
                outs.append(F.softmax(out, dim=-1))
            avg = torch.stack(outs).mean(dim=0)
            pred = avg.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    finally:
        if force_random:
            model._sample_point_indices = original

    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--pts-size', type=int, default=256)
    parser.add_argument('--n-tta', type=int, default=10)
    parser.add_argument('--force-random', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    print(f'Building dataloader...')
    loader = build_dataloader(cfg)
    print(f'Loaded {len(loader.dataset)} test samples')
    print(f'TTA={args.n_tta}, force_random={args.force_random}, pts_size={args.pts_size}')

    model = build_model(cfg, args.weights, args.pts_size)
    correct, total = tta_eval(model, loader, args.n_tta, args.force_random)
    acc = correct / total * 100
    print(f'\n=== {args.weights} ===')
    print(f'Total Correct: {correct}/{total}')
    print(f'Overall Accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()
