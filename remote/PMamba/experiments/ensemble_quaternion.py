"""Ensemble across v8a + v14a + v11b + v13b + v14b.

Each hit 79-80% via different feature paths. Average softmax logits across
all models per test sample. If unique-correct samples differ across models,
ensemble > best single.
"""
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import sys
sys.path.insert(0, '.')

from nvidia_dataloader import NvidiaQuaternionQCCParityLoader
from models.reqnn_motion import BearingQCCFeatureMotion

# (config yaml, best epoch checkpoint, qcc_variant override if needed)
MODELS = [
    ('quaternion_noaux_warmrestart_v8a.yaml',
     './work_dir/quaternion_noaux_warmrestart_v8a/epoch140_model.pt', None),
    ('quaternion_cycrig_side_warmrestart_v14a.yaml',
     './work_dir/quaternion_cycrig_side_warmrestart_v14a/epoch119_model.pt', None),
    ('quaternion_multiscale_warmrestart_v11b.yaml',
     './work_dir/quaternion_multiscale_warmrestart_v11b/epoch107_model.pt', None),
    ('quaternion_cycrig_warmrestart_v13b.yaml',
     './work_dir/quaternion_cycrig_warmrestart_v13b/epoch110_model.pt', None),
    ('quaternion_cycrig_mlp_warmrestart_v14b.yaml',
     './work_dir/quaternion_cycrig_mlp_warmrestart_v14b/epoch117_model.pt', None),
]
TTA = 5


def _to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: _to_cuda(v) for k, v in x.items()}
    return x


def eval_model(cfg_path, ckpt_path, variant_override=None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    test_args = dict(cfg['test_loader_args'])
    test_args['phase'] = 'test'
    test_args['pts_size'] = cfg['pts_size']

    model_args = dict(cfg['model_args'])
    if variant_override is not None:
        model_args['qcc_variant'] = variant_override

    ds = NvidiaQuaternionQCCParityLoader(**test_args)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    model = BearingQCCFeatureMotion(**model_args).cuda()
    state = torch.load(ckpt_path, map_location='cuda')
    sd = state['model_state_dict'] if 'model_state_dict' in state else state
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'  {cfg_path}: loaded (missing {len(missing)}, unexpected {len(unexpected)})')
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            image = _to_cuda(data[0])
            label = data[1]
            outs = []
            for _ in range(TTA):
                o = model(image)
                if isinstance(o, (list, tuple)):
                    o = o[0]
                outs.append(F.softmax(o, dim=-1))
            probs = torch.stack(outs).mean(dim=0)  # (1, C)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(label.numpy() if torch.is_tensor(label) else np.array(label))
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def main():
    print(f'Evaluating {len(MODELS)} models with TTA={TTA}...\n')
    per_model_probs = []
    labels = None
    for cfg_path, ckpt_path, variant in MODELS:
        probs, lbl = eval_model(cfg_path, ckpt_path, variant)
        per_model_probs.append(probs)
        if labels is None:
            labels = lbl
        preds = probs.argmax(axis=-1)
        acc = (preds == labels).mean() * 100
        print(f'  -> solo acc: {acc:.2f}%')

    print('\n=== Ensemble (mean softmax) ===')
    avg = np.mean(per_model_probs, axis=0)
    preds_ens = avg.argmax(axis=-1)
    acc_ens = (preds_ens == labels).mean() * 100
    print(f'Ensemble acc (TTA={TTA}): {acc_ens:.2f}%  ({(preds_ens==labels).sum()}/{len(labels)})')

    print('\n=== Majority vote ===')
    all_preds = np.stack([p.argmax(axis=-1) for p in per_model_probs])  # (M, N)
    maj = np.array([np.bincount(all_preds[:, i]).argmax() for i in range(all_preds.shape[1])])
    acc_maj = (maj == labels).mean() * 100
    print(f'Majority-vote acc: {acc_maj:.2f}%')

    print('\n=== Unique-correct analysis ===')
    for i, (cfg, _, _) in enumerate(MODELS):
        name = cfg.split('_')[1]
        correct = (all_preds[i] == labels)
        unique = correct.copy()
        for j in range(len(MODELS)):
            if j != i:
                unique &= ~(all_preds[j] == labels)
        print(f'  {cfg}: {correct.sum()} correct, {unique.sum()} UNIQUE correct')

    # Oracle: for each sample, if ANY model is correct, count it
    any_correct = np.any(all_preds == labels[None, :], axis=0)
    print(f'\nOracle (any model correct): {any_correct.sum()}/{len(labels)} = {any_correct.mean()*100:.2f}%')


if __name__ == '__main__':
    main()
