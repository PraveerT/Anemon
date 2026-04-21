"""Per-class accuracy + confusion matrix for v8a (80.08%) to find weaknesses."""
import torch
import numpy as np
import yaml
import sys
sys.path.insert(0, '.')

from nvidia_dataloader import NvidiaQuaternionQCCParityLoader
from models.reqnn_motion import BearingQCCFeatureMotion

CFG = 'quaternion_noaux_warmrestart_v8a.yaml'
CKPT = './work_dir/quaternion_noaux_warmrestart_v8a/epoch140_model.pt'

def main():
    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    test_args = dict(cfg['test_loader_args'])
    test_args['phase'] = 'test'
    test_args['pts_size'] = cfg['pts_size']

    ds = NvidiaQuaternionQCCParityLoader(**test_args)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    model = BearingQCCFeatureMotion(**cfg['model_args']).cuda()
    state = torch.load(CKPT, map_location='cuda')
    sd = state['model_state_dict'] if 'model_state_dict' in state else state
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'Loaded. missing={len(missing)} unexpected={len(unexpected)}')
    if missing:
        print('  missing keys (first 5):', list(missing)[:5])
    if unexpected:
        print('  unexpected keys (first 5):', list(unexpected)[:5])
    model.eval()

    num_classes = cfg['model_args']['num_classes']

    def _to_cuda(x):
        if torch.is_tensor(x):
            return x.cuda()
        if isinstance(x, dict):
            return {k: _to_cuda(v) for k, v in x.items()}
        return x

    all_preds, all_labels = [], []
    N_TTA = 3
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            image = _to_cuda(data[0])
            label = data[1]
            outputs = []
            for _ in range(N_TTA):
                output = model(image)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                outputs.append(output)
            avg = torch.stack(outputs).mean(dim=0)
            pred = avg.argmax(dim=-1).cpu().numpy()
            lbl = label.numpy() if torch.is_tensor(label) else np.array(label)
            all_preds.extend(pred.tolist())
            all_labels.extend(lbl.tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    acc = (preds == labels).mean() * 100
    print(f'\nOverall accuracy (TTA={N_TTA}): {acc:.2f}%  ({(preds==labels).sum()}/{len(labels)})')

    print('\nPer-class acc (worst first):')
    per_class = []
    for c in range(num_classes):
        mask = labels == c
        n = int(mask.sum())
        if n > 0:
            a = (preds[mask] == c).mean() * 100
            per_class.append((a, c, n))
    per_class.sort()
    for a, c, n in per_class:
        flag = ' <-- WORST' if a < 60 else (' <-- weak' if a < 75 else '')
        print(f'  class {c:2d}: {a:5.1f}%  ({n} samples){flag}')

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    print('\nTop confusions (true -> pred: count):')
    confs = []
    for t in range(num_classes):
        for p in range(num_classes):
            if t != p and cm[t][p] > 0:
                confs.append((cm[t][p], t, p))
    confs.sort(reverse=True)
    for cnt, t, p in confs[:20]:
        print(f'  {t:2d} -> {p:2d}: {cnt}')

    # Summary: is failure mode motion-heavy or pose-heavy?
    # Low-accuracy classes that confuse with each other may indicate motion similarity
    print('\nClasses with acc < 70%:', [c for _, c, _ in per_class if _ < 70])

if __name__ == '__main__':
    main()
