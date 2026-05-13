"""For each model, find epoch with highest 'Mean training acc'."""
import os, re

logs = {
    'RD(N1)':  'pmamba_baseline_realdeltanet',
    'RD(N2)':  'pmamba_dtw_realdeltanet',
    'RD(N14)': 'pmamba_antinet1_realdeltanet',
    'BRD':     'pmamba_baseline_bilateralrd',
    'AttRD':   'pmamba_baseline_attrd',
}

root = '/notebooks/PMamba/experiments/work_dir'
for name, d in logs.items():
    path = f'{root}/{d}/log.txt'
    if not os.path.exists(path):
        print(f'SKIP {name}: missing {path}'); continue
    ep_acc = []
    cur_ep = None
    with open(path) as f:
        for line in f:
            m = re.search(r'Training epoch:\s+(\d+)', line)
            if m:
                cur_ep = int(m.group(1))
                continue
            m = re.search(r'Mean training acc:\s+([\d.]+)%', line)
            if m and cur_ep is not None:
                ep_acc.append((cur_ep, float(m.group(1))))
    # top 5 by acc
    top = sorted(ep_acc, key=lambda x: -x[1])[:5]
    print(f'\n=== {name} ===  (total epochs: {len(ep_acc)})')
    for ep, acc in top:
        print(f'  ep{ep:3d}  train_acc={acc:.4f}%')
