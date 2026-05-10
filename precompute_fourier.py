"""Precompute per-sample Fourier band-energy targets from MediaPipe landmarks.
5 fingertips x 3 axes x 4 bands = 60-D target vector per sample.
"""
import numpy as np

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
OUT = '/notebooks/PMamba/dataset/Nvidia/Processed/fourier_targets.npz'
TIPS = [4, 8, 12, 16, 20]

def fillnan(arr):
    valid = np.isfinite(arr[..., 0]).all(axis=-1)
    last = None; out = arr.copy()
    for t in range(out.shape[0]):
        if valid[t]: last = out[t]
        elif last is not None: out[t] = last
    for t in range(out.shape[0]):
        if not np.isfinite(out[t]).all():
            for t2 in range(t+1, out.shape[0]):
                if valid[t2]: out[t] = out[t2]; break
            else: out[t] = 0
    return out

def fourier_target(lm):
    """Returns log(1+band_energy) for 5 tips x 3 axes x 4 bands = 60."""
    out = []
    for tip in TIPS:
        for ax in range(3):
            sig = lm[:, tip, ax]
            sig = sig - sig.mean()
            f = np.abs(np.fft.rfft(sig)) ** 2
            n = len(f); b = max(1, n // 4)
            bands = [f[:b].sum(), f[b:2*b].sum(), f[2*b:3*b].sum(), f[3*b:].sum()]
            out.extend(bands)
    return np.log1p(np.array(out, dtype=np.float32))

print('loading landmarks...')
sk = dict(np.load(SK, allow_pickle=False))
print(f'{len(sk)} samples')

targets = {}
for i, (k, lm_raw) in enumerate(sk.items()):
    if lm_raw.shape[0] < 4:
        targets[k] = np.zeros(60, dtype=np.float32); continue
    lm = fillnan(lm_raw)
    targets[k] = fourier_target(lm)
    if i % 200 == 0:
        print(f'  {i+1}/{len(sk)}', flush=True)

np.savez(OUT, **targets)
print(f'saved {len(targets)} samples to {OUT}')
print(f'sanity: dim={list(targets.values())[0].shape}, mean={np.mean(list(targets.values())):.3f}, std={np.std(list(targets.values())):.3f}')
