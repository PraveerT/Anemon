"""Count BDN buffer ejections per forward pass on NVGesture (synthetic input).

We just need to know what T the BDeltaBlock sees inside the PMamba pipeline.
A synthetic input of the standard shape (B, 3, T=framesize, N=pts_size)
exercises the same code path; the BDN ejection count depends only on T.
"""
import sys, os, yaml, importlib, torch
sys.path.insert(0, '/notebooks/PMamba')
sys.path.insert(0, '/notebooks/PMamba/experiments')

from models.motion_bdn import BDeltaBlock

_records = []
_orig_forward = BDeltaBlock.forward
def _patched(self, x):
    B, T, _ = x.shape
    _records.append((T, self.W, max(0, T - self.W)))
    return _orig_forward(self, x)
BDeltaBlock.forward = _patched

cfg = yaml.safe_load(open('/notebooks/PMamba/experiments/pmamba_baseline_bdn.yaml'))

# Synthetic input matching the BDN config (framesize=32, pts_size=96)
B = 1
C = 3
T_in = cfg['framesize']           # 32
N_in = 256                        # training ramps pts to 256 by mid-training
x = torch.randn(B, C, T_in, N_in, device='cuda')
print(f'Synthetic input (B,C,T,N): {tuple(x.shape)}')

model_args = dict(cfg['model_args'])
mod_path, cls_name = cfg['model'].rsplit('.', 1)
Model = getattr(importlib.import_module(mod_path), cls_name)
model = Model(**model_args).cuda().eval()
print(f'Model: {type(model).__name__}, buffer_size in BDeltaBlock = {model_args["bdn_buffer_size"]}')

with torch.no_grad():
    try:
        out = model(x)
    except Exception as e:
        # Some Motion models return additional outputs and may need a label arg.
        # Fall back to invoking just up through the temporal encoder.
        print(f'Full forward failed ({e!r}); will trace partial forward.')
        # Manually run the spatial pipeline up to .mamba
        try:
            from models.motion import Motion
            # We rely on the inherited forward up to .mamba being a method named the same
            # Easier path: monkey-trace by calling each component manually using captured
            # intermediate. For a probe, just attempt mamba on a guessed shape.
            hidden = getattr(model.mamba, 'in_channels', 128)
            # Standard PMamba: temporal encoder sees (B, hidden, T, N) after spatial.
            fake = torch.randn(B, hidden, T_in, N_in, device='cuda')
            _ = model.mamba(fake)
        except Exception as e2:
            print(f'Partial forward also failed: {e2!r}')

print()
print('=== BDN block ejection report ===')
print(f'Total BDeltaBlock.forward calls : {len(_records)}')
if _records:
    Ts  = [r[0] for r in _records]
    Ws  = [r[1] for r in _records]
    ejs = [r[2] for r in _records]
    print(f'T (seq len seen by block)       : min={min(Ts)} max={max(Ts)} unique={sorted(set(Ts))}')
    print(f'W (buffer size)                 : {sorted(set(Ws))}')
    print(f'Ejections per call              : min={min(ejs)} max={max(ejs)} mean={sum(ejs)/len(ejs):.1f}')
    n_zero = sum(1 for e in ejs if e == 0)
    print(f'Calls with ZERO ejections       : {n_zero}/{len(ejs)} ({100*n_zero/len(ejs):.0f}%)')
    print()
    if n_zero == len(ejs):
        print('VERDICT: buffer NEVER overflows. BDN degenerates to pure attention over T<=W tokens.')
        print('         Long-term delta state is UNUSED -> hybrid mechanism INACTIVE.')
    elif n_zero > 0:
        print('VERDICT: buffer overflows sometimes. Mixed regime; hybrid partly active.')
    else:
        print(f'VERDICT: buffer always overflows ({sum(ejs)//len(ejs)} ejections/call avg). Hybrid ACTIVE.')
