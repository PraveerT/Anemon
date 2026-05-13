"""Learn per-model fusion weights and temperatures on TRAIN softmaxes,
apply on TEST. Honest because no test info used for tuning."""
import os, itertools, numpy as np
from scipy.optimize import minimize

os.chdir('/notebooks/PMamba/experiments')
D = 'dump_probs_runs'

def load_dsn(p, t=9.5):
    z = np.load(p); L = z['logits']*t; L=L-L.max(1,keepdims=True); e=np.exp(L); return e/e.sum(1,keepdims=True), z['labels']
def load_p(p):
    z = np.load(p); return z['probs'], z['labels']

# Load test and train sets per model
test = {}; train = {}; lt = None; ltr = None
for name, (test_p, train_p, is_dsn) in {
    'DSN':   ('cvpr_dsn_K_depth.npz', 'cvpr_dsn_K_depth_TRAIN.npz', True),
    'RD':    ('realdeltanet_ep118.npz', 'realdeltanet_ep118_TRAIN.npz', False),
    'BRD':   ('brd_ep112.npz', 'brd_ep112_TRAIN.npz', False),
    'AttRD': ('attrd_ep120.npz', 'attrd_ep120_TRAIN.npz', False),
    'RD(N2)':('realdeltanet_n2_ep118.npz', 'realdeltanet_n2_ep118_TRAIN.npz', False),
}.items():
    if is_dsn:
        test[name], lt_ = load_dsn(f'{D}/{test_p}')
        train[name], ltr_ = load_dsn(f'{D}/{train_p}')
    else:
        test[name], lt_ = load_p(f'{D}/{test_p}')
        train[name], ltr_ = load_p(f'{D}/{train_p}')
    if lt is None: lt = lt_
    if ltr is None: ltr = ltr_

print('train sizes:', {n: train[n].shape for n in train})
print('test sizes:',  {n: test[n].shape for n in test})

def acc(p, labels): return (p.argmax(1) == labels).mean() * 100

def apply_temperature(p, T, eps=1e-9):
    log_p = np.log(p + eps) / T
    e = np.exp(log_p - log_p.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def fuse_weighted(probs_list, weights):
    w = np.array(weights); w = w / w.sum()
    return sum(w_i * p for w_i, p in zip(w, probs_list))

def fuse_temp_weighted(probs_list, weights, temps):
    w = np.array(weights); w = w / w.sum()
    return sum(w_i * apply_temperature(p, t) for w_i, p, t in zip(w, probs_list, temps))

names_pool = list(test.keys())

# 1) Baseline uniform 1/K on train-best DSN+BRD+AttRD trio
combo = ['DSN', 'BRD', 'AttRD']
train_ps = [train[n] for n in combo]
test_ps  = [test[n] for n in combo]
print(f"\n=== {' + '.join(combo)} (uniform) ===")
print(f"  train acc = {acc(fuse_weighted(train_ps, [1]*len(combo)), ltr):.2f}%")
print(f"  test acc  = {acc(fuse_weighted(test_ps,  [1]*len(combo)), lt):.2f}%")

# 2) Optimize weights on TRAIN, apply on TEST
print(f"\n=== Weight optimization on train ===")
def neg_train_acc_w(w, probs):
    w = np.exp(w); w = w / w.sum()
    p = sum(w_i * pi for w_i, pi in zip(w, probs))
    return -((p.argmax(1) == ltr).mean())

for combo in [['DSN','BRD','AttRD'], ['DSN','RD','BRD','AttRD'], ['DSN','RD','BRD','AttRD','RD(N2)']]:
    train_ps = [train[n] for n in combo]
    test_ps  = [test[n] for n in combo]
    w0 = np.zeros(len(combo))
    # Direct grid over simplex (rough)
    best_train_acc = -1; best_w = None
    grid = np.linspace(0.1, 3.0, 15)
    for ws in itertools.product(grid, repeat=len(combo)):
        w = np.array(ws); w = w / w.sum()
        p = sum(wi * pi for wi, pi in zip(w, train_ps))
        a = (p.argmax(1) == ltr).mean()
        if a > best_train_acc:
            best_train_acc = a; best_w = w
    p_test = sum(wi * pi for wi, pi in zip(best_w, test_ps))
    print(f"  {' + '.join(combo):40s}  best_w={best_w.round(3).tolist()}  train={best_train_acc*100:.2f}  test={(p_test.argmax(1)==lt).mean()*100:.2f}")

# 3) Per-model temperature on train (set per-model T to align uncertainty)
print(f"\n=== Temperature optimization on train ===")
def best_temp_uniform(combo, T_grid=np.linspace(0.3, 3.0, 14)):
    train_ps = [train[n] for n in combo]
    test_ps  = [test[n] for n in combo]
    best_train_acc = -1; best_T = None
    for Ts in itertools.product(T_grid, repeat=len(combo)):
        tps = [apply_temperature(p, t) for p, t in zip(train_ps, Ts)]
        avg = np.mean(tps, axis=0)
        a = (avg.argmax(1) == ltr).mean()
        if a > best_train_acc:
            best_train_acc = a; best_T = Ts
    tps_test = [apply_temperature(p, t) for p, t in zip(test_ps, best_T)]
    test_acc = (np.mean(tps_test, axis=0).argmax(1) == lt).mean() * 100
    return best_T, best_train_acc * 100, test_acc

for combo in [['DSN','BRD','AttRD'], ['DSN','RD','BRD','AttRD']]:
    if len(combo) > 3:
        # Coarser grid for 4 models
        T_grid = np.linspace(0.5, 2.0, 6)
    else:
        T_grid = np.linspace(0.3, 3.0, 12)
    T_best, ta, te = best_temp_uniform(combo, T_grid)
    print(f"  {' + '.join(combo):30s}  T={[f'{t:.2f}' for t in T_best]}  train={ta:.2f}  test={te:.2f}")
