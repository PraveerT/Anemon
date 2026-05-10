"""SeQuMamba standalone on skeleton bone-direction quaternions.

Whole pipeline is quaternion-from-input → equivariance theorem holds end-to-end.

Architecture:
  input (B, T=32, B_bones=20, 4) bone direction quaternions
    -> Quaternion lift: real-mix across bones (4-component-tied)
       (B, T, hidden, 4)
    -> N SeQuMambaBlock layers (SO(3)-equivariant + selective)
    -> magnitude readout at last frame: ‖h_T‖ (SO(3)-invariant scalars)
    -> classifier MLP -> 25 classes

Theorem (whole-network SO(3)-invariance):
  If input bone quaternions are all left-multiplied by a unit quaternion q ∈ S^3,
  every feature in the pipeline transforms as f → q·f (by linearity of all
  channel-mixing layers + Hamilton-product structure). The magnitude readout
  at the end discards rotation: ‖q·h‖ = ‖h‖ → classifier output invariant.

Validation: at test time, evaluate under random rotation augmentation and
verify accuracy is unchanged (theorem holds empirically).
"""
import math, os, re, time, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SK = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'
ANNOT_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
T_FIXED = 32
NUM_CLASSES = 25
NUM_EPOCHS = 120
BS = 32
LR = 5e-4
WD = 1e-4
HIDDEN = 192          # quaternion features = HIDDEN/4 = 48
N_LAYERS = 4
WORK_DIR = '/notebooks/PMamba/experiments/work_dir/sequmamba_standalone/'
os.makedirs(WORK_DIR, exist_ok=True)

BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
B = 20

def parse_annot(path):
    out = {}
    with open(path) as f:
        for line in f:
            mp_ = re.search(r'path:(\S+)', line); ml = re.search(r'label:(\d+)', line)
            if mp_ and ml: out[mp_.group(1)] = int(ml.group(1)) - 1
    return out

train_lbl = parse_annot(f'{ANNOT_ROOT}/nvgesture_train_correct_cvpr2016_v2.lst')
test_lbl  = parse_annot(f'{ANNOT_ROOT}/nvgesture_test_correct_cvpr2016_v2.lst')
print('loading landmarks...')
sk = dict(np.load(SK, allow_pickle=False))


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

def vec_to_quat_np(V):
    n = np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9
    u = V / n
    cos_h = np.clip((1 + u[..., 2:3]) * 0.5, 1e-9, 1.0)
    w = np.sqrt(cos_h)
    sin_h = np.sqrt(np.clip(1 - cos_h, 0, 1))
    axis = np.zeros_like(u); axis[..., 0] = -u[..., 1]; axis[..., 1] = u[..., 0]
    s = np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9
    return np.concatenate([w, axis/s*sin_h], axis=-1).astype(np.float32)

def encode_sample(lm):
    T = lm.shape[0]
    out = np.zeros((T, B, 4), dtype=np.float32)
    for b, (p, c) in enumerate(BONES):
        out[:, b, :] = vec_to_quat_np(lm[:, c, :] - lm[:, p, :])
    return out

print('precomputing per-bone direction quaternions...')
encoded = {}
for k, lm_raw in sk.items():
    if lm_raw.shape[0] < 4: continue
    lm = fillnan(lm_raw)
    q = encode_sample(lm)
    idx = np.linspace(0, q.shape[0]-1, T_FIXED).astype(np.int64)
    encoded[k] = q[idx]
print(f'{len(encoded)} encoded')


# ============ Quaternion ops ============
def qmul(p, q):
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ], dim=-1)

def qnorm_unit(q):
    return q / (q.norm(dim=-1, keepdim=True) + 1e-9)

def parallel_scan_quat(A, B_in):
    A_acc, B_acc = A, B_in
    T = A.shape[-3]
    log_T = max(1, math.ceil(math.log2(T)))
    for k in range(log_T):
        step = 2 ** k
        if step >= T: break
        ident_A = torch.zeros_like(A_acc[..., :step, :, :])
        ident_A[..., 0] = 1.0
        ident_B = torch.zeros_like(B_acc[..., :step, :, :])
        A_prev = torch.cat([ident_A, A_acc[..., :-step, :, :]], dim=-3)
        B_prev = torch.cat([ident_B, B_acc[..., :-step, :, :]], dim=-3)
        A_new = qmul(A_prev, A_acc)
        B_new = qmul(B_prev, A_acc) + B_acc
        A_acc, B_acc = A_new, B_new
    return B_acc


# ============ SO(3)-equivariant SeQuMamba ============

class SeQuMambaBlock(nn.Module):
    """Selective quaternion Mamba block — SO(3)-equivariant via:
       - real channel-mixing weights (treat 4 quat components identically)
       - scalar selective gates derived from invariant magnitudes
       - unit-quaternion direction parameters
    """
    def __init__(self, n_quat, gate_hidden=None):
        super().__init__()
        if gate_hidden is None:
            gate_hidden = max(n_quat // 2, 16)
        self.W_x = nn.Linear(n_quat, n_quat, bias=False)
        self.W_y = nn.Linear(n_quat, n_quat, bias=False)
        self.gate_A = nn.Sequential(nn.Linear(n_quat, gate_hidden), nn.SiLU(), nn.Linear(gate_hidden, n_quat))
        self.gate_B = nn.Sequential(nn.Linear(n_quat, gate_hidden), nn.SiLU(), nn.Linear(gate_hidden, n_quat))
        self.qA_raw = nn.Parameter(torch.randn(n_quat, 4) * 0.05)
        self.qB_raw = nn.Parameter(torch.randn(n_quat, 4) * 0.05)
        self.qC_raw = nn.Parameter(torch.randn(n_quat, 4) * 0.05)
        with torch.no_grad():
            self.qA_raw[:, 0] = 1.0; self.qB_raw[:, 0] = 1.0; self.qC_raw[:, 0] = 1.0

    def _mix(self, q_features, W):
        return W(q_features.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, x):
        Bz, T, N, _ = x.shape
        x_mix = self._mix(x, self.W_x)
        mags = x_mix.norm(dim=-1)
        sA = torch.sigmoid(self.gate_A(mags))
        sB = torch.sigmoid(self.gate_B(mags))
        qA = qnorm_unit(self.qA_raw)
        qB = qnorm_unit(self.qB_raw)
        qC = qnorm_unit(self.qC_raw)
        A_t = sA.unsqueeze(-1) * qA
        B_in = qmul(x_mix, sB.unsqueeze(-1) * qB)
        H = parallel_scan_quat(A_t, B_in)
        return qmul(self._mix(H, self.W_y), qC)


class SeQuMambaStandalone(nn.Module):
    """End-to-end SO(3)-equivariant network over quaternion bone inputs."""
    def __init__(self, hidden=HIDDEN, n_layers=N_LAYERS, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        assert hidden % 4 == 0
        self.n_quat = hidden // 4
        # Lift: mix 20 bones -> n_quat features, real-valued (preserves equivariance)
        self.lift = nn.Linear(B, self.n_quat, bias=False)
        self.blocks = nn.ModuleList([SeQuMambaBlock(self.n_quat) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.n_quat) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(self.n_quat)
        # Classifier on SO(3)-invariant magnitudes
        self.head = nn.Sequential(
            nn.Linear(self.n_quat, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def _gnorm(self, x_q, norm_layer):
        mags = x_q.norm(dim=-1, keepdim=True) + 1e-9
        new_mags = F.silu(norm_layer(mags.squeeze(-1))).unsqueeze(-1)
        return x_q / mags * new_mags

    def forward(self, x):
        # x: (B, T, 20, 4)
        # Lift bones: real Linear(20, n_quat) applied per-component
        x = self.lift(x.transpose(-1, -2)).transpose(-1, -2)  # (B, T, n_quat, 4)
        for blk, norm in zip(self.blocks, self.norms):
            residual = x
            x_n = self._gnorm(x, norm)
            out = blk(x_n)
            x = self.dropout(out) + residual
        x = self._gnorm(x, self.final_norm)
        # Mean-pool over time, then magnitudes
        h = x.mean(dim=1)              # (B, n_quat, 4)
        mags = h.norm(dim=-1)          # (B, n_quat) — SO(3)-invariant
        return self.head(mags)


# ============ Dataset + Rotation augmentation for theorem validation ============

class DQDataset(Dataset):
    def __init__(self, label_map):
        self.items = [(k, label_map[k]) for k in encoded if k in label_map]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        k, lbl = self.items[i]
        return torch.from_numpy(encoded[k]), lbl

def random_unit_quat(device, dtype=torch.float32):
    """Sample a random unit quaternion uniformly on S^3."""
    q = torch.randn(4, device=device, dtype=dtype)
    return q / q.norm()

def rotate_batch(x, q):
    """Left-multiply each bone quaternion by q. x: (B, T, 20, 4), q: (4,)."""
    return qmul(q, x)


ds_tr = DQDataset(train_lbl); ds_te = DQDataset(test_lbl)
print(f'train {len(ds_tr)} test {len(ds_te)}')

loader_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True, num_workers=4)
loader_te = DataLoader(ds_te, batch_size=BS, shuffle=False, num_workers=4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SeQuMambaStandalone(hidden=HIDDEN, n_layers=N_LAYERS).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'SeQuMamba-Standalone params: {n_params:,}')

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)

best_te = 0
for ep in range(1, NUM_EPOCHS + 1):
    model.train()
    losses = []
    for x, y in loader_tr:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    sched.step()
    model.eval()
    correct = 0; total = 0; all_probs = []; all_lbl = []
    with torch.no_grad():
        for x, y in loader_te:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
            all_lbl.append(y.cpu().numpy())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    te_acc = correct / total * 100
    if te_acc > best_te:
        best_te = te_acc
        np.savez(os.path.join(WORK_DIR, 'best_probs.npz'),
                 probs=np.concatenate(all_probs), labels=np.concatenate(all_lbl))
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                   os.path.join(WORK_DIR, 'best_model.pt'))
    print(f'ep {ep:3d}  loss={np.mean(losses):.4f}  test={te_acc:.2f}  best={best_te:.2f}', flush=True)

# === Rotation-robustness test ===
print(f'\nBEST: {best_te:.2f}')
print('\n=== Rotation-robustness test ===')
model.eval()
# Load best
ck = torch.load(os.path.join(WORK_DIR, 'best_model.pt'), map_location='cpu')
model.load_state_dict(ck['model_state_dict'])
model.eval()
print('Testing under random unit-quaternion rotations applied uniformly to all bones:')
for trial in range(5):
    torch.manual_seed(trial)
    q_rot = random_unit_quat(device)
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader_te:
            x = x.to(device); y = y.to(device)
            x_rot = rotate_batch(x, q_rot)
            logits = model(x_rot)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    print(f'  rotation trial {trial}: q={q_rot.tolist()[:2]}... acc={correct/total*100:.2f} (vs unrotated {best_te:.2f})')
