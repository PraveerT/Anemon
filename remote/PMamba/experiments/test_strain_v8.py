"""V8: per-point strain tensor field as 6-ch input feature.

For each pair (t, t+1) and each sampled point i with k=8 nearest neighbours
(in xyz space at frame t):
  - delta_src_j = p_j^t   - p_i^t       (j in N(i))
  - delta_tgt_j = p_j^{t+1} - p_i^{t+1}
  - Local least-squares: F_i = arg min sum ||delta_tgt - F * delta_src||^2
                            = (sum delta_tgt @ delta_src^T) (sum delta_src @ delta_src^T)^{-1}
  - Green-Lagrange strain: eps_i = (F_i^T F_i - I) / 2  (3x3 symmetric)
  - Store 6 unique entries: eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz

Per-point 6-ch strain tensor concatenated to Cfbq -> 16ch input
[xyz(3), rf(3), rb(3), strain(6), t(1)] = 16ch
"""
import sys, os
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, requests
import nvidia_dataloader

torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)


TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"
def tg(msg):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            chat_id = r["result"][-1]["message"]["chat"]["id"]
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception: pass


def rot_to_quat(R):
    m00,m01,m02 = R[0,0], R[0,1], R[0,2]
    m10,m11,m12 = R[1,0], R[1,1], R[1,2]
    m20,m21,m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = torch.sqrt(tr.clamp(min=-0.999) + 1) * 2
        q = torch.stack([0.25*s, (m21-m12)/s, (m02-m20)/s, (m10-m01)/s])
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1 + m00 - m11 - m22).clamp(min=1e-8) * 2
        q = torch.stack([(m21-m12)/s, 0.25*s, (m01+m10)/s, (m02+m20)/s])
    elif m11 > m22:
        s = torch.sqrt(1 + m11 - m00 - m22).clamp(min=1e-8) * 2
        q = torch.stack([(m02-m20)/s, (m01+m10)/s, 0.25*s, (m12+m21)/s])
    else:
        s = torch.sqrt(1 + m22 - m00 - m11).clamp(min=1e-8) * 2
        q = torch.stack([(m10-m01)/s, (m02+m20)/s, (m12+m21)/s, 0.25*s])
    q = F.normalize(q, dim=-1)
    if q[0] < 0: q = -q
    return q


def hamilton(a, b):
    aw,ax,ay,az = a[...,0], a[...,1], a[...,2], a[...,3]
    bw,bx,by,bz = b[...,0], b[...,1], b[...,2], b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def quat_rotate_pts(q, points):
    N = points.shape[0]
    q_b = q.unsqueeze(0).expand(N, -1)
    pq = torch.cat([torch.zeros(N, 1, device=points.device, dtype=points.dtype), points], dim=-1)
    q_conj = torch.cat([q_b[:, 0:1], -q_b[:, 1:]], dim=-1)
    return hamilton(hamilton(q_b, pq), q_conj)[:, 1:]


def kabsch_quat_w(src, tgt, w_vec):
    w = w_vec.unsqueeze(0).clamp(min=0)
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src.unsqueeze(0) * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tgt.unsqueeze(0) * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src.unsqueeze(0) - sm; tc = tgt.unsqueeze(0) - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=src.device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    q = rot_to_quat(R[0])
    t = tm.squeeze(0).squeeze(0) - quat_rotate_pts(q, sm.squeeze(0).squeeze(0).unsqueeze(0))[0]
    return q, t


def corr_sample_indices(orig_flat_idx, corr_target, corr_weight, pts_size, F_, P):
    device = orig_flat_idx.device
    idx0 = torch.linspace(0, P-1, pts_size, device=device).long()
    sampled_idx = torch.zeros(F_, pts_size, dtype=torch.long, device=device)
    matched = torch.zeros(F_-1, pts_size, dtype=torch.bool, device=device)
    sampled_idx[0] = idx0
    current_prov = orig_flat_idx[0, idx0].long()
    total_pts = corr_target.shape[-1]; raw_ppf = total_pts // F_
    for t in range(F_ - 1):
        next_prov = orig_flat_idx[t+1].long()
        reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
        reverse_map[next_prov] = torch.arange(P, device=device)
        tgt_flat = corr_target[current_prov]
        tgt_w = corr_weight[current_prov]
        tgt_flat_safe = tgt_flat.clamp(min=0)
        tgt_frame = tgt_flat // raw_ppf
        tgt_pos = reverse_map[tgt_flat_safe]
        valid = (tgt_flat >= 0) & (tgt_w > 0) & (tgt_frame == t+1) & (tgt_pos >= 0)
        next_idx = torch.randint(0, P, (pts_size,), device=device)
        next_idx[valid] = tgt_pos[valid]
        sampled_idx[t+1] = next_idx; matched[t] = valid
        current_prov = orig_flat_idx[t+1, next_idx].long()
    return sampled_idx, matched


PTS = 256
K_NN_STRAIN = 8


def compute_strain_field(p_t, p_tp1, k=K_NN_STRAIN):
    """For each point in p_t (P,3), compute deformation gradient and strain tensor.
    Returns strain (P, 6) flattened symmetric tensor [xx, yy, zz, xy, xz, yz].
    """
    P = p_t.shape[0]
    # kNN at frame t
    d = torch.cdist(p_t, p_t)
    _, knn_idx = d.topk(k + 1, dim=-1, largest=False)
    knn_idx = knn_idx[:, 1:]  # drop self           # (P, k)
    # Neighbor positions
    nb_t = p_t[knn_idx]                              # (P, k, 3)
    nb_tp1 = p_tp1[knn_idx]                          # (P, k, 3)
    # Deltas (centred at the query point)
    dsrc = nb_t - p_t.unsqueeze(1)                   # (P, k, 3)
    dtgt = nb_tp1 - p_tp1.unsqueeze(1)               # (P, k, 3)
    # F_i = (sum dtgt @ dsrc^T) (sum dsrc @ dsrc^T)^{-1}
    A = torch.einsum('pki,pkj->pij', dtgt, dsrc)     # (P, 3, 3)
    B = torch.einsum('pki,pkj->pij', dsrc, dsrc)     # (P, 3, 3)
    # Tikhonov regularised inverse: relative regularisation by trace
    trace_B = B.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    reg = (1e-2 * trace_B / 3.0).clamp(min=1e-3)
    B_reg = B + reg * torch.eye(3, device=B.device).unsqueeze(0)
    F_grad = A @ torch.linalg.inv(B_reg)              # (P, 3, 3)
    # Green-Lagrange strain
    eps = 0.5 * (F_grad.transpose(-1, -2) @ F_grad
                 - torch.eye(3, device=F_grad.device).unsqueeze(0))
    # Reduce to 2 robust scalars per point:
    #   - Frobenius norm: total non-rigidity magnitude
    #   - Trace:           volume change (positive = stretch, negative = compress)
    eps_frob = eps.flatten(-2).norm(dim=-1, keepdim=True)            # (P, 1)
    eps_trace = eps.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) # (P, 1)
    # log1p compression to handle extreme values
    strain2 = torch.cat([
        torch.sign(eps_frob)  * torch.log1p(eps_frob.abs()),
        torch.sign(eps_trace) * torch.log1p(eps_trace.abs()),
    ], dim=-1)                                                       # (P, 2)
    # Final clamp for safety
    strain2 = strain2.clamp(min=-3.0, max=3.0)
    return strain2


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True)
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RF = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RB = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    STRAIN = np.zeros((N, 32, PTS, 2), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    for i in range(N):
        s = loader[i]; pts_d = s[0]; label = s[1]
        pts = pts_d['points'].cuda()
        F_, P, C = pts.shape
        xyz = pts[..., :3]
        orig = pts_d['orig_flat_idx'].cuda()
        ctgt = torch.from_numpy(pts_d['corr_full_target_idx']).long().cuda()
        cw = torch.from_numpy(pts_d['corr_full_weight']).float().cuda()
        sidx, matched = corr_sample_indices(orig, ctgt, cw, PTS, F_, P)
        xyz_s = torch.gather(xyz, 1, sidx.unsqueeze(-1).expand(-1, -1, 3))
        XYZ[i] = xyz_s.cpu().numpy()
        for t in range(F_ - 1):
            mask_t = matched[t].float()
            q_f, tr_f = kabsch_quat_w(xyz_s[t], xyz_s[t+1], mask_t)
            q_b, tr_b = kabsch_quat_w(xyz_s[t+1], xyz_s[t], mask_t)
            RF[i, t+1] = (xyz_s[t+1] - (quat_rotate_pts(q_f, xyz_s[t]) + tr_f)).cpu().numpy()
            RB[i, t]   = (xyz_s[t]   - (quat_rotate_pts(q_b, xyz_s[t+1]) + tr_b)).cpu().numpy()
            # Strain tensor field at frame t (relating t -> t+1)
            STRAIN[i, t] = compute_strain_field(xyz_s[t], xyz_s[t+1]).cpu().numpy()
        labels[i] = label
        if (i+1) % 200 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RF), torch.from_numpy(RB),
            torch.from_numpy(STRAIN), torch.from_numpy(labels))


print('=== Phase A: Cfbq + strain tensor field ===')
xyz_tr, rf_tr, rb_tr, str_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, str_te, y_te = collect('test')
print(f"Strain abs mean: {str_tr.abs().mean():.4f}  max: {str_tr.abs().max():.3f}")

T = 32
t_tr_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0], T, PTS, 1)
# Baseline 10ch and full 16ch (with strain)
C0_tr = torch.cat([xyz_tr, rf_tr, rb_tr, t_tr_ch], dim=-1)
C0_te = torch.cat([xyz_te, rf_te, rb_te, t_te_ch], dim=-1)
C8_tr = torch.cat([xyz_tr, rf_tr, rb_tr, str_tr, t_tr_ch], dim=-1)
C8_te = torch.cat([xyz_te, rf_te, rb_te, str_te, t_te_ch], dim=-1)


def knn_xyz(xyz, k):
    d = torch.cdist(xyz, xyz)
    _, idx = d.topk(k, dim=-1, largest=False)
    return idx


def gather_neighbors(x, idx):
    BT, P, C = x.shape
    _, _, k = idx.shape
    bi = torch.arange(BT, device=x.device).view(BT, 1, 1).expand(-1, P, k)
    return x[bi, idx]


class EdgeConv(nn.Module):
    def __init__(self, in_ch=10, out_ch=128, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(2*in_ch, 64, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())
    def forward(self, x):
        BT, P, C = x.shape
        xyz = x[..., :3].contiguous()
        idx = knn_xyz(xyz, self.k)
        x_j = gather_neighbors(x, idx)
        x_i = x.unsqueeze(2).expand(-1, -1, self.k, -1)
        edge = torch.cat([x_i, x_j - x_i], dim=-1).permute(0, 3, 1, 2)
        h = self.mlp(edge)
        return h.max(-1).values.transpose(1, 2)


class TinyKNN(nn.Module):
    def __init__(self, in_ch=10, num_classes=25, k=16):
        super().__init__()
        self.edge = EdgeConv(in_ch=in_ch, out_ch=128, k=k)
        self.pt_mlp = nn.Sequential(
            nn.Linear(128 + in_ch, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU())
        self.proj = nn.Conv1d(512, 256, 1)
        self.c1 = nn.Conv1d(256,256,3,padding=1); self.b1 = nn.BatchNorm1d(256)
        self.c2 = nn.Conv1d(256,256,3,padding=1); self.b2 = nn.BatchNorm1d(256)
        self.c3 = nn.Conv1d(256,256,3,padding=1); self.b3 = nn.BatchNorm1d(256)
        self.c4 = nn.Conv1d(256,256,3,padding=1); self.b4 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(128, num_classes))
    def forward(self, x):
        B, T_, P_, C_ = x.shape
        x_bt = x.reshape(B*T_, P_, C_)
        local = self.edge(x_bt)
        h = torch.cat([local, x_bt], dim=-1)
        h = self.pt_mlp(h).reshape(B, T_, P_, -1)
        h = torch.cat([h.max(2).values, h.mean(2)], dim=-1).transpose(1, 2)
        h = self.proj(h)
        h = F.gelu(self.b1(self.c1(h)))
        h = h + F.gelu(self.b2(self.c2(h)))
        h = h + F.gelu(self.b3(self.c3(h)))
        h = h + F.gelu(self.b4(self.c4(h)))
        h = self.drop(h).max(-1).values
        return self.head(h)


SAVE_ROOT = '/notebooks/PMamba/experiments/work_dir/qcc_branch/v8'
os.makedirs(SAVE_ROOT, exist_ok=True)


def train(label, X_tr, X_te, in_ch, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = TinyKNN(in_ch=in_ch).cuda()
    print(f"[{label}] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/5 if ep<5
        else 0.5*(1+math.cos(math.pi*(ep-5)/max(1,120-5))))
    Xtr = X_tr.cuda(); ytr_c = y_tr.cuda(); Xte = X_te.cuda(); yte_c = y_te.cuda()
    best = 0.0; best_logits = None; best_state = None; best_ep = -1
    for ep in range(120):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(X_tr), generator=g)
        for i in range(0, len(X_tr), 16):
            idx = perm[i:i+16]
            xb = Xtr[idx]; yb = ytr_c[idx]
            B_, T_, P_, _ = xb.shape
            xb = xb * (torch.rand(B_, 1, P_, 1, device=xb.device) > 0.10).float()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(X_te), 32):
                lg.append(model(Xte[i:i+32]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == yte_c).float().mean().item()
        if acc > best:
            best = acc; best_ep = ep; best_logits = lg.clone()
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 119:
            print(f"  [{label}] ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})")
    return best, best_ep, best_logits, best_state


tg("V8 strain tensor test starting (V0 baseline + V8 strain).")

print('\n=== V0 baseline (Cfbq alone) ===')
v0, v0_ep, v0_lg, v0_st = train('V0', C0_tr, C0_te, in_ch=10, seed=1)
torch.save({'state_dict': v0_st, 'best_acc': v0, 'best_ep': v0_ep},
           f'{SAVE_ROOT}/v0_best.pt')
np.savez(f'{SAVE_ROOT}/v0_logits.npz', logits=v0_lg.numpy(), labels=y_te.numpy(),
         acc=v0, best_ep=v0_ep)

print('\n=== V8 Cfbq + strain scalars (12ch input) ===')
v8, v8_ep, v8_lg, v8_st = train('V8', C8_tr, C8_te, in_ch=12, seed=1)
torch.save({'state_dict': v8_st, 'best_acc': v8, 'best_ep': v8_ep},
           f'{SAVE_ROOT}/v8_best.pt')
np.savez(f'{SAVE_ROOT}/v8_logits.npz', logits=v8_lg.numpy(), labels=y_te.numpy(),
         acc=v8, best_ep=v8_ep)

msg = f"""
=== V8 strain tensor as input feature (Phase A seed 0, training seed 1) ===
V0 (Cfbq alone, 10ch):                {v0*100:.2f}%
V8 (Cfbq + strain scalars, 12ch):     {v8*100:.2f}%  ({(v8-v0)*100:+.2f}pp)
"""
print(msg); tg(msg)
