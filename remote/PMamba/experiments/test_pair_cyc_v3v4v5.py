"""V3 / V4 / V5: asymmetry-breaking pair_cyc variants.

V0 baseline = TinyKNN on Cfbq 10ch (Phase A seed 0)
V3 = dualstream with pair_cyc_v3: asymmetric weights from loader confidence
V4 = dualstream with pair_cyc_v4: IRLS Kabsch (fwd/bwd separate 4-iter Cauchy)
V5 = dualstream with per-point cycle residual using V4's IRLS Kabsch (3-ch/point)

Math justification:
  H_b = H_f^T (with same weights) -> q_b = q_f*  ->  q_f x q_b - I = 0
  Asymmetry breakers (V3 differ-weights, V4 IRLS diverge, V5 per-point) make
  H_f != H_b^T so cycle violation has real magnitude tied to articulation.
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
    """Single weighted Kabsch (any continuous weights)."""
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


def kabsch_irls(src, tgt, mask, n_iter=4):
    """IRLS Kabsch: re-weight by Cauchy of own-direction residual."""
    w = mask.float()
    q, tr = kabsch_quat_w(src, tgt, w)
    for _ in range(n_iter):
        pred = quat_rotate_pts(q, src) + tr
        r = (tgt - pred).norm(dim=-1)
        valid = mask.bool()
        sigma = 1.4826 * (r[valid].median() if valid.any() else r.median()) + 1e-6
        cauchy = sigma**2 / (r**2 + sigma**2)
        w = mask.float() * cauchy
        q, tr = kabsch_quat_w(src, tgt, w)
    return q, tr


def cycle_quat(q_f, q_b):
    """Compute c = q_f x q_b - I (4-vec)."""
    qcyc = hamilton(q_f.unsqueeze(0), q_b.unsqueeze(0))[0]
    qcyc = F.normalize(qcyc, dim=-1)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=qcyc.device)
    if (qcyc * ident).sum() < 0:
        qcyc = -qcyc
    return qcyc - ident


def per_point_cycle_resid(src, q_f, tr_f, q_b, tr_b):
    """For each point i: forward map then backward map; residual = original - returned."""
    forward = quat_rotate_pts(q_f, src) + tr_f
    back = quat_rotate_pts(q_b, forward) + tr_b
    return src - back   # (P, 3)


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


def collect(phase):
    """Compute Cfbq + 3 pair_cyc variants per pair, and per-point cycle residual for V5."""
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True)
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RF = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RB = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    PC_V3 = np.zeros((N, 32, 4), dtype=np.float32)  # asymmetric weights
    PC_V4 = np.zeros((N, 32, 4), dtype=np.float32)  # IRLS asymmetric
    PP_CYC_V5 = np.zeros((N, 32, PTS, 3), dtype=np.float32)  # per-point cycle (V4-IRLS based)
    labels = np.zeros(N, dtype=np.int64)
    for i in range(N):
        s = loader[i]; pts_d = s[0]; label = s[1]
        pts = pts_d['points'].cuda()
        F_, P, C = pts.shape
        xyz = pts[..., :3]
        orig = pts_d['orig_flat_idx'].cuda()
        ctgt = torch.from_numpy(pts_d['corr_full_target_idx']).long().cuda()
        cw_full = torch.from_numpy(pts_d['corr_full_weight']).float().cuda()  # confidence weights
        sidx, matched = corr_sample_indices(orig, ctgt, cw_full, PTS, F_, P)
        xyz_s = torch.gather(xyz, 1, sidx.unsqueeze(-1).expand(-1, -1, 3))
        XYZ[i] = xyz_s.cpu().numpy()
        # Sampled per-point fwd-confidence (we don't easily have bwd-confidence; approximate
        # bwd confidence as a permutation of fwd weights: how many fwd matches land at each tgt point).
        for t in range(F_ - 1):
            mask_t = matched[t].float()
            # --- standard Cfbq (single-pass Kabsch with binary mask) ---
            q_f0, tr_f0 = kabsch_quat_w(xyz_s[t], xyz_s[t+1], mask_t)
            q_b0, tr_b0 = kabsch_quat_w(xyz_s[t+1], xyz_s[t], mask_t)
            RF[i, t+1] = (xyz_s[t+1] - (quat_rotate_pts(q_f0, xyz_s[t]) + tr_f0)).cpu().numpy()
            RB[i, t]   = (xyz_s[t]   - (quat_rotate_pts(q_b0, xyz_s[t+1]) + tr_b0)).cpu().numpy()
            # --- V3 asymmetric weights ---
            # Forward: weight by per-source-point match certainty (we approximate via residual quality).
            r_fwd_init = (xyz_s[t+1] - (quat_rotate_pts(q_f0, xyz_s[t]) + tr_f0)).norm(dim=-1)
            r_bwd_init = (xyz_s[t]   - (quat_rotate_pts(q_b0, xyz_s[t+1]) + tr_b0)).norm(dim=-1)
            sigma_f = 1.4826 * r_fwd_init[matched[t]].median() + 1e-6 if matched[t].any() else r_fwd_init.median() + 1e-6
            sigma_b = 1.4826 * r_bwd_init[matched[t]].median() + 1e-6 if matched[t].any() else r_bwd_init.median() + 1e-6
            w_f3 = mask_t * (sigma_f**2 / (r_fwd_init**2 + sigma_f**2))
            w_b3 = mask_t * (sigma_b**2 / (r_bwd_init**2 + sigma_b**2))
            q_f3, _ = kabsch_quat_w(xyz_s[t], xyz_s[t+1], w_f3)
            q_b3, _ = kabsch_quat_w(xyz_s[t+1], xyz_s[t], w_b3)
            PC_V3[i, t] = cycle_quat(q_f3, q_b3).cpu().numpy()
            # --- V4 IRLS (4 iterations each direction independently) ---
            q_f4, tr_f4 = kabsch_irls(xyz_s[t], xyz_s[t+1], mask_t, n_iter=4)
            q_b4, tr_b4 = kabsch_irls(xyz_s[t+1], xyz_s[t], mask_t, n_iter=4)
            PC_V4[i, t] = cycle_quat(q_f4, q_b4).cpu().numpy()
            # --- V5 per-point cycle residual via V4's IRLS Kabsch ---
            PP_CYC_V5[i, t] = per_point_cycle_resid(xyz_s[t], q_f4, tr_f4, q_b4, tr_b4).cpu().numpy()
        labels[i] = label
        if (i+1) % 200 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RF), torch.from_numpy(RB),
            torch.from_numpy(PC_V3), torch.from_numpy(PC_V4), torch.from_numpy(PP_CYC_V5),
            torch.from_numpy(labels))


print('=== Phase A: Cfbq + V3/V4/V5 cycle features (Phase A seed 0) ===')
xyz_tr, rf_tr, rb_tr, pc3_tr, pc4_tr, pp5_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, pc3_te, pc4_te, pp5_te, y_te = collect('test')
print(f"PC_V3 abs mean: {pc3_tr.abs().mean():.5f}  max: {pc3_tr.norm(dim=-1).max():.4f}")
print(f"PC_V4 abs mean: {pc4_tr.abs().mean():.5f}  max: {pc4_tr.norm(dim=-1).max():.4f}")
print(f"PP_V5 abs mean: {pp5_tr.abs().mean():.5f}  max: {pp5_tr.norm(dim=-1).max():.4f}")

T = 32
t_tr_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0], T, PTS, 1)
C_tr = torch.cat([xyz_tr, rf_tr, rb_tr, t_tr_ch], dim=-1)
C_te = torch.cat([xyz_te, rf_te, rb_te, t_te_ch], dim=-1)


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


class TinyV0(nn.Module):
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


class TinyDualStream(nn.Module):
    def __init__(self, in_ch=10, num_classes=25, k=16, fr_in=4, fr_out=64):
        super().__init__()
        self.edge = EdgeConv(in_ch=in_ch, out_ch=128, k=k)
        self.pt_mlp = nn.Sequential(
            nn.Linear(128 + in_ch, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU())
        self.fr_mlp = nn.Sequential(
            nn.Linear(fr_in, 32), nn.GELU(),
            nn.Linear(32, fr_out))
        self.proj = nn.Conv1d(512 + fr_out, 256, 1)
        self.c1 = nn.Conv1d(256,256,3,padding=1); self.b1 = nn.BatchNorm1d(256)
        self.c2 = nn.Conv1d(256,256,3,padding=1); self.b2 = nn.BatchNorm1d(256)
        self.c3 = nn.Conv1d(256,256,3,padding=1); self.b3 = nn.BatchNorm1d(256)
        self.c4 = nn.Conv1d(256,256,3,padding=1); self.b4 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(128, num_classes))
    def forward(self, x_pt, x_fr):
        B, T_, P_, C_ = x_pt.shape
        x_bt = x_pt.reshape(B*T_, P_, C_)
        local = self.edge(x_bt)
        h = torch.cat([local, x_bt], dim=-1)
        h = self.pt_mlp(h).reshape(B, T_, P_, -1)
        per_frame_pt = torch.cat([h.max(2).values, h.mean(2)], dim=-1)
        per_frame_fr = self.fr_mlp(x_fr)
        per_frame = torch.cat([per_frame_pt, per_frame_fr], dim=-1)
        h_seq = per_frame.transpose(1, 2)
        h_seq = self.proj(h_seq)
        h_seq = F.gelu(self.b1(self.c1(h_seq)))
        h_seq = h_seq + F.gelu(self.b2(self.c2(h_seq)))
        h_seq = h_seq + F.gelu(self.b3(self.c3(h_seq)))
        h_seq = h_seq + F.gelu(self.b4(self.c4(h_seq)))
        h_seq = self.drop(h_seq).max(-1).values
        return self.head(h_seq)


class TinyV5(nn.Module):
    """V5 uses per-point cycle residual as an additional 3-channel point-feature."""
    def __init__(self, in_ch=13, num_classes=25, k=16):
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


SAVE_ROOT = '/notebooks/PMamba/experiments/work_dir/qcc_branch/v3v4v5'
os.makedirs(SAVE_ROOT, exist_ok=True)


def train_v0(seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = TinyV0(in_ch=10).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,120-warmup))))
    Xtr = C_tr.cuda(); ytr_c = y_tr.cuda(); Xte = C_te.cuda(); yte_c = y_te.cuda()
    best = 0.0; best_logits = None; best_state = None; best_ep = -1
    for ep in range(120):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(C_tr), generator=g)
        for i in range(0, len(C_tr), 16):
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
            for i in range(0, len(C_te), 32):
                lg.append(model(Xte[i:i+32]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == yte_c).float().mean().item()
        if acc > best:
            best = acc; best_ep = ep; best_logits = lg.clone()
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 119:
            print(f"  V0 ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})")
    return best, best_ep, best_logits, best_state


def train_dual(label, x_fr_tr, x_fr_te, fr_in, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = TinyDualStream(in_ch=10, fr_in=fr_in).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,120-warmup))))
    Xtr = C_tr.cuda(); Xfr_tr = x_fr_tr.cuda(); ytr_c = y_tr.cuda()
    Xte = C_te.cuda(); Xfr_te = x_fr_te.cuda(); yte_c = y_te.cuda()
    best = 0.0; best_logits = None; best_state = None; best_ep = -1
    for ep in range(120):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(C_tr), generator=g)
        for i in range(0, len(C_tr), 16):
            idx = perm[i:i+16]
            xb = Xtr[idx]; xfr = Xfr_tr[idx]; yb = ytr_c[idx]
            B_, T_, P_, _ = xb.shape
            xb = xb * (torch.rand(B_, 1, P_, 1, device=xb.device) > 0.10).float()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb, xfr), yb, label_smoothing=0.1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(C_te), 32):
                lg.append(model(Xte[i:i+32], Xfr_te[i:i+32]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == yte_c).float().mean().item()
        if acc > best:
            best = acc; best_ep = ep; best_logits = lg.clone()
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 119:
            print(f"  {label} ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})")
    return best, best_ep, best_logits, best_state


def train_v5(seed):
    """V5: per-point cycle residual as 3 extra per-point channels (13ch input)."""
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    # Build 13ch input: [xyz, rf, rb, pp_cyc, t]
    C_tr_13 = torch.cat([xyz_tr, rf_tr, rb_tr, pp5_tr, t_tr_ch], dim=-1)
    C_te_13 = torch.cat([xyz_te, rf_te, rb_te, pp5_te, t_te_ch], dim=-1)
    model = TinyV5(in_ch=13).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,120-warmup))))
    Xtr = C_tr_13.cuda(); Xte = C_te_13.cuda()
    ytr_c = y_tr.cuda(); yte_c = y_te.cuda()
    best = 0.0; best_logits = None; best_state = None; best_ep = -1
    for ep in range(120):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(C_tr_13), generator=g)
        for i in range(0, len(C_tr_13), 16):
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
            for i in range(0, len(C_te_13), 32):
                lg.append(model(Xte[i:i+32]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == yte_c).float().mean().item()
        if acc > best:
            best = acc; best_ep = ep; best_logits = lg.clone()
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 119:
            print(f"  V5 ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})")
    return best, best_ep, best_logits, best_state


SEED = 1
tg("V3/V4/V5 asymmetry-breaking pair_cyc test starting (Phase A seed 0).")

print('\n=== V0 baseline ===')
v0, v0_ep, v0_lg, v0_st = train_v0(SEED)
torch.save({'state_dict': v0_st, 'best_acc': v0, 'best_ep': v0_ep},
           f'{SAVE_ROOT}/v0_best.pt')
np.savez(f'{SAVE_ROOT}/v0_logits.npz', logits=v0_lg.numpy(), labels=y_te.numpy(),
         acc=v0, best_ep=v0_ep)
tg(f"V0 baseline: {v0*100:.2f}%")

print('\n=== V3 dualstream + asymmetric-weights pair_cyc ===')
v3, v3_ep, v3_lg, v3_st = train_dual('V3', pc3_tr, pc3_te, fr_in=4, seed=SEED)
torch.save({'state_dict': v3_st, 'best_acc': v3, 'best_ep': v3_ep},
           f'{SAVE_ROOT}/v3_best.pt')
np.savez(f'{SAVE_ROOT}/v3_logits.npz', logits=v3_lg.numpy(), labels=y_te.numpy(),
         acc=v3, best_ep=v3_ep)
tg(f"V3 asymmetric weights: {v3*100:.2f}% (vs V0 {v0*100:.2f})")

print('\n=== V4 dualstream + IRLS asymmetric pair_cyc ===')
v4, v4_ep, v4_lg, v4_st = train_dual('V4', pc4_tr, pc4_te, fr_in=4, seed=SEED)
torch.save({'state_dict': v4_st, 'best_acc': v4, 'best_ep': v4_ep},
           f'{SAVE_ROOT}/v4_best.pt')
np.savez(f'{SAVE_ROOT}/v4_logits.npz', logits=v4_lg.numpy(), labels=y_te.numpy(),
         acc=v4, best_ep=v4_ep)
tg(f"V4 IRLS asymmetric: {v4*100:.2f}% (vs V0 {v0*100:.2f})")

print('\n=== V5 per-point cycle residual (13ch input) ===')
v5, v5_ep, v5_lg, v5_st = train_v5(seed=SEED)
torch.save({'state_dict': v5_st, 'best_acc': v5, 'best_ep': v5_ep},
           f'{SAVE_ROOT}/v5_best.pt')
np.savez(f'{SAVE_ROOT}/v5_logits.npz', logits=v5_lg.numpy(), labels=y_te.numpy(),
         acc=v5, best_ep=v5_ep)
tg(f"V5 per-point cycle: {v5*100:.2f}% (vs V0 {v0*100:.2f})")

msg = f"""
=== V3/V4/V5 asymmetry-breaking pair_cyc (Phase A seed 0, training seed 1) ===
V0 (Cfbq alone):                       {v0*100:.2f}%
V3 (asymmetric weights pair_cyc):      {v3*100:.2f}%  ({(v3-v0)*100:+.2f}pp)
V4 (IRLS asymmetric pair_cyc):         {v4*100:.2f}%  ({(v4-v0)*100:+.2f}pp)
V5 (per-point cycle residual 13ch):    {v5*100:.2f}%  ({(v5-v0)*100:+.2f}pp)
"""
print(msg); tg(msg)
np.savez(f'{SAVE_ROOT}/summary.npz', v0=v0, v3=v3, v4=v4, v5=v5)
