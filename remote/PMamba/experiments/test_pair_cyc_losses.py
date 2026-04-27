"""Three pair_cyc loss designs on Cfbq baseline (V0, no dualstream).

Base: TinyKNN on Cfbq 10ch (the 82.57 winner architecture).
Add ONE pair_cyc-derived loss and see if it beats V0.

L1  Predict pair_cyc from per-frame features (aux MSE task)
L2  Cycle closure: backbone predicts (q_f, q_b) per pair, loss enforces composition = identity
L3  Pair_cyc-weighted cls loss (per-clip weighting by mean ||pair_cyc||)

Phase A seeded for reproducibility.
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


def kabsch_quat(src, tgt, mask):
    w = mask.float().unsqueeze(0)
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
IDENT = torch.tensor([1.0, 0.0, 0.0, 0.0])


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True)
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RF = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RB = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    PAIRCYC = np.zeros((N, 32, 4), dtype=np.float32)
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
            qf, trf = kabsch_quat(xyz_s[t], xyz_s[t+1], matched[t])
            qb, trb = kabsch_quat(xyz_s[t+1], xyz_s[t], matched[t])
            RF[i, t+1] = (xyz_s[t+1] - (quat_rotate_pts(qf, xyz_s[t]) + trf)).cpu().numpy()
            RB[i, t]   = (xyz_s[t]   - (quat_rotate_pts(qb, xyz_s[t+1]) + trb)).cpu().numpy()
            qcyc = hamilton(qf.unsqueeze(0), qb.unsqueeze(0))[0]
            qcyc = F.normalize(qcyc, dim=-1)
            if (qcyc * IDENT.to(qcyc.device)).sum() < 0:
                qcyc = -qcyc
            PAIRCYC[i, t] = (qcyc - IDENT.to(qcyc.device)).cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RF),
            torch.from_numpy(RB), torch.from_numpy(PAIRCYC),
            torch.from_numpy(labels))


print('=== Phase A: Cfbq + pair_cyc ===')
xyz_tr, rf_tr, rb_tr, pc_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, pc_te, y_te = collect('test')
T = 32
t_tr_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0], T, PTS, 1)
C_tr = torch.cat([xyz_tr, rf_tr, rb_tr, t_tr_ch], dim=-1)  # 10ch baseline
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


class TinyV0WithAux(nn.Module):
    """82.57 baseline architecture + optional aux heads."""
    def __init__(self, in_ch=10, num_classes=25, k=16, mode='V0'):
        super().__init__()
        self.mode = mode
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
        # L1: predict pair_cyc (4-vec) from per-frame features
        if mode == 'L1':
            self.aux_l1 = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 4))
        # L2: predict q_f and q_b per pair from per-frame features
        if mode == 'L2':
            self.aux_l2 = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 8))  # 4 + 4

    def forward(self, x):
        B, T_, P_, C_ = x.shape
        x_bt = x.reshape(B*T_, P_, C_)
        local = self.edge(x_bt)
        h = torch.cat([local, x_bt], dim=-1)
        h = self.pt_mlp(h).reshape(B, T_, P_, -1)
        per_frame = torch.cat([h.max(2).values, h.mean(2)], dim=-1)  # (B, T, 512)
        h_seq = per_frame.transpose(1, 2)
        h_seq = self.proj(h_seq)
        h_seq = F.gelu(self.b1(self.c1(h_seq)))
        h_seq = h_seq + F.gelu(self.b2(self.c2(h_seq)))
        h_seq = h_seq + F.gelu(self.b3(self.c3(h_seq)))
        h_seq = h_seq + F.gelu(self.b4(self.c4(h_seq)))
        # Per-frame final feature for aux heads (B, 256, T) -> (B, T, 256)
        frame_feats = h_seq.transpose(1, 2)
        h_pool = self.drop(h_seq).max(-1).values
        logits = self.head(h_pool)
        aux_out = None
        if self.mode == 'L1':
            aux_out = self.aux_l1(frame_feats)        # (B, T, 4)
        elif self.mode == 'L2':
            aux_out = self.aux_l2(frame_feats)        # (B, T, 8)
        return logits, aux_out


def train_eval(seed, X_tr, X_tr_pc, y_tr, X_te, X_te_pc, y_te, mode,
               aux_lambda=0.5, epochs=120, bs=16, lr=2e-3, k=16):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = TinyV0WithAux(in_ch=10, k=k, mode=mode).cuda()
    print(f"[{mode}] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,epochs-warmup))))
    Xtr = X_tr.cuda(); Xtr_pc = X_tr_pc.cuda(); y_tr_c = y_tr.cuda()
    Xte = X_te.cuda(); Xte_pc = X_te_pc.cuda(); y_te_c = y_te.cuda()
    best = 0.0; best_ep = -1
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(X_tr), generator=g)
        cls_acc, aux_acc, n = 0.0, 0.0, 0
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i+bs]
            xb = Xtr[idx]; pcb = Xtr_pc[idx]; yb = y_tr_c[idx]
            B_, T_, P_, _ = xb.shape
            xb = xb * (torch.rand(B_, 1, P_, 1, device=xb.device) > 0.10).float()
            opt.zero_grad()
            logits, aux = model(xb)
            # Per-clip cls (possibly weighted for L3)
            if mode == 'L3':
                clip_w = pcb.norm(dim=-1).mean(-1)        # (B,)
                clip_w = clip_w / (clip_w.mean() + 1e-6)  # normalize so mean=1
                ce = F.cross_entropy(logits, yb, label_smoothing=0.1, reduction='none')
                loss_cls = (ce * clip_w).mean()
            else:
                loss_cls = F.cross_entropy(logits, yb, label_smoothing=0.1)
            loss_aux = torch.zeros((), device=logits.device)
            if mode == 'L1':
                # MSE between predicted and true pair_cyc; mask last frame (no pair)
                target = pcb[:, :-1]                       # (B, T-1, 4)
                pred = aux[:, :-1]                          # (B, T-1, 4)
                loss_aux = aux_lambda * ((pred - target) ** 2).mean()
            elif mode == 'L2':
                # Cycle closure: ||Hamilton(qf_pred, qb_pred) - identity||²
                qf_pred = F.normalize(aux[:, :-1, :4], dim=-1)
                qb_pred = F.normalize(aux[:, :-1, 4:], dim=-1)
                comp = hamilton(qf_pred, qb_pred)
                # Hemisphere fix
                ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=comp.device)
                sign = (comp * ident).sum(-1, keepdim=True).sign().clamp(min=-1, max=1)
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                comp = comp * sign
                loss_aux = aux_lambda * ((comp - ident) ** 2).sum(-1).mean()
            loss = loss_cls + loss_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            cls_acc += loss_cls.item(); aux_acc += float(loss_aux); n += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(X_te), 32):
                logits_te, _ = model(Xte[i:i+32])
                lg.append(logits_te.cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == y_te_c).float().mean().item()
        if acc > best: best = acc; best_ep = ep
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  [{mode}] ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep}) cls={cls_acc/n:.3f} aux={aux_acc/n:.4f}")
    return best


tg("Pair_cyc loss tests starting: V0 + L1 + L2 + L3 (Phase A seeded).")

print('\n=== V0: TinyKNN baseline (no aux) ===')
v0 = train_eval(1, C_tr, pc_tr, y_tr, C_te, pc_te, y_te, mode='V0', epochs=120)
tg(f"V0 baseline (seeded Phase A): {v0*100:.2f}%")

print('\n=== L1: predict pair_cyc from features (aux MSE) ===')
l1 = train_eval(1, C_tr, pc_tr, y_tr, C_te, pc_te, y_te, mode='L1', aux_lambda=0.5, epochs=120)
tg(f"L1 predict pair_cyc: {l1*100:.2f}% (vs V0 {v0*100:.2f})")

print('\n=== L2: cycle-closure on predicted q_f, q_b ===')
l2 = train_eval(1, C_tr, pc_tr, y_tr, C_te, pc_te, y_te, mode='L2', aux_lambda=0.5, epochs=120)
tg(f"L2 cycle closure: {l2*100:.2f}% (vs V0 {v0*100:.2f})")

print('\n=== L3: pair_cyc-weighted cls loss ===')
l3 = train_eval(1, C_tr, pc_tr, y_tr, C_te, pc_te, y_te, mode='L3', epochs=120)
tg(f"L3 weighted cls: {l3*100:.2f}% (vs V0 {v0*100:.2f})")

msg = f"""
=== pair_cyc loss tests (Cfbq baseline + loss) ===
V0 (no aux):              {v0*100:.2f}%
L1 predict pair_cyc:      {l1*100:.2f}%  ({(l1-v0)*100:+.2f}pp)
L2 cycle-closure:         {l2*100:.2f}%  ({(l2-v0)*100:+.2f}pp)
L3 weighted cls:          {l3*100:.2f}%  ({(l3-v0)*100:+.2f}pp)
"""
print(msg); tg(msg)
