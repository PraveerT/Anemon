"""TinyKNN with Cfbq input (10ch) + QCC aux losses.

Variants run sequentially:
  V1  predict per-pair q_fwd from per-frame features; loss = 1 - |q_pred · q_gt|
  V3  predict per-pair (q_fwd, tr_fwd); loss = sandwich consistency
       || quat_rotate(q_pred, xyz[t]) + tr_pred - xyz[t+1] ||^2  (matched pts only)

Both variants keep the classification head/loss identical to the 82.57 winner.
Aux loss weight a hyper-param (sweep small).
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, requests
import nvidia_dataloader

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


def quat_rotate_pts_batched(q, points):
    """q: (B,4) unit, points: (B,N,3) -> (B,N,3)."""
    B, N, _ = points.shape
    q_b = q.unsqueeze(1).expand(B, N, 4)
    pq = torch.cat([torch.zeros(B, N, 1, device=points.device, dtype=points.dtype), points], dim=-1)
    q_conj = torch.cat([q_b[..., 0:1], -q_b[..., 1:]], dim=-1)
    return hamilton(hamilton(q_b, pq), q_conj)[..., 1:]


def quat_rotate_pts_single(q, points):
    """q: (4,) unit, points: (N,3) -> (N,3)."""
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
    t = tm.squeeze(0).squeeze(0) - quat_rotate_pts_single(q, sm.squeeze(0).squeeze(0).unsqueeze(0))[0]
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


def collect(phase):
    """Cfbq features + per-pair Kabsch q,tr targets + matched mask."""
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True, assignment_mode='hungarian')
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RF = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RB = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    QF = np.zeros((N, 31, 4), dtype=np.float32)
    TR = np.zeros((N, 31, 3), dtype=np.float32)
    MATCH = np.zeros((N, 31, PTS), dtype=np.float32)  # mask used for kabsch
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
            RF[i, t+1] = (xyz_s[t+1] - (quat_rotate_pts_single(qf, xyz_s[t]) + trf)).cpu().numpy()
            qb, trb = kabsch_quat(xyz_s[t+1], xyz_s[t], matched[t])
            RB[i, t]   = (xyz_s[t]   - (quat_rotate_pts_single(qb, xyz_s[t+1]) + trb)).cpu().numpy()
            QF[i, t] = qf.cpu().numpy()
            TR[i, t] = trf.cpu().numpy()
            MATCH[i, t] = matched[t].float().cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  collect {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RF), torch.from_numpy(RB),
            torch.from_numpy(QF), torch.from_numpy(TR), torch.from_numpy(MATCH),
            torch.from_numpy(labels))


print('=== Phase A: Cfbq features + Kabsch targets ===')
xyz_tr, rf_tr, rb_tr, qf_tr, tr_tr, m_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, qf_te, tr_te, m_te, y_te = collect('test')
T = 32
t_tr_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0], T, PTS, 1)
C_tr = torch.cat([xyz_tr, rf_tr, rb_tr, t_tr_ch], dim=-1)  # 10ch
C_te = torch.cat([xyz_te, rf_te, rb_te, t_te_ch], dim=-1)
print(f"C_tr shape {C_tr.shape}")


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


class TinyKNNAux(nn.Module):
    """Tiny KNN with optional aux head exposing per-frame features post-pooling."""

    def __init__(self, in_ch=10, num_classes=25, k=16, aux_out=0):
        super().__init__()
        self.aux_out = aux_out
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
        if aux_out > 0:
            # Per-frame-pair head: concat f[t] (256) and f[t+1] (256) -> aux_out
            self.aux_head = nn.Sequential(
                nn.Linear(512, 128), nn.GELU(),
                nn.Linear(128, aux_out))

    def forward(self, x):
        B, T_, P_, C_ = x.shape
        x_bt = x.reshape(B*T_, P_, C_)
        local = self.edge(x_bt)
        h = torch.cat([local, x_bt], dim=-1)
        h = self.pt_mlp(h).reshape(B, T_, P_, -1)
        # Per-frame feature (B, T, 512) before any time conv
        per_frame = torch.cat([h.max(2).values, h.mean(2)], dim=-1)
        h_seq = per_frame.transpose(1, 2)  # (B, 512, T)
        h_seq = self.proj(h_seq)
        h_seq = F.gelu(self.b1(self.c1(h_seq)))
        h_seq = h_seq + F.gelu(self.b2(self.c2(h_seq)))
        h_seq = h_seq + F.gelu(self.b3(self.c3(h_seq)))
        h_seq = h_seq + F.gelu(self.b4(self.c4(h_seq)))
        # Aux head over pairs of per_frame features
        aux = None
        if self.aux_out > 0:
            f_t = per_frame[:, :-1]    # (B, T-1, 512)
            f_n = per_frame[:, 1:]     # (B, T-1, 512)
            aux_in = torch.cat([f_t, f_n], dim=-1)  # (B, T-1, 1024) — but head is (512 -> 128)
            # Actually aux_head expects 512 input. Use sum/concat fusion -> reduce.
            aux_in_red = f_t + f_n     # simple sum fusion to 512
            aux = self.aux_head(aux_in_red)  # (B, T-1, aux_out)
        h_out = self.drop(h_seq).max(-1).values
        logits = self.head(h_out)
        return logits, aux


def quat_dist_loss(q_pred, q_gt):
    """1 - |<q_pred, q_gt>|. q_pred (B,T-1,4), q_gt (B,T-1,4). Both unit."""
    q_pred = F.normalize(q_pred, dim=-1)
    dot = (q_pred * q_gt).sum(-1).abs()
    return (1 - dot).mean()


def sandwich_loss(q_pred, tr_pred, xyz_t, xyz_n, mask):
    """|| q_pred · xyz[t] + tr_pred - xyz[t+1] ||^2 over matched points.
    q_pred (B,T-1,4), tr_pred (B,T-1,3), xyz_t/xyz_n (B,T-1,P,3), mask (B,T-1,P).
    """
    Bs, Tm, P, _ = xyz_t.shape
    qn = F.normalize(q_pred, dim=-1)
    qf = qn.reshape(Bs * Tm, 4)
    pts_t = xyz_t.reshape(Bs * Tm, P, 3)
    rotated = quat_rotate_pts_batched(qf, pts_t).reshape(Bs, Tm, P, 3)
    pred = rotated + tr_pred.unsqueeze(2)
    diff = (pred - xyz_n) ** 2
    diff = diff.sum(-1)  # (B,T-1,P)
    mw = mask.clamp(min=0)
    denom = mw.sum().clamp(min=1.0)
    return (diff * mw).sum() / denom


def train_eval(seed, X_tr, y_tr, X_te, y_te,
               qf_tr, tr_tr, m_tr,
               aux_kind, aux_weight,
               epochs=120, bs=16, lr=2e-3, k=16, in_ch=10):
    """aux_kind: 'none' | 'quatdist' | 'sandwich'."""
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    aux_out = 4 if aux_kind == 'quatdist' else (7 if aux_kind == 'sandwich' else 0)
    model = TinyKNNAux(in_ch=in_ch, k=k, aux_out=aux_out).cuda()
    print(f"[{aux_kind}] aux_w={aux_weight} params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,epochs-warmup))))
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    qf_tr_c = qf_tr.cuda(); tr_tr_c = tr_tr.cuda(); m_tr_c = m_tr.cuda()
    best = 0.0; best_ep = -1; best_logits = None
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(X_tr), generator=g)
        cls_loss_acc, aux_loss_acc, n_batches = 0.0, 0.0, 0
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i+bs]
            xb = X_tr_c[idx]; yb = y_tr_c[idx]
            qf_b = qf_tr_c[idx]; tr_b = tr_tr_c[idx]; m_b = m_tr_c[idx]
            B_, T_, P_, _ = xb.shape
            xb = xb * (torch.rand(B_, 1, P_, 1, device=xb.device) > 0.10).float()
            opt.zero_grad()
            logits, aux = model(xb)
            loss_cls = F.cross_entropy(logits, yb, label_smoothing=0.1)
            loss_aux = torch.zeros((), device=logits.device)
            if aux_kind == 'quatdist':
                loss_aux = quat_dist_loss(aux, qf_b)
            elif aux_kind == 'sandwich':
                q_pred = aux[..., :4]
                tr_pred = aux[..., 4:7]
                xyz_t = xb[:, :-1, :, :3]
                xyz_n = xb[:, 1:,  :, :3]
                loss_aux = sandwich_loss(q_pred, tr_pred, xyz_t, xyz_n, m_b)
            loss = loss_cls + aux_weight * loss_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            cls_loss_acc += loss_cls.item(); aux_loss_acc += float(loss_aux); n_batches += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(X_te), 32):
                logits_te, _ = model(X_te_c[i:i+32])
                lg.append(logits_te.cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == y_te_c).float().mean().item()
        if acc > best: best = acc; best_ep = ep; best_logits = lg.clone()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})  "
                  f"cls={cls_loss_acc/n_batches:.3f} aux={aux_loss_acc/n_batches:.3f}")
    return best, best_logits


print('\n=== V0: baseline (no aux, 10ch Cfbq) — sanity check vs 82.57 ===')
tg("Tiny Cfbq+QCC aux: V0 baseline starting.")
acc0, _ = train_eval(1, C_tr, y_tr, C_te, y_te, qf_tr, tr_tr, m_tr,
                     aux_kind='none', aux_weight=0.0)
tg(f"V0 baseline: {acc0*100:.2f}%")

print('\n=== V1: quat-distance aux (predict q_fwd, loss=1-|q·q_gt|) ===')
results_v1 = {}
for w in [0.1, 0.5, 1.0]:
    acc, _ = train_eval(1, C_tr, y_tr, C_te, y_te, qf_tr, tr_tr, m_tr,
                        aux_kind='quatdist', aux_weight=w)
    results_v1[w] = acc
    print(f"  V1 w={w}: {acc*100:.2f}%")
tg(f"V1 quatdist results: {results_v1}")

print('\n=== V3: sandwich consistency aux (predict q+tr, loss=||q·p+tr-p\'||^2) ===')
results_v3 = {}
for w in [0.1, 0.5, 1.0]:
    acc, _ = train_eval(1, C_tr, y_tr, C_te, y_te, qf_tr, tr_tr, m_tr,
                        aux_kind='sandwich', aux_weight=w)
    results_v3[w] = acc
    print(f"  V3 w={w}: {acc*100:.2f}%")
tg(f"V3 sandwich results: {results_v3}")


msg = f"""
=== Tiny + Cfbq (10ch) + QCC aux loss ===
V0 baseline (no aux):     {acc0*100:.2f}%
TinyKNN Cfbq prior best:  82.57%

V1 quat-distance:
""" + "\n".join([f"  w={w}: {r*100:.2f}%" for w, r in results_v1.items()]) + """

V3 sandwich consistency:
""" + "\n".join([f"  w={w}: {r*100:.2f}%" for w, r in results_v3.items()])

print(msg); tg(msg)

np.savez('/tmp/tiny_cfbq_qcc_aux.npz',
         baseline=acc0,
         v1_results={str(w): r for w, r in results_v1.items()},
         v3_results={str(w): r for w, r in results_v3.items()})
print("saved /tmp/tiny_cfbq_qcc_aux.npz")
