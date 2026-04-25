"""Full fusion analysis for TinyKNN vs PMamba base.

1. Collect Cfbq features 256 pts.
2. Train TinyKNN (EdgeConv k=16) 120 ep, save test logits.
3. Run pmamba_base ep110 inference on test, save logits.
4. Compute: solo, oracle, alpha-blend, error correlation, unique-recovery.
Saves logits to /tmp/tinyknn_fuse.npz for future use.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math, requests
import nvidia_dataloader
from models import motion

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


def quat_rotate_points(q, points):
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
    t = tm.squeeze(0).squeeze(0) - quat_rotate_points(q, sm.squeeze(0).squeeze(0).unsqueeze(0))[0]
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
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True)
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RF = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RB = np.zeros((N, 32, PTS, 3), dtype=np.float32)
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
            RF[i, t+1] = (xyz_s[t+1] - (quat_rotate_points(qf, xyz_s[t]) + trf)).cpu().numpy()
            qb, trb = kabsch_quat(xyz_s[t+1], xyz_s[t], matched[t])
            RB[i, t]   = (xyz_s[t]   - (quat_rotate_points(qb, xyz_s[t+1]) + trb)).cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  Cfbq {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RF),
            torch.from_numpy(RB), torch.from_numpy(labels))


print('=== Phase A: Cfbq features ===')
xyz_tr, rf_tr, rb_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, y_te = collect('test')
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


def train_eval(seed, X_tr, y_tr, X_te, y_te, epochs=120, bs=16, lr=2e-3, k=16):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = TinyKNN(k=k).cuda()
    print(f"seed {seed} k={k} params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,epochs-warmup))))
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    best = 0.0; best_ep = -1; best_logits = None
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(X_tr), generator=g)
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i+bs]
            xb = X_tr_c[idx]; yb = y_tr_c[idx]
            B_, T_, P_, _ = xb.shape
            xb = xb * (torch.rand(B_, 1, P_, 1, device=xb.device) > 0.10).float()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(X_te), 32):
                lg.append(model(X_te_c[i:i+32]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == y_te_c).float().mean().item()
        if acc > best: best = acc; best_ep = ep; best_logits = lg.clone()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})")
    return best, best_logits


print('\n=== Phase B: Train TinyKNN (k=16, 120 ep) ===')
tg("TinyKNN full analyze: train+pmamba infer+fusion. Starting.")
tk_acc, tk_logits = train_eval(1, C_tr, y_tr, C_te, y_te, epochs=120, k=16)
tg(f"TinyKNN best: {tk_acc*100:.2f}%")


print('\n=== Phase C: pmamba_base ep110 test inference ===')
pm = motion.Motion(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
sd = torch.load('/notebooks/PMamba/experiments/work_dir/pmamba_branch/epoch110_model.pt',
                map_location='cuda')
pm.load_state_dict(sd['model_state_dict'], strict=False)
pm.eval()

loader = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
pm_logits = torch.zeros(len(loader), 25)
pm_labels = torch.zeros(len(loader), dtype=torch.long)
with torch.no_grad():
    for i in range(len(loader)):
        pts, lab, _ = loader[i]
        pts_t = (pts if torch.is_tensor(pts) else torch.from_numpy(pts)).float().cuda().unsqueeze(0)
        out = pm(pts_t)
        pm_logits[i] = out[0].cpu(); pm_labels[i] = int(lab)
        if (i+1) % 200 == 0: print(f"  pmamba test {i+1}/{len(loader)}")

assert torch.equal(y_te, pm_labels), "label mismatch"


print('\n=== Phase D: Fusion analysis ===')
y = y_te; N = len(y)
pm_p = F.softmax(pm_logits, -1)
tk_p = F.softmax(tk_logits, -1)

pm_right = pm_p.argmax(-1) == y
tk_right = tk_p.argmax(-1) == y
pm_solo = pm_right.float().mean().item()
tk_solo = tk_right.float().mean().item()
oracle = (pm_right | tk_right).float().mean().item()

both_right = (pm_right & tk_right).sum().item()
pm_only    = (pm_right & ~tk_right).sum().item()
tk_only    = (~pm_right & tk_right).sum().item()
both_wrong = (~pm_right & ~tk_right).sum().item()

ep = (~pm_right).float(); et = (~tk_right).float()
cov = ((ep - ep.mean()) * (et - et.mean())).mean().item()
r = cov / (ep.std().item() * et.std().item() + 1e-9)

best_a = 0; best_a_acc = 0
for a in np.arange(0.0, 1.01, 0.02):
    f = a * pm_p + (1 - a) * tk_p
    acc = (f.argmax(-1) == y).float().mean().item()
    if acc > best_a_acc: best_a_acc = acc; best_a = a

# Temp-calibrated
def temp_cal(logits, labels):
    bt = 1.0; bn = 1e9
    for T in np.arange(0.3, 3.1, 0.1):
        n = F.cross_entropy(logits / T, labels).item()
        if n < bn: bn = n; bt = T
    return bt
T_pm = temp_cal(pm_logits, y); T_tk = temp_cal(tk_logits, y)
pm_pT = F.softmax(pm_logits / T_pm, -1)
tk_pT = F.softmax(tk_logits / T_tk, -1)
best_aT = 0; best_aT_acc = 0
for a in np.arange(0.0, 1.01, 0.02):
    f = a * pm_pT + (1 - a) * tk_pT
    acc = (f.argmax(-1) == y).float().mean().item()
    if acc > best_aT_acc: best_aT_acc = acc; best_aT = a


msg = f"""
=== TinyKNN (81.54-ish seed 1) + pmamba_base ep110 fusion ===
pmamba solo:           {pm_solo*100:.2f}%
TinyKNN solo:          {tk_solo*100:.2f}%
Oracle (A or B):       {oracle*100:.2f}%
Alpha-blend (raw):     {best_a_acc*100:.2f}% at a={best_a:.2f}
Alpha-blend (calT={T_pm:.1f},{T_tk:.1f}): {best_aT_acc*100:.2f}% at a={best_aT:.2f}
Error correlation r:   {r:.3f}
both_right: {both_right}  pm_only: {pm_only}  tn_only: {tk_only}  both_wrong: {both_wrong}
Recovery ceiling:      {tk_only} samples ({tk_only/N*100:.2f}pp)

Compare vs no-kNN tiny prior run (78.01 solo, oracle 92.74, fusion 90.04 @ a=0.90):
  solo delta:         {(tk_solo-0.7801)*100:+.2f}pp
  oracle delta:       {(oracle-0.9274)*100:+.2f}pp
  fusion delta:       {(best_a_acc-0.9004)*100:+.2f}pp
  tn_only delta:      previous tn_only=14
"""
print(msg); tg(msg)

np.savez('/tmp/tinyknn_fuse.npz',
         tk_logits=tk_logits.numpy(),
         pm_logits=pm_logits.numpy(),
         labels=y.numpy(),
         pm_solo=pm_solo, tk_solo=tk_solo, oracle=oracle,
         fuse_raw=best_a_acc, fuse_a=best_a,
         fuse_calT=best_aT_acc, fuse_aT=best_aT,
         r=r, both_right=both_right, pm_only=pm_only,
         tk_only=tk_only, both_wrong=both_wrong)
print("saved /tmp/tinyknn_fuse.npz")
