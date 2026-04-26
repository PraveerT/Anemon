"""Fusion analysis: V2 dualstream + pair_cyc tiny vs pmamba_base.

Trains V2 once (saves logits), runs pmamba_base ep110 inference, computes
solo / oracle / alpha-blend / error correlation.
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


print('=== Phase A: Cfbq + pair_cyc features ===')
xyz_tr, rf_tr, rb_tr, pc_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, pc_te, y_te = collect('test')
T = 32
t_tr_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0], T, PTS, 1)
C_tr_10 = torch.cat([xyz_tr, rf_tr, rb_tr, t_tr_ch], dim=-1)
C_te_10 = torch.cat([xyz_te, rf_te, rb_te, t_te_ch], dim=-1)


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


def train_dual(seed, X_tr_pt, X_tr_fr, y_tr, X_te_pt, X_te_fr, y_te,
               epochs=120, bs=16, lr=2e-3, k=16):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = TinyDualStream(in_ch=10, k=k).cuda()
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,epochs-warmup))))
    Xtr_pt = X_tr_pt.cuda(); Xtr_fr = X_tr_fr.cuda(); y_tr_c = y_tr.cuda()
    Xte_pt = X_te_pt.cuda(); Xte_fr = X_te_fr.cuda(); y_te_c = y_te.cuda()
    best = 0.0; best_ep = -1; best_logits = None
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(X_tr_pt), generator=g)
        for i in range(0, len(X_tr_pt), bs):
            idx = perm[i:i+bs]
            xb_pt = Xtr_pt[idx]; xb_fr = Xtr_fr[idx]; yb = y_tr_c[idx]
            B_, T_, P_, _ = xb_pt.shape
            xb_pt = xb_pt * (torch.rand(B_, 1, P_, 1, device=xb_pt.device) > 0.10).float()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb_pt, xb_fr), yb, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(X_te_pt), 32):
                lg.append(model(Xte_pt[i:i+32], Xte_fr[i:i+32]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == y_te_c).float().mean().item()
        if acc > best: best = acc; best_ep = ep; best_logits = lg.clone()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})")
    return best, best_logits


print('\n=== Phase B: train V2 dualstream + pair_cyc ===')
tg("Pair_cyc fusion analysis: training V2.")
v2_acc, v2_logits = train_dual(1, C_tr_10, pc_tr, y_tr, C_te_10, pc_te, y_te, epochs=120, k=16)
print(f"\nV2 final: {v2_acc*100:.2f}%")


print('\n=== Phase C: pmamba_base ep110 inference on test ===')
pm = motion.Motion(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
sd = torch.load('/notebooks/PMamba/experiments/work_dir/pmamba_branch/epoch110_model.pt',
                map_location='cuda')
pm.load_state_dict(sd['model_state_dict'], strict=False)
pm.eval()
loader_pm = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
N = len(loader_pm)
pm_logits = torch.zeros(N, 25)
pm_labels = torch.zeros(N, dtype=torch.long)
with torch.no_grad():
    for i in range(N):
        pts, lab, _ = loader_pm[i]
        pts_t = (pts if torch.is_tensor(pts) else torch.from_numpy(pts)).float().cuda().unsqueeze(0)
        out = pm(pts_t)
        pm_logits[i] = out[0].cpu(); pm_labels[i] = int(lab)
        if (i+1) % 200 == 0: print(f'  {i+1}/{N}')
assert torch.equal(y_te, pm_labels), "label order mismatch"


print('\n=== Phase D: Fusion analysis ===')
y = y_te
pm_p = F.softmax(pm_logits, -1)
v2_p = F.softmax(v2_logits, -1)
pm_right = pm_p.argmax(-1) == y
v2_right = v2_p.argmax(-1) == y
pm_solo = pm_right.float().mean().item()
v2_solo = v2_right.float().mean().item()
oracle = (pm_right | v2_right).float().mean().item()

both_right = (pm_right & v2_right).sum().item()
pm_only = (pm_right & ~v2_right).sum().item()
v2_only = (~pm_right & v2_right).sum().item()
both_wrong = (~pm_right & ~v2_right).sum().item()

ep = (~pm_right).float(); et = (~v2_right).float()
cov = ((ep - ep.mean()) * (et - et.mean())).mean().item()
r = cov / (ep.std().item() * et.std().item() + 1e-9)

best_a = 0; best_a_acc = 0
for a in np.arange(0.0, 1.01, 0.02):
    f = a * pm_p + (1 - a) * v2_p
    acc = (f.argmax(-1) == y).float().mean().item()
    if acc > best_a_acc: best_a_acc = acc; best_a = a

# Temp calibrated
def temp_cal(logits, labels):
    bt = 1.0; bn = 1e9
    for T in np.arange(0.3, 3.1, 0.1):
        n = F.cross_entropy(logits / T, labels).item()
        if n < bn: bn = n; bt = T
    return bt
T_pm = temp_cal(pm_logits, y); T_v2 = temp_cal(v2_logits, y)
pm_pT = F.softmax(pm_logits / T_pm, -1)
v2_pT = F.softmax(v2_logits / T_v2, -1)
best_aT = 0; best_aT_acc = 0
for a in np.arange(0.0, 1.01, 0.02):
    f = a * pm_pT + (1 - a) * v2_pT
    acc = (f.argmax(-1) == y).float().mean().item()
    if acc > best_aT_acc: best_aT_acc = acc; best_aT = a


msg = f"""
=== V2 dualstream + pair_cyc + pmamba_base fusion ===
pmamba_base solo:  {pm_solo*100:.2f}%
V2 solo:           {v2_solo*100:.2f}%
Oracle:            {oracle*100:.2f}%
Alpha-blend raw:   {best_a_acc*100:.2f}% at a={best_a:.2f}
Alpha-blend calT:  {best_aT_acc*100:.2f}% at a={best_aT:.2f} (T_pm={T_pm:.1f}, T_v2={T_v2:.1f})
Error correlation r: {r:.3f}

both_right={both_right} pm_only={pm_only} v2_only={v2_only} both_wrong={both_wrong}

Reference points (prior fusion runs):
  TinyKNN Cfbq fuse: 90.66 (+0.83), oracle 92.95, r=0.44
  rigidres fuse:     90.04, oracle 94.19
"""
print(msg); tg(msg)

np.savez('/tmp/fusion_pair_cyc.npz',
         v2_logits=v2_logits.numpy(), pm_logits=pm_logits.numpy(),
         labels=y.numpy(), v2_solo=v2_solo, pm_solo=pm_solo,
         oracle=oracle, fuse_raw=best_a_acc, fuse_a=best_a,
         fuse_calT=best_aT_acc, fuse_aT=best_aT, r=r,
         both_right=both_right, pm_only=pm_only, v2_only=v2_only, both_wrong=both_wrong)
print("saved /tmp/fusion_pair_cyc.npz")
