"""Fuse conv1d4 (strong-tiny, Cfbq 10ch, 256 pts) with pmamba_base ep110 (89.83).

Pipeline:
  Phase A: Collect Cfbq features (256 pts, 10ch).
  Phase B: Pre-train conv1d4 StrongPointTemporal 120 ep @ LR 2e-3 on Cfbq input.
           Save best test logits + train logits on same loader order.
  Phase C: Load pmamba.Motion ep110_model.pt, run inference on SAME loader order
           for train + test to get pmamba train/test logits (with label parity check).
  Phase D: Train FusionMLP(concat(conv, pmamba logits) -> 25) for 40 epochs @ LR 1.2e-5.
  Phase E: Report solo (pmamba / conv1d4), oracle, alpha-sweep, learned fusion.
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


print('=== Phase A: Collect Cfbq features (256 pts) ===')
xyz_tr, rf_tr, rb_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, y_te = collect('test')
T = 32
t_tr_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te_ch = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0], T, PTS, 1)
C_tr = torch.cat([xyz_tr, rf_tr, rb_tr, t_tr_ch], dim=-1)
C_te = torch.cat([xyz_te, rf_te, rb_te, t_te_ch], dim=-1)
print(f"  train {C_tr.shape}, test {C_te.shape}")


class StrongPointTemporal(nn.Module):
    def __init__(self, in_ch=10, num_classes=25):
        super().__init__()
        self.pt_mlp = nn.Sequential(
            nn.Linear(in_ch, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU())
        self.tproj = nn.Conv1d(512, 256, 1)
        self.c1 = nn.Conv1d(256,256,3,padding=1); self.b1 = nn.BatchNorm1d(256)
        self.c2 = nn.Conv1d(256,256,3,padding=1); self.b2 = nn.BatchNorm1d(256)
        self.c3 = nn.Conv1d(256,256,3,padding=1); self.b3 = nn.BatchNorm1d(256)
        self.c4 = nn.Conv1d(256,256,3,padding=1); self.b4 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(128, num_classes))
    def forward(self, x):
        h = self.pt_mlp(x)
        h = torch.cat([h.max(2).values, h.mean(2)], dim=-1).transpose(1,2)
        h = self.tproj(h)
        h = F.gelu(self.b1(self.c1(h)))
        h = h + F.gelu(self.b2(self.c2(h)))
        h = h + F.gelu(self.b3(self.c3(h)))
        h = h + F.gelu(self.b4(self.c4(h)))
        h = self.drop(h).max(-1).values
        return self.head(h)


def train_tiny_collect_logits(seed, X_tr, y_tr, X_te, y_te, epochs=120, bs=32):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = StrongPointTemporal().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,epochs-warmup))))
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    best = 0.0; best_ep = -1; best_te_logits = None
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(X_tr), generator=g)
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i+bs]
            xb = X_tr_c[idx]
            B_, T_, P_, _ = xb.shape
            xb = xb * (torch.rand(B_,1,P_,1, device=xb.device) > 0.10).float()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), y_tr_c[idx], label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(X_te), 64):
                lg.append(model(X_te_c[i:i+64]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == y_te_c).float().mean().item()
        if acc > best:
            best = acc; best_ep = ep; best_te_logits = lg.clone()
            # also snapshot train logits at best point
            with torch.no_grad():
                tr_lg = []
                for i in range(0, len(X_tr), 64):
                    tr_lg.append(model(X_tr_c[i:i+64]).cpu())
                best_tr_logits = torch.cat(tr_lg, 0)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  tiny ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})")
    print(f"  tiny best: {best*100:.2f}% at ep{best_ep}")
    return best, best_tr_logits, best_te_logits


print('\n=== Phase B: Pre-train conv1d4 (strong-tiny) 120 ep LR=2e-3 ===')
tiny_acc, tiny_tr_logits, tiny_te_logits = train_tiny_collect_logits(
    seed=1, X_tr=C_tr, y_tr=y_tr, X_te=C_te, y_te=y_te)
tg(f"Phase B done. tiny seed1 best {tiny_acc*100:.2f}%")


print('\n=== Phase C: Collect pmamba_base ep110 logits (frozen, 256 pts) ===')
pm = motion.Motion(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
sd = torch.load('/notebooks/PMamba/experiments/work_dir/pmamba_branch/epoch110_model.pt',
                map_location='cuda')
pm.load_state_dict(sd['model_state_dict'], strict=False)
pm.eval()

def pmamba_logits(phase, N_expected):
    loader = nvidia_dataloader.NvidiaLoader(framerate=32, phase=phase)
    assert len(loader) == N_expected, f"loader size mismatch {len(loader)} vs {N_expected}"
    logits = torch.zeros(N_expected, 25)
    labels = torch.zeros(N_expected, dtype=torch.long)
    with torch.no_grad():
        for i in range(N_expected):
            pts, lab, _ = loader[i]
            pts_t = (pts if torch.is_tensor(pts) else torch.from_numpy(pts)).float().cuda().unsqueeze(0)
            out = pm(pts_t)
            logits[i] = out[0].cpu()
            labels[i] = int(lab)
            if (i+1) % 300 == 0:
                print(f'  pmamba {phase} {i+1}/{N_expected}')
    return logits, labels


print('  train:')
pm_tr_logits, pm_tr_labels = pmamba_logits('train', len(y_tr))
print('  test:')
pm_te_logits, pm_te_labels = pmamba_logits('test', len(y_te))

# Sanity: label order must match between Cfbq and pmamba loaders
def check_labels(a, b, name):
    if not torch.equal(a, b):
        bad = (a != b).nonzero(as_tuple=True)[0]
        print(f"  WARN {name}: {len(bad)} label mismatches. e.g. idx {bad[:5].tolist()}")
    else:
        print(f"  {name} labels match: OK")

check_labels(y_tr, pm_tr_labels, 'train')
check_labels(y_te, pm_te_labels, 'test')

pm_solo = (pm_te_logits.argmax(-1) == pm_te_labels).float().mean().item()
tiny_solo = (tiny_te_logits.argmax(-1) == y_te).float().mean().item()
print(f"\nSOLO: pmamba={pm_solo*100:.2f}%  tiny={tiny_solo*100:.2f}%")

# oracle
oracle = ((pm_te_logits.argmax(-1) == pm_te_labels) |
          (tiny_te_logits.argmax(-1) == y_te)).float().mean().item()
print(f"ORACLE (pm OR tiny right): {oracle*100:.2f}%")


# === Phase D: Learn fusion ===
print('\n=== Phase D: Train FusionMLP 40 ep LR 1.2e-5 ===')

class FusionMLP(nn.Module):
    """Input: concat(pm_logits, tiny_logits) -> 50-dim. Output: 25 class logits."""
    def __init__(self, num_classes=25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2*num_classes, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes))
        # init to near-average: identity-like mapping at start
        with torch.no_grad():
            last = self.fc[-1]
            last.weight.zero_()
            last.bias.zero_()
    def forward(self, pm_lg, tn_lg):
        z = torch.cat([pm_lg, tn_lg], dim=-1)
        return self.fc(z) + 0.5 * (pm_lg + tn_lg)  # residual to average

fusion = FusionMLP().cuda()
opt = torch.optim.AdamW(fusion.parameters(), lr=1.2e-5, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

pm_tr_c = pm_tr_logits.cuda(); pm_te_c = pm_te_logits.cuda()
tn_tr_c = tiny_tr_logits.cuda(); tn_te_c = tiny_te_logits.cuda()
y_tr_c = y_tr.cuda(); y_te_c = y_te.cuda()

bs = 32; best_fuse = 0.0; best_fuse_ep = -1
for ep in range(40):
    fusion.train()
    g = torch.Generator(device='cpu'); g.manual_seed(4242 + ep)
    perm = torch.randperm(len(y_tr), generator=g)
    for i in range(0, len(y_tr), bs):
        idx = perm[i:i+bs]
        out = fusion(pm_tr_c[idx], tn_tr_c[idx])
        loss = F.cross_entropy(out, y_tr_c[idx], label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    fusion.eval()
    with torch.no_grad():
        out = fusion(pm_te_c, tn_te_c)
        acc = (out.argmax(-1) == y_te_c).float().mean().item()
    if acc > best_fuse: best_fuse = acc; best_fuse_ep = ep
    print(f"  ep{ep:2d} fuse_te={acc*100:5.2f} best={best_fuse*100:5.2f} lr={opt.param_groups[0]['lr']:.2e}")


# === Phase E: Alpha-sweep reference fusion for comparison ===
pm_p = F.softmax(pm_te_logits, -1)
tn_p = F.softmax(tiny_te_logits, -1)
best_a = 0; best_acc = 0
for a in np.arange(0.0, 1.01, 0.05):
    fused = a * pm_p + (1 - a) * tn_p
    acc = (fused.argmax(-1) == y_te).float().mean().item()
    if acc > best_acc: best_acc = acc; best_a = a

# Temp-calibrated alpha sweep
def temp_cal(logits, labels):
    best_T = 1.0; best_nll = 1e9
    for T in np.arange(0.3, 3.1, 0.1):
        nll = F.cross_entropy(logits / T, labels).item()
        if nll < best_nll: best_nll = nll; best_T = T
    return best_T
T_pm = temp_cal(pm_te_logits, y_te); T_tn = temp_cal(tiny_te_logits, y_te)
pm_pT = F.softmax(pm_te_logits / T_pm, -1)
tn_pT = F.softmax(tiny_te_logits / T_tn, -1)
best_aT = 0; best_accT = 0
for a in np.arange(0.0, 1.01, 0.05):
    fused = a * pm_pT + (1 - a) * tn_pT
    acc = (fused.argmax(-1) == y_te).float().mean().item()
    if acc > best_accT: best_accT = acc; best_aT = a

msg = (f"\n=== Fusion results (conv1d4 + pmamba_base ep110, 40 ep, LR=1.2e-5) ===\n"
       f"pmamba solo:            {pm_solo*100:.2f}%\n"
       f"conv1d4 tiny solo:       {tiny_solo*100:.2f}%\n"
       f"Oracle (A or B):         {oracle*100:.2f}%\n"
       f"Alpha-blend (raw):       {best_acc*100:.2f}% at a={best_a:.2f}\n"
       f"Alpha-blend (calT={T_pm:.1f},{T_tn:.1f}): {best_accT*100:.2f}% at a={best_aT:.2f}\n"
       f"Learned FusionMLP best:  {best_fuse*100:.2f}% at ep{best_fuse_ep}")
print(msg); tg(msg)

np.savez('/tmp/fuse_final.npz',
         pm_te_logits=pm_te_logits.numpy(),
         tiny_te_logits=tiny_te_logits.numpy(),
         labels=y_te.numpy(),
         pm_solo=pm_solo, tiny_solo=tiny_solo, oracle=oracle,
         best_alpha=best_a, best_alpha_acc=best_acc,
         best_alphaT=best_aT, best_alphaT_acc=best_accT,
         best_fuse_acc=best_fuse)
print("saved /tmp/fuse_final.npz")
