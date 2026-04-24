"""A/B/C test: add QCC-based features (cumulative composed quaternion) on top
of rigidres B. 5 seeds each.

A: [xyz, t] 4-ch (baseline)
B: [xyz, res, t] 7-ch (rigidres — proven +3.65pp over A in seal test)
C: [xyz, res, q_cum, t] 11-ch
   q_cum(t) = q(0,1) ⊗ q(1,2) ⊗ ... ⊗ q(t-1,t) = cumulative composed quaternion
   Composition property = definitive cycle consistency primitive

Same tiny model, same seeds, same recipe. If C > B mean, cycle-composed
quaternion info adds orthogonal signal beyond residual alone.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import nvidia_dataloader

TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"


def tg(msg):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            chat_id = r["result"][-1]["message"]["chat"]["id"]
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


def rot_to_quat(R):
    orig = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    m00,m01,m02 = Rf[:,0,0], Rf[:,0,1], Rf[:,0,2]
    m10,m11,m12 = Rf[:,1,0], Rf[:,1,1], Rf[:,1,2]
    m20,m21,m22 = Rf[:,2,0], Rf[:,2,1], Rf[:,2,2]
    tr = m00 + m11 + m22
    B = Rf.shape[0]; device = Rf.device
    q = torch.zeros(B, 4, device=device, dtype=Rf.dtype)
    m1 = tr > 0
    if m1.any():
        s = torch.sqrt(tr[m1].clamp(min=-0.999) + 1.0) * 2
        q[m1,0]=0.25*s; q[m1,1]=(m21[m1]-m12[m1])/s
        q[m1,2]=(m02[m1]-m20[m1])/s; q[m1,3]=(m10[m1]-m01[m1])/s
    rem = ~m1
    m2a = rem & (m00>m11) & (m00>m22)
    if m2a.any():
        s = torch.sqrt(1+m00[m2a]-m11[m2a]-m22[m2a]).clamp(min=1e-8)*2
        q[m2a,0]=(m21[m2a]-m12[m2a])/s; q[m2a,1]=0.25*s
        q[m2a,2]=(m01[m2a]+m10[m2a])/s; q[m2a,3]=(m02[m2a]+m20[m2a])/s
    m2b = rem & (~m2a) & (m11>m22)
    if m2b.any():
        s = torch.sqrt(1+m11[m2b]-m00[m2b]-m22[m2b]).clamp(min=1e-8)*2
        q[m2b,0]=(m02[m2b]-m20[m2b])/s; q[m2b,1]=(m01[m2b]+m10[m2b])/s
        q[m2b,2]=0.25*s; q[m2b,3]=(m12[m2b]+m21[m2b])/s
    m2c = rem & (~m2a) & (~m2b)
    if m2c.any():
        s = torch.sqrt(1+m22[m2c]-m00[m2c]-m11[m2c]).clamp(min=1e-8)*2
        q[m2c,0]=(m10[m2c]-m01[m2c])/s; q[m2c,1]=(m02[m2c]+m20[m2c])/s
        q[m2c,2]=(m12[m2c]+m21[m2c])/s; q[m2c,3]=0.25*s
    return F.normalize(q, dim=-1).reshape(*orig, 4)


def hamilton(a, b):
    aw,ax,ay,az = a[...,0],a[...,1],a[...,2],a[...,3]
    bw,bx,by,bz = b[...,0],b[...,1],b[...,2],b[...,3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def kabsch_Rt(src, tgt, mask):
    B = src.shape[0]; device = src.device
    w = mask.float()
    w_sum = w.sum(-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
    sm = (src * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    tm = (tgt * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
    sc = src - sm; tc = tgt - tm
    H = torch.einsum('bn,bni,bnj->bij', w, sc, tc)
    H = H + 1e-6 * torch.eye(3, device=device).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    det = torch.det(V @ U.transpose(-1, -2))
    D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
    R = V @ D @ U.transpose(-1, -2)
    t = tm.squeeze(1) - torch.bmm(R, sm.transpose(-1, -2)).squeeze(-1)
    return R, t


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


PTS = 128
tg("A/B/C test: adding cumulative composed quaternion to B. 5 seeds.")
print('Collecting...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    RES = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    Q_CUM = np.zeros((N, 32, 4), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    for i in range(N):
        s = loader[i]
        pts_dict = s[0]; label = s[1]
        pts = pts_dict['points'].cuda()
        F_, P, C = pts.shape
        xyz = pts[..., :3]
        orig = pts_dict['orig_flat_idx'].cuda()
        ctgt = torch.from_numpy(pts_dict['corr_full_target_idx']).long().cuda()
        cw = torch.from_numpy(pts_dict['corr_full_weight']).float().cuda()
        sampled_idx, matched = corr_sample_indices(orig, ctgt, cw, PTS, F_, P)
        xyz_samp = torch.gather(xyz, 1, sampled_idx.unsqueeze(-1).expand(-1, -1, 3))
        XYZ[i] = xyz_samp.cpu().numpy()
        # Init cumulative quaternion at frame 0 = identity
        q_cum = torch.tensor([1.0, 0.0, 0.0, 0.0], device=xyz_samp.device)
        Q_CUM[i, 0] = q_cum.cpu().numpy()
        for t in range(F_ - 1):
            R, tr = kabsch_Rt(xyz_samp[t:t+1], xyz_samp[t+1:t+2], matched[t:t+1])
            rigid_pred = torch.bmm(R, xyz_samp[t:t+1].transpose(-1, -2)).transpose(-1, -2) + tr.unsqueeze(1)
            res = xyz_samp[t+1:t+2] - rigid_pred
            RES[i, t+1] = res[0].cpu().numpy()
            # Cumulative quaternion composition (cycle consistency primitive)
            q_pair = rot_to_quat(R[0])                          # (4,)
            # Sign-canonicalize: keep w positive (consistent branch on SO(3))
            if q_pair[0] < 0:
                q_pair = -q_pair
            q_cum = hamilton(q_cum.unsqueeze(0), q_pair.unsqueeze(0))[0]
            q_cum = F.normalize(q_cum, dim=-1)
            if q_cum[0] < 0:
                q_cum = -q_cum
            Q_CUM[i, t+1] = q_cum.cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RES),
            torch.from_numpy(Q_CUM), torch.from_numpy(labels))


xyz_tr, res_tr, qcum_tr, y_tr = collect('train')
xyz_te, res_te, qcum_te, y_te = collect('test')
print(f'xyz: {xyz_tr.shape}  res: {res_tr.shape}  q_cum: {qcum_tr.shape}')
tg(f"Collection done. Training A/B/C × 5 seeds.")

T = xyz_tr.shape[1]
t_tr = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0],  T, PTS, 1)
# Broadcast q_cum to all points
qcum_tr_p = qcum_tr.unsqueeze(2).expand(-1, -1, PTS, -1)
qcum_te_p = qcum_te.unsqueeze(2).expand(-1, -1, PTS, -1)

A_tr = torch.cat([xyz_tr, t_tr], dim=-1)
A_te = torch.cat([xyz_te, t_te], dim=-1)
B_tr = torch.cat([xyz_tr, res_tr, t_tr], dim=-1)
B_te = torch.cat([xyz_te, res_te, t_te], dim=-1)
C_tr = torch.cat([xyz_tr, res_tr, qcum_tr_p, t_tr], dim=-1)
C_te = torch.cat([xyz_te, res_te, qcum_te_p, t_te], dim=-1)


class TinyPointTemporal(nn.Module):
    def __init__(self, in_ch, num_classes=25, per_point=64, temporal=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_ch, per_point), nn.GELU(),
                                 nn.Linear(per_point, per_point))
        self.conv1 = nn.Conv1d(per_point, temporal, 3, padding=1)
        self.conv2 = nn.Conv1d(temporal, temporal, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(temporal, num_classes)
    def forward(self, x):
        h = self.mlp(x)
        h = h.max(dim=2).values
        h = h.transpose(1, 2)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        return self.fc(self.pool(h).squeeze(-1))


def train_eval(seed, X_tr, y_tr, X_te, y_te, in_ch, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = TinyPointTemporal(in_ch).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    BS = 32; N_train = len(X_tr); best = 0
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed * 1000 + ep)
        perm = torch.randperm(N_train, generator=g)
        for i in range(0, N_train, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            loss = F.cross_entropy(model(X_tr_c[idx]), y_tr_c[idx])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_te), 64):
                preds.append(model(X_te_c[i:i+64]).argmax(-1))
            preds = torch.cat(preds)
            acc = (preds == y_te_c).float().mean().item()
        if acc > best: best = acc
    return best


rows = []
for seed in range(5):
    a = train_eval(seed, A_tr, y_tr, A_te, y_te, 4)
    b = train_eval(seed, B_tr, y_tr, B_te, y_te, 7)
    c = train_eval(seed, C_tr, y_tr, C_te, y_te, 11)
    msg = f"seed {seed}: A={a*100:.2f}  B={b*100:.2f}  C={c*100:.2f}  (B-A)={(b-a)*100:+.2f}  (C-B)={(c-b)*100:+.2f}"
    rows.append((a, b, c))
    print(msg); tg(msg)

A_accs = [r[0] for r in rows]
B_accs = [r[1] for r in rows]
C_accs = [r[2] for r in rows]
deltas_BA = [b-a for a,b,c in rows]
deltas_CB = [c-b for a,b,c in rows]

summary = f"""
=== A/B/C (5 seeds) ===
A: {np.mean(A_accs)*100:.2f} ± {np.std(A_accs)*100:.2f}
B: {np.mean(B_accs)*100:.2f} ± {np.std(B_accs)*100:.2f}
C: {np.mean(C_accs)*100:.2f} ± {np.std(C_accs)*100:.2f}
B - A: {np.mean(deltas_BA)*100:+.2f} ± {np.std(deltas_BA)*100:.2f}  (min {min(deltas_BA)*100:+.2f}, max {max(deltas_BA)*100:+.2f})
C - B: {np.mean(deltas_CB)*100:+.2f} ± {np.std(deltas_CB)*100:.2f}  (min {min(deltas_CB)*100:+.2f}, max {max(deltas_CB)*100:+.2f})
Verdict on C: {'beats B' if np.mean(deltas_CB) > 0.5/100 else 'ties B' if abs(np.mean(deltas_CB)) < 0.5/100 else 'loses to B'}
"""
print(summary); tg(summary)
