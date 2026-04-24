"""A/B/C_fb/D_ms/E_fd/E_ms with DUAL-STREAM architecture.

Per-point stream: per-point feats -> MLP -> max-pool over P -> per-frame h_pt
Per-frame stream: quaternion/cycle broadcast -> small MLP -> per-frame h_fr
Concat(h_pt, h_fr) -> temporal 1D-conv -> classify

Now broadcast features (quaternion) enter AFTER the point pool — no capacity
competition with per-point info. Fair comparison.

A, B, C_fb, D_ms: per_frame = None (only per-point stream active)
E_fd: per_frame = q_fwd (4-ch)
E_ms: per_frame = cycle_err (4-ch)
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
    aw,ax,ay,az = a[0],a[1],a[2],a[3]; bw,bx,by,bz = b[0],b[1],b[2],b[3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def kabsch_Rt(src, tgt, mask):
    device = src.device
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


def kabsch_pair(src, tgt, mask):
    R, tr = kabsch_Rt(src.unsqueeze(0), tgt.unsqueeze(0), mask.unsqueeze(0))
    rigid = torch.bmm(R, src.unsqueeze(0).transpose(-1, -2)).transpose(-1, -2) + tr.unsqueeze(1)
    res = (tgt.unsqueeze(0) - rigid)[0]
    q = rot_to_quat(R[0])
    return q, res


PTS = 128
tg("DUAL-STREAM fair comparison: quaternion enters after point-pool. 5 seeds.")
print('Collecting...')


def collect(phase):
    loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
        framerate=32, phase=phase, return_correspondence=True,
    )
    N = len(loader)
    XYZ = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    R1_F = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    R1_B = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    R2 = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    R4 = np.zeros((N, 32, PTS, 3), dtype=np.float32)
    Q_F = np.zeros((N, 32, 4), dtype=np.float32)
    CYC_Q = np.zeros((N, 32, 4), dtype=np.float32)
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
        q_pairs = []
        for t in range(F_ - 1):
            qf, resf = kabsch_pair(xyz_samp[t], xyz_samp[t+1], matched[t])
            R1_F[i, t+1] = resf.cpu().numpy()
            Q_F[i, t+1] = qf.cpu().numpy()
            q_pairs.append(qf)
            qb, resb = kabsch_pair(xyz_samp[t+1], xyz_samp[t], matched[t])
            R1_B[i, t] = resb.cpu().numpy()
        for t in range(2, F_):
            mchain = matched[t-2] & matched[t-1]
            q2, res2 = kabsch_pair(xyz_samp[t-2], xyz_samp[t], mchain)
            R2[i, t] = res2.cpu().numpy()
            q_comp = hamilton(q_pairs[t-2], q_pairs[t-1])
            q_comp = F.normalize(q_comp, dim=-1)
            if q_comp[0] < 0: q_comp = -q_comp
            if (q2 * q_comp).sum() < 0: q_comp = -q_comp
            CYC_Q[i, t] = (q2 - q_comp).cpu().numpy()
        for t in range(4, F_):
            mchain = matched[t-4] & matched[t-3] & matched[t-2] & matched[t-1]
            _, res4 = kabsch_pair(xyz_samp[t-4], xyz_samp[t], mchain)
            R4[i, t] = res4.cpu().numpy()
        labels[i] = label
        if (i+1) % 300 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(R1_F), torch.from_numpy(R1_B),
            torch.from_numpy(R2), torch.from_numpy(R4),
            torch.from_numpy(Q_F), torch.from_numpy(CYC_Q),
            torch.from_numpy(labels))


xyz_tr, r1f_tr, r1b_tr, r2_tr, r4_tr, qf_tr, cyc_tr, y_tr = collect('train')
xyz_te, r1f_te, r1b_te, r2_te, r4_te, qf_te, cyc_te, y_te = collect('test')
tg("Collection done. Training 6 variants × 5 seeds (dual-stream).")

T = xyz_tr.shape[1]
t_tr = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0],  T, PTS, 1)

# Per-point inputs (same as ABCD)
A_ppt_tr = torch.cat([xyz_tr, t_tr], dim=-1)
A_ppt_te = torch.cat([xyz_te, t_te], dim=-1)
B_ppt_tr = torch.cat([xyz_tr, r1f_tr, t_tr], dim=-1)
B_ppt_te = torch.cat([xyz_te, r1f_te, t_te], dim=-1)
Cfb_ppt_tr = torch.cat([xyz_tr, r1f_tr, r1b_tr, t_tr], dim=-1)
Cfb_ppt_te = torch.cat([xyz_te, r1f_te, r1b_te, t_te], dim=-1)
Dms_ppt_tr = torch.cat([xyz_tr, r1f_tr, r2_tr, r4_tr, t_tr], dim=-1)
Dms_ppt_te = torch.cat([xyz_te, r1f_te, r2_te, r4_te, t_te], dim=-1)
# E variants: per-point = Cfb / B, per-frame = qf / cyc
Efd_ppt_tr = Cfb_ppt_tr; Efd_ppt_te = Cfb_ppt_te
Ems_ppt_tr = B_ppt_tr; Ems_ppt_te = B_ppt_te


class DualStreamModel(nn.Module):
    def __init__(self, in_ch_pt, in_ch_fr, num_classes=25, per_point=64, fr_dim=32, temporal=128):
        super().__init__()
        self.mlp_pt = nn.Sequential(nn.Linear(in_ch_pt, per_point), nn.GELU(),
                                    nn.Linear(per_point, per_point))
        self.has_fr = in_ch_fr > 0
        if self.has_fr:
            self.mlp_fr = nn.Sequential(nn.Linear(in_ch_fr, fr_dim), nn.GELU(),
                                        nn.Linear(fr_dim, fr_dim))
            fused_dim = per_point + fr_dim
        else:
            fused_dim = per_point
        self.conv1 = nn.Conv1d(fused_dim, temporal, 3, padding=1)
        self.conv2 = nn.Conv1d(temporal, temporal, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(temporal, num_classes)

    def forward(self, x_pt, x_fr=None):
        # x_pt: (B, T, P, C_pt); x_fr: (B, T, C_fr) or None
        h_pt = self.mlp_pt(x_pt).max(dim=2).values  # (B, T, per_point)
        if self.has_fr and x_fr is not None:
            h_fr = self.mlp_fr(x_fr)                  # (B, T, fr_dim)
            h = torch.cat([h_pt, h_fr], dim=-1)
        else:
            h = h_pt
        h = h.transpose(1, 2)
        h = F.gelu(self.conv1(h)); h = F.gelu(self.conv2(h))
        return self.fc(self.pool(h).squeeze(-1))


def train_eval(seed, X_ppt_tr, y_tr, X_ppt_te, y_te, in_ch_pt,
               X_fr_tr=None, X_fr_te=None, in_ch_fr=0, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = DualStreamModel(in_ch_pt, in_ch_fr).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X_ppt_tr_c = X_ppt_tr.cuda(); y_tr_c = y_tr.cuda()
    X_ppt_te_c = X_ppt_te.cuda(); y_te_c = y_te.cuda()
    X_fr_tr_c = X_fr_tr.cuda() if X_fr_tr is not None else None
    X_fr_te_c = X_fr_te.cuda() if X_fr_te is not None else None
    BS = 32; N_train = len(X_ppt_tr); best = 0
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed * 1000 + ep)
        perm = torch.randperm(N_train, generator=g)
        for i in range(0, N_train, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            xfr = X_fr_tr_c[idx] if X_fr_tr_c is not None else None
            loss = F.cross_entropy(model(X_ppt_tr_c[idx], xfr), y_tr_c[idx])
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_ppt_te), 64):
                xfr = X_fr_te_c[i:i+64] if X_fr_te_c is not None else None
                preds.append(model(X_ppt_te_c[i:i+64], xfr).argmax(-1))
            preds = torch.cat(preds)
            acc = (preds == y_te_c).float().mean().item()
        if acc > best: best = acc
    return best


rows = []
for seed in range(5):
    a = train_eval(seed, A_ppt_tr, y_tr, A_ppt_te, y_te, 4)
    b = train_eval(seed, B_ppt_tr, y_tr, B_ppt_te, y_te, 7)
    c = train_eval(seed, Cfb_ppt_tr, y_tr, Cfb_ppt_te, y_te, 10)
    d = train_eval(seed, Dms_ppt_tr, y_tr, Dms_ppt_te, y_te, 13)
    efd = train_eval(seed, Efd_ppt_tr, y_tr, Efd_ppt_te, y_te, 10,
                     X_fr_tr=qf_tr, X_fr_te=qf_te, in_ch_fr=4)
    ems = train_eval(seed, Ems_ppt_tr, y_tr, Ems_ppt_te, y_te, 7,
                     X_fr_tr=cyc_tr, X_fr_te=cyc_te, in_ch_fr=4)
    msg = f"seed {seed}: A={a*100:.2f} B={b*100:.2f} Cfb={c*100:.2f} Dms={d*100:.2f} Efd={efd*100:.2f} Ems={ems*100:.2f}"
    rows.append((a, b, c, d, efd, ems)); print(msg); tg(msg)

A_ = [r[0] for r in rows]; B_ = [r[1] for r in rows]
C_ = [r[2] for r in rows]; D_ = [r[3] for r in rows]
Ef = [r[4] for r in rows]; Em = [r[5] for r in rows]

summary = f"""
=== DUAL-STREAM 5 seeds (quaternion AFTER point-pool) ===
A:     {np.mean(A_)*100:.2f} ± {np.std(A_)*100:.2f}
B:     {np.mean(B_)*100:.2f} ± {np.std(B_)*100:.2f}   (B-A: {(np.mean(B_)-np.mean(A_))*100:+.2f})
Cfb:   {np.mean(C_)*100:.2f} ± {np.std(C_)*100:.2f}   (Cfb-B: {(np.mean(C_)-np.mean(B_))*100:+.2f})
Dms:   {np.mean(D_)*100:.2f} ± {np.std(D_)*100:.2f}   (Dms-B: {(np.mean(D_)-np.mean(B_))*100:+.2f})
Efd:   {np.mean(Ef)*100:.2f} ± {np.std(Ef)*100:.2f}   (Efd-Cfb: {(np.mean(Ef)-np.mean(C_))*100:+.2f})
Ems:   {np.mean(Em)*100:.2f} ± {np.std(Em)*100:.2f}   (Ems-B: {(np.mean(Em)-np.mean(B_))*100:+.2f})
"""
print(summary); tg(summary)
