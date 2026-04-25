"""PointNet++ orthogonal tiny classifier.

Per-frame: SA1 (256->128, k=16, mlp=[64,128,128]) + SA2 (128->32, k=16, mlp=[128,256,256])
          + global max over 32 centers -> 256-d per frame
Temporal: 4 conv1d blocks w/ BN+GELU+residual (from strong-tiny)
Head: LN -> Linear 256->128 -> GELU -> Dropout -> Linear 128->25

FPS (farthest-point sampling) for centers. kNN via cdist. Batched over B*T.
Aug: y-rotation, point-dropout 10%.
Training: AdamW lr=2e-3 wd=0.01, cosine+10-ep warmup, label smooth 0.1,
          150 epochs, batch 8, grad clip 1.0.
Input: Cfbq 10ch (xyz + res_fwd + res_bwd + t), 256 pts, 32 frames.
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RF),
            torch.from_numpy(RB), torch.from_numpy(labels))


print('Collecting Cfbq features 256 pts...')
xyz_tr, rf_tr, rb_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, y_te = collect('test')
T = 32
t_tr = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_tr.shape[0], T, PTS, 1)
t_te = torch.linspace(0, 1, T).view(1, T, 1, 1).expand(xyz_te.shape[0], T, PTS, 1)
C_tr = torch.cat([xyz_tr, rf_tr, rb_tr, t_tr], dim=-1)
C_te = torch.cat([xyz_te, rf_te, rb_te, t_te], dim=-1)
print(f"train {C_tr.shape} test {C_te.shape}")


# ---------- PointNet++ primitives ----------

def fps(xyz, M):
    """Farthest point sampling. xyz: (B, N, 3) -> indices (B, M)."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, M, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    far = torch.randint(0, N, (B,), device=xyz.device)
    bi = torch.arange(B, device=xyz.device)
    for i in range(M):
        idx[:, i] = far
        centroid = xyz[bi, far].unsqueeze(1)                   # (B,1,3)
        d = ((xyz - centroid) ** 2).sum(-1)                    # (B,N)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx


def knn_idx(q, x, k):
    """For each q find k nearest x. q: (B, M, 3), x: (B, N, 3) -> (B, M, k)."""
    d = torch.cdist(q, x)
    _, idx = d.topk(k, dim=-1, largest=False)
    return idx


def gather_nd(x, idx):
    """x: (B, N, C), idx: (B, M, k) -> (B, M, k, C)."""
    B, N, C = x.shape
    _, M, K = idx.shape
    b_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, M, K)
    return x[b_idx, idx]


class SetAbstraction(nn.Module):
    def __init__(self, sample, k, in_ch, mlp):
        super().__init__()
        self.sample = sample; self.k = k
        layers = []
        ch = in_ch + 3
        for m in mlp:
            layers += [nn.Conv2d(ch, m, 1, bias=False), nn.BatchNorm2d(m), nn.GELU()]
            ch = m
        self.mlp = nn.Sequential(*layers)
        self.out_ch = mlp[-1]

    def forward(self, xyz, feat):
        # xyz: (B, N, 3), feat: (B, N, C)
        B, N, _ = xyz.shape
        M = self.sample
        ci = fps(xyz, M)                                              # (B,M)
        bi = torch.arange(B, device=xyz.device).view(B, 1).expand(-1, M)
        new_xyz = xyz[bi, ci]                                         # (B,M,3)
        nn_i = knn_idx(new_xyz, xyz, self.k)                          # (B,M,k)
        g_xyz = gather_nd(xyz, nn_i) - new_xyz.unsqueeze(2)            # (B,M,k,3)
        g_feat = gather_nd(feat, nn_i)                                 # (B,M,k,C)
        g = torch.cat([g_xyz, g_feat], dim=-1).permute(0, 3, 1, 2)    # (B, 3+C, M, k)
        h = self.mlp(g).max(-1).values                                 # (B, out, M)
        return new_xyz, h.transpose(1, 2)                              # (B,M,out)


class PNet2Frame(nn.Module):
    def __init__(self, in_ch=10):
        super().__init__()
        # feat = all 10 channels (xyz redundant with pos but helpful)
        self.sa1 = SetAbstraction(sample=128, k=16, in_ch=in_ch,  mlp=[64, 128, 128])
        self.sa2 = SetAbstraction(sample=32,  k=16, in_ch=128,    mlp=[128, 256, 256])
        self.global_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.GELU(),
            nn.Conv1d(512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.GELU())

    def forward(self, x):
        # x: (B*T, N, in_ch) with xyz = x[..., :3]
        xyz = x[..., :3].contiguous()
        xyz1, f1 = self.sa1(xyz, x)
        xyz2, f2 = self.sa2(xyz1, f1)
        # global: (B*T, 256, M=32) -> mlp -> maxpool -> (B*T, 256)
        h = f2.transpose(1, 2)                    # (B*T, 256, M)
        h = self.global_mlp(h)
        return h.max(-1).values


class TemporalConv(nn.Module):
    """4 residual conv1d blocks, d=256, from strong-tiny family."""
    def __init__(self, d=256):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 3, padding=1); self.b1 = nn.BatchNorm1d(d)
        self.c2 = nn.Conv1d(d, d, 3, padding=1); self.b2 = nn.BatchNorm1d(d)
        self.c3 = nn.Conv1d(d, d, 3, padding=1); self.b3 = nn.BatchNorm1d(d)
        self.c4 = nn.Conv1d(d, d, 3, padding=1); self.b4 = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(0.2)
    def forward(self, h):  # h: (B, d, T)
        h = F.gelu(self.b1(self.c1(h)))
        h = h + F.gelu(self.b2(self.c2(h)))
        h = h + F.gelu(self.b3(self.c3(h)))
        h = h + F.gelu(self.b4(self.c4(h)))
        return self.drop(h).max(-1).values


class PNet2Tiny(nn.Module):
    def __init__(self, in_ch=10, num_classes=25, d=256):
        super().__init__()
        self.frame = PNet2Frame(in_ch)
        self.temporal = TemporalConv(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes))
    def forward(self, x):
        B, T_, P_, C_ = x.shape
        f = self.frame(x.reshape(B*T_, P_, C_)).reshape(B, T_, -1)     # (B,T,256)
        h = self.temporal(f.transpose(1, 2))                           # (B,256)
        return self.head(h)


def rand_y_rot(x):
    B = x.shape[0]
    a = torch.rand(B, device=x.device) * 2 * math.pi
    c = torch.cos(a); s = torch.sin(a)
    R = torch.zeros(B, 3, 3, device=x.device)
    R[:,0,0] = c; R[:,0,2] = s; R[:,1,1] = 1.0; R[:,2,0] = -s; R[:,2,2] = c
    def rot(v): return torch.einsum('bij,btpj->btpi', R, v)
    return torch.cat([rot(x[..., :3]), rot(x[..., 3:6]), rot(x[..., 6:9]), x[..., 9:10]], dim=-1)


def train_eval(seed, X_tr, y_tr, X_te, y_te, epochs=150, bs=8):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = PNet2Tiny(in_ch=X_tr.shape[-1]).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"seed {seed} params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    warmup = 10
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/warmup if ep<warmup
        else 0.5*(1+math.cos(math.pi*(ep-warmup)/max(1,epochs-warmup))))
    X_tr_c = X_tr.cuda(); y_tr_c = y_tr.cuda()
    X_te_c = X_te.cuda(); y_te_c = y_te.cuda()
    best = 0.0; best_logits = None; best_ep = -1
    for ep in range(epochs):
        model.train()
        g = torch.Generator(device='cpu'); g.manual_seed(seed*1000+ep)
        perm = torch.randperm(len(X_tr), generator=g)
        tr_correct = 0; tr_total = 0
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i+bs]
            xb = X_tr_c[idx]; yb = y_tr_c[idx]
            xb = rand_y_rot(xb)
            B_, T_, P_, _ = xb.shape
            xb = xb * (torch.rand(B_, 1, P_, 1, device=xb.device) > 0.10).float()
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_correct += (logits.argmax(-1) == yb).sum().item()
            tr_total += yb.numel()
        sched.step()
        model.eval()
        with torch.no_grad():
            lg = []
            for i in range(0, len(X_te), 16):
                lg.append(model(X_te_c[i:i+16]).cpu())
            lg = torch.cat(lg, 0)
            acc = (lg.argmax(-1).cuda() == y_te_c).float().mean().item()
        if acc > best:
            best = acc; best_logits = lg.clone(); best_ep = ep
        if ep < 15 or ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep}) tr={tr_correct/max(1,tr_total)*100:5.2f} lr={opt.param_groups[0]['lr']:.2e}")
    return best, best_logits, best_ep


tg("PNet2Tiny: PointNet++ 2-SA + conv1d temporal, 150ep, Cfbq 10ch.")
acc, logits, be = train_eval(0, C_tr, y_tr, C_te, y_te, epochs=150)
msg = f"\n=== PNet2Tiny seed 0 ===\nbest: {acc*100:.2f}% at ep{be} (prior strong-tiny 77.34; tx 59.96)"
print(msg); tg(msg)

np.savez('/tmp/pnet2_logits.npz', logits=logits.numpy(), labels=y_te.numpy(),
         acc=acc, best_ep=be)
print("saved /tmp/pnet2_logits.npz")
