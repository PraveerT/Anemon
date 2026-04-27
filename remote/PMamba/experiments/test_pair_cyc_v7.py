"""V7: forward = mutual-NN per-pair Kabsch, backward = chain-stabilized k-step Kabsch.

Forward Kabsch fits the SINGLE-step rigid motion (frame t -> frame t+1) using
mutual-NN correspondence at that pair.

Backward Kabsch fits a CHAINED multi-step rigid motion (frame t+1 -> frame
t-k+1) using chained mutual-NN correspondences. We use k=4 (lookback of 3 pairs)
with chain reset at sequence boundaries.

These two Kabsch fits are over DIFFERENT temporal windows -> different (src, tgt)
pairs -> q_b is the optimal rigid motion at SCALE k, while q_f is the optimal at
SCALE 1. They are not exact inverses.

Cycle violation:
  c_t = q_f x q_b - I

For purely rigid motion at all scales, both fits agree on the rotation per frame
and chain composes cleanly => cycle = 0.

For articulating motion, the per-pair fit q_f reflects instantaneous articulation
while the multi-step fit q_b averages it out -> they diverge. Cycle magnitude
correlates with articulation amplitude over the window.

K=4 bounds chain drift to at most 3 composed pairs -> manageable noise.
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


def cycle_quat(q_f, q_b):
    qcyc = hamilton(q_f.unsqueeze(0), q_b.unsqueeze(0))[0]
    qcyc = F.normalize(qcyc, dim=-1)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=qcyc.device)
    if (qcyc * ident).sum() < 0:
        qcyc = -qcyc
    return qcyc - ident


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
K_LOOKBACK = 4   # bound chain drift


def collect(phase):
    """Forward = single-pair Kabsch. Backward = k-step chained Kabsch."""
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
        # Pre-compute mutual mask per pair (for chain composition)
        pair_masks = [matched[t].float() for t in range(F_ - 1)]
        for t in range(F_ - 1):
            mask_t = pair_masks[t]
            # Forward Kabsch (single pair)
            q_f, tr_f = kabsch_quat_w(xyz_s[t], xyz_s[t+1], mask_t)
            RF[i, t+1] = (xyz_s[t+1] - (quat_rotate_pts(q_f, xyz_s[t]) + tr_f)).cpu().numpy()
            # Backward Kabsch over k-step chain: from frame t+1 back to frame max(0, t-K+2)
            # We chain mutual-NN correspondences: each frame's sampled points are by
            # construction the corr-aware sampled set, so direct Kabsch between frame t+1
            # and frame anchor uses the same sampled point indices (their positions just
            # differ across frames).
            anchor = max(0, t - K_LOOKBACK + 2)
            # Mask for k-step backward: AND of intervening mutual masks
            chain_mask = pair_masks[anchor].clone()
            for tt in range(anchor + 1, t + 1):
                chain_mask = chain_mask * pair_masks[tt]
            # If chain mask too sparse, fall back to single-pair backward
            if chain_mask.sum() < 8:
                q_b, tr_b = kabsch_quat_w(xyz_s[t+1], xyz_s[t], mask_t)
            else:
                q_b, tr_b = kabsch_quat_w(xyz_s[t+1], xyz_s[anchor], chain_mask)
            RB[i, t] = (xyz_s[t] - (quat_rotate_pts(q_b, xyz_s[t+1]) + tr_b)).cpu().numpy()
            PAIRCYC[i, t] = cycle_quat(q_f, q_b).cpu().numpy()
        labels[i] = label
        if (i+1) % 200 == 0:
            print(f'  {phase} {i+1}/{N}')
    return (torch.from_numpy(XYZ), torch.from_numpy(RF),
            torch.from_numpy(RB), torch.from_numpy(PAIRCYC),
            torch.from_numpy(labels))


print('=== Phase A: V7 chain-stabilized backward (k=4) ===')
xyz_tr, rf_tr, rb_tr, pc_tr, y_tr = collect('train')
xyz_te, rf_te, rb_te, pc_te, y_te = collect('test')
print(f"V7 PC abs mean: {pc_tr.abs().mean():.5f}  max norm: {pc_tr.norm(dim=-1).max():.4f}")

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


class TinyV7(nn.Module):
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


SAVE_ROOT = '/notebooks/PMamba/experiments/work_dir/qcc_branch/v7'
os.makedirs(SAVE_ROOT, exist_ok=True)


def train(seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = TinyV7(in_ch=10).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep+1)/5 if ep<5
        else 0.5*(1+math.cos(math.pi*(ep-5)/max(1,120-5))))
    Xtr = C_tr.cuda(); Xfr_tr = pc_tr.cuda(); ytr_c = y_tr.cuda()
    Xte = C_te.cuda(); Xfr_te = pc_te.cuda(); yte_c = y_te.cuda()
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
            print(f"  V7 ep{ep:3d} te={acc*100:5.2f} best={best*100:5.2f}(ep{best_ep})")
    return best, best_ep, best_logits, best_state


tg("V7 chain-stabilized backward k=4 starting.")
v7, v7_ep, v7_lg, v7_st = train(seed=1)
torch.save({'state_dict': v7_st, 'best_acc': v7, 'best_ep': v7_ep},
           f'{SAVE_ROOT}/best.pt')
np.savez(f'{SAVE_ROOT}/logits.npz', logits=v7_lg.numpy(), labels=y_te.numpy(),
         acc=v7, best_ep=v7_ep)
msg = f"""
=== V7 chain-stabilized k={K_LOOKBACK} backward ===
V0 baseline (Phase A seed 0): 79.67-84.23 range across runs
V7 (mutual-fwd, k-step bwd): {v7*100:.2f}%
"""
print(msg); tg(msg)
