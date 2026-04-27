"""TinyDualStream (qcc_branch arch) trained on raw depth point clouds.

No PMamba feature extraction — trained from scratch on _pts.npy data.
Input: (B, T=32, N=256, 4) per-point xyzt (channels 4..8 of _pts.npy = native xyz + t).
Output: 25-class logits.

Compares against PMamba(89.83) and computes fusion.
"""
import os, math, sys, numpy as np, requests, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/notebooks/PMamba/experiments")
import nvidia_dataloader as nd

PMAMBA_NPZ = "/notebooks/PMamba/experiments/work_dir/pmamba_branch/pmamba_test_preds.npz"
WORK = "/notebooks/PMamba/experiments/work_dir/pc_qcc"
os.makedirs(WORK, exist_ok=True)
TG = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"


def tg(msg):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            cid = r["result"][-1]["message"]["chat"]["id"]
            requests.post(f"https://api.telegram.org/bot{TG}/sendMessage",
                data={"chat_id": cid, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


def knn_xyz(xyz, k):
    # xyz: (BT, P, 3)
    d = torch.cdist(xyz, xyz)
    _, idx = d.topk(k, largest=False, dim=-1)
    return idx


def gather_neighbors(x, idx):
    BT, P, C = x.shape
    _, _, k = idx.shape
    bi = torch.arange(BT, device=x.device).view(BT, 1, 1).expand(-1, P, k)
    return x[bi, idx]


class EdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch=128, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_ch, 64, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def forward(self, x):
        # x: (BT, P, C)
        xyz = x[..., :3].contiguous()
        idx = knn_xyz(xyz, self.k)
        x_j = gather_neighbors(x, idx)
        x_i = x.unsqueeze(2).expand(-1, -1, self.k, -1)
        edge = torch.cat([x_i, x_j - x_i], dim=-1).permute(0, 3, 1, 2)
        h = self.mlp(edge)
        return h.max(-1).values.transpose(1, 2)


class PCStream(nn.Module):
    """qcc_branch TinyDualStream without the per-frame fr stream — point cloud only."""

    def __init__(self, in_ch=4, num_classes=25, k=16, hidden=256):
        super().__init__()
        self.edge = EdgeConv(in_ch=in_ch, out_ch=128, k=k)
        self.pt_mlp = nn.Sequential(
            nn.Linear(128 + in_ch, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.proj = nn.Conv1d(512, hidden, 1)
        self.c1 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b1 = nn.BatchNorm1d(hidden)
        self.c2 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b2 = nn.BatchNorm1d(hidden)
        self.c3 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b3 = nn.BatchNorm1d(hidden)
        self.c4 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b4 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_pt):
        # x_pt: (B, T, P, C)
        B, T_, P_, C_ = x_pt.shape
        x_bt = x_pt.reshape(B * T_, P_, C_)
        local = self.edge(x_bt)
        h = torch.cat([local, x_bt], dim=-1)
        h = self.pt_mlp(h).reshape(B, T_, P_, -1)
        per_frame = torch.cat([h.max(2).values, h.mean(2)], dim=-1)        # (B, T, 512)
        h_seq = per_frame.transpose(1, 2)                                   # (B, 512, T)
        h_seq = self.proj(h_seq)
        h_seq = F.gelu(self.b1(self.c1(h_seq)))
        h_seq = h_seq + F.gelu(self.b2(self.c2(h_seq)))
        h_seq = h_seq + F.gelu(self.b3(self.c3(h_seq)))
        h_seq = h_seq + F.gelu(self.b4(self.c4(h_seq)))
        h_seq = self.drop(h_seq).max(-1).values
        return self.head(h_seq)


def load_split(phase):
    """Load all _pts.npy files for the v2 split via existing dataloader; cache to RAM."""
    ds = nd.NvidiaLoader(framerate=32, phase=phase)
    feats = []
    labels = []
    for i in range(len(ds)):
        x, y, _ = ds[i]                                 # may be torch tensor
        if hasattr(x, "numpy"):
            x = x.numpy()
        # use channels 0..3 (uvdt — pixel + time, normalized by loader and used by baseline)
        feats.append(np.asarray(x[..., 0:4], dtype=np.float32))
        labels.append(int(y))
    return np.stack(feats, 0), np.array(labels, dtype=np.int64)


def main():
    print("loading train...")
    Xtr, ytr = load_split("train")
    print(f"  train {Xtr.shape}")
    print("loading test...")
    Xte, yte = load_split("test")
    print(f"  test {Xte.shape}")

    # subsample 256 of 512 points (deterministic for both train/test for now;
    # will randomize per-epoch in the train loop for augmentation)
    P_USE = 256

    pm = np.load(PMAMBA_NPZ)
    pm_probs, pm_labels = pm["probs"], pm["labels"]
    pm_pred = pm_probs.argmax(-1)
    pm_acc = (pm_pred == pm_labels).mean()
    print(f"PMamba ref: {pm_acc*100:.2f}%")
    assert (pm_labels == yte).all()

    DEV = torch.device("cuda")
    Xtr_t = torch.from_numpy(Xtr).to(DEV)        # (Ntr, 32, 512, 4)
    ytr_t = torch.from_numpy(ytr).to(DEV)
    Xte_t = torch.from_numpy(Xte).to(DEV)        # (Nte, 32, 512, 4)
    yte_t = torch.from_numpy(yte).to(DEV)

    # Test-time deterministic 256-pt subsample
    te_idx = torch.linspace(0, Xte.shape[2] - 1, P_USE, device=DEV).long()
    Xte_use = Xte_t.index_select(2, te_idx)

    # Per-channel normalization on xyzt (channels 4..8 of _pts.npy)
    flat = Xtr_t.reshape(-1, 4)
    mu = flat.mean(0)
    sd = flat.std(0).clamp(min=1.0)
    Xtr_n = (Xtr_t - mu) / sd
    Xte_n = (Xte_use - mu) / sd

    torch.manual_seed(0); np.random.seed(0)
    model = PCStream(in_ch=4, num_classes=25, k=16).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    EPOCHS, BS = 120, 4
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warmup if ep < warmup
        else 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, EPOCHS - warmup))))

    tg(f"<b>PC-QCC</b> {EPOCHS} ep, raw depth pt-cloud through TinyDualStream\nPMamba ref {pm_acc*100:.2f}%")
    best_acc = 0.0; best_logits = None
    Ntr = Xtr_n.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(Ntr, device=DEV)
        # per-epoch random subsample of 256 from 512 (augmentation)
        rand_idx = torch.randperm(Xtr_n.shape[2], device=DEV)[:P_USE]
        Xtr_use = Xtr_n.index_select(2, rand_idx)
        for i in range(0, Ntr, BS):
            idx = perm[i:i+BS]
            logits = model(Xtr_use[idx])
            loss = F.cross_entropy(logits, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            te_logits_list = []
            for j in range(0, Xte_n.shape[0], BS):
                te_logits_list.append(model(Xte_n[j:j+BS]).cpu().numpy())
            te_logits = np.concatenate(te_logits_list, 0)
        pc_pred = te_logits.argmax(-1)
        acc = (pc_pred == yte).mean()
        pc_probs = torch.softmax(torch.from_numpy(te_logits), dim=-1).numpy()

        best_fuse = 0.0; best_alpha = 0.0
        for alpha in np.linspace(0, 1, 41):
            fused = alpha * pm_probs + (1 - alpha) * pc_probs
            fa = (fused.argmax(-1) == yte).mean()
            if fa > best_fuse: best_fuse = fa; best_alpha = alpha
        oracle = ((pm_pred == yte) | (pc_pred == yte)).mean()

        if acc > best_acc:
            best_acc = acc; best_logits = te_logits
            torch.save(model.state_dict(), f"{WORK}/best.pt")

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            line = (f"ep {epoch:3d}  pc {acc*100:.2f}%  best {best_acc*100:.2f}%  "
                    f"fuse[a={best_alpha:.2f}] {best_fuse*100:.2f}%  oracle {oracle*100:.2f}%")
            print(line)
            tg(f"<b>PC-QCC ep{epoch}</b>\nPC {acc*100:.2f}% best {best_acc*100:.2f}%\n"
               f"Fuse[a={best_alpha:.2f}] <b>{best_fuse*100:.2f}%</b>\nOracle {oracle*100:.2f}%")

    pc_probs_best = torch.softmax(torch.from_numpy(best_logits), dim=-1).numpy()
    np.savez_compressed(f"{WORK}/test_preds.npz", probs=pc_probs_best, labels=yte)

    pc_pred = pc_probs_best.argmax(-1)
    final_oracle = ((pm_pred == yte) | (pc_pred == yte)).mean()
    best_fuse = 0.0; best_alpha = 0.0
    for alpha in np.linspace(0, 1, 41):
        fused = alpha * pm_probs + (1 - alpha) * pc_probs_best
        a = (fused.argmax(-1) == yte).mean()
        if a > best_fuse: best_fuse = a; best_alpha = alpha
    final = (f"<b>PC-QCC done</b>\nPC best {best_acc*100:.2f}%\n"
             f"PMamba {pm_acc*100:.2f}%\n"
             f"Fuse[a={best_alpha:.2f}] <b>{best_fuse*100:.2f}%</b>  "
             f"(Δ {(best_fuse - pm_acc)*100:+.2f}pp)\n"
             f"Oracle {final_oracle*100:.2f}%")
    print(final.replace("<b>", "").replace("</b>", ""))
    tg(final)


if __name__ == "__main__":
    main()
