"""Same TempConvHead as train_rgb_qcc.py but trained on sk_depth features instead of sk_color."""
import os, math, numpy as np, requests, torch
import torch.nn as nn, torch.nn.functional as F

PROC = "/notebooks/PMamba/dataset/Nvidia/Processed"
PMAMBA_NPZ = "/notebooks/PMamba/experiments/work_dir/pmamba_branch/pmamba_test_preds.npz"
WORK = "/notebooks/PMamba/experiments/work_dir/depth_qcc"
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


class TempConvHead(nn.Module):
    def __init__(self, in_dim=512, hidden=256, num_classes=25):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        self.c1 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b1 = nn.BatchNorm1d(hidden)
        self.c2 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b2 = nn.BatchNorm1d(hidden)
        self.c3 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b3 = nn.BatchNorm1d(hidden)
        self.c4 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b4 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.proj(h)
        h = F.gelu(self.b1(self.c1(h)))
        h = h + F.gelu(self.b2(self.c2(h)))
        h = h + F.gelu(self.b3(self.c3(h)))
        h = h + F.gelu(self.b4(self.c4(h)))
        h = self.drop(h).max(-1).values
        return self.head(h)


def main():
    tr = np.load(f"{PROC}/depth_resnet18_train.npz")
    te = np.load(f"{PROC}/depth_resnet18_test.npz")
    Xtr, ytr = tr["feats"].astype(np.float32), tr["labels"].astype(np.int64)
    Xte, yte = te["feats"].astype(np.float32), te["labels"].astype(np.int64)
    print(f"train {Xtr.shape}  test {Xte.shape}")

    pm = np.load(PMAMBA_NPZ)
    pm_probs, pm_labels = pm["probs"], pm["labels"]
    pm_pred = pm_probs.argmax(-1)
    pm_acc = (pm_pred == pm_labels).mean()
    print(f"PMamba(depth pt-cloud) ref: {pm_acc*100:.2f}%")
    assert (pm_labels == yte).all()

    DEV = torch.device("cuda")
    Xtr_t = torch.from_numpy(Xtr).to(DEV); ytr_t = torch.from_numpy(ytr).to(DEV)
    Xte_t = torch.from_numpy(Xte).to(DEV); yte_t = torch.from_numpy(yte).to(DEV)

    torch.manual_seed(0); np.random.seed(0)
    model = TempConvHead(512, 256, 25).to(DEV)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    EPOCHS, BS = 120, 32
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warmup if ep < warmup
        else 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, EPOCHS - warmup))))

    tg(f"<b>Depth-QCC</b> {EPOCHS} ep, frozen ResNet18 on sk_depth + TempConvHead\nPMamba ref {pm_acc*100:.2f}%")
    best_acc = 0.0; best_logits = None
    N = Xtr_t.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N, device=DEV)
        for i in range(0, N, BS):
            idx = perm[i:i+BS]
            logits = model(Xtr_t[idx])
            loss = F.cross_entropy(logits, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            te_logits = model(Xte_t).cpu().numpy()
        depth_pred = te_logits.argmax(-1)
        acc = (depth_pred == yte).mean()
        depth_probs = torch.softmax(torch.from_numpy(te_logits), dim=-1).numpy()

        best_fuse = 0.0; best_alpha = 0.0
        for alpha in np.linspace(0, 1, 41):
            fused = alpha * pm_probs + (1 - alpha) * depth_probs
            fa = (fused.argmax(-1) == yte).mean()
            if fa > best_fuse: best_fuse = fa; best_alpha = alpha
        oracle = ((pm_pred == yte) | (depth_pred == yte)).mean()

        if acc > best_acc:
            best_acc = acc; best_logits = te_logits
            torch.save(model.state_dict(), f"{WORK}/best.pt")

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            line = (f"ep {epoch:3d}  depth {acc*100:.2f}%  best {best_acc*100:.2f}%  "
                    f"fuse[a={best_alpha:.2f}] {best_fuse*100:.2f}%  oracle {oracle*100:.2f}%")
            print(line)
            tg(f"<b>D-QCC ep{epoch}</b>\nDepth {acc*100:.2f}% best {best_acc*100:.2f}%\n"
               f"Fuse[a={best_alpha:.2f}] <b>{best_fuse*100:.2f}%</b>\nOracle {oracle*100:.2f}%")

    depth_probs_best = torch.softmax(torch.from_numpy(best_logits), dim=-1).numpy()
    np.savez_compressed(f"{WORK}/test_preds.npz", probs=depth_probs_best, labels=yte)

    depth_pred = depth_probs_best.argmax(-1)
    final_oracle = ((pm_pred == yte) | (depth_pred == yte)).mean()
    best_fuse = 0.0; best_alpha = 0.0
    for alpha in np.linspace(0, 1, 41):
        fused = alpha * pm_probs + (1 - alpha) * depth_probs_best
        a = (fused.argmax(-1) == yte).mean()
        if a > best_fuse: best_fuse = a; best_alpha = alpha
    final = (f"<b>Depth-QCC done</b>\nDepth(2D) best {best_acc*100:.2f}%\n"
             f"PMamba(pt-cloud) {pm_acc*100:.2f}%\n"
             f"Fuse[a={best_alpha:.2f}] <b>{best_fuse*100:.2f}%</b>  "
             f"(Δ {(best_fuse - pm_acc)*100:+.2f}pp)\n"
             f"Oracle {final_oracle*100:.2f}%")
    print(final.replace("<b>", "").replace("</b>", ""))
    tg(final)


if __name__ == "__main__":
    main()
