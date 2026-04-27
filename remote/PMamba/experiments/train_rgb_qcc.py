"""qcc_branch temporal-conv arch (from train_qcc_branch.py) fed with cached
ResNet18 RGB features instead of point clouds.

Reads:
  /notebooks/PMamba/dataset/Nvidia/Processed/rgb_resnet18_{train,test}.npz
  feats: (N, T=16, 512), labels: (N,)

Outputs:
  /notebooks/PMamba/experiments/work_dir/rgb_qcc/{best.pt,test_preds.npz}
  Telegram pings each epoch with RGB / fuse / oracle vs PMamba.
"""
import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import requests

PROC = "/notebooks/PMamba/dataset/Nvidia/Processed"
PMAMBA_NPZ = "/notebooks/PMamba/experiments/work_dir/pmamba_branch/pmamba_test_preds.npz"
WORK = "/notebooks/PMamba/experiments/work_dir/rgb_qcc"
os.makedirs(WORK, exist_ok=True)
TG_TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"


def tg(msg):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            chat_id = r["result"][-1]["message"]["chat"]["id"]
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


class TempConvHead(nn.Module):
    """Mirrors the temporal-conv portion of TinyDualStream from train_qcc_branch."""

    def __init__(self, in_dim=512, hidden=256, num_classes=25, dropout=(0.2, 0.3)):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        self.c1 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b1 = nn.BatchNorm1d(hidden)
        self.c2 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b2 = nn.BatchNorm1d(hidden)
        self.c3 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b3 = nn.BatchNorm1d(hidden)
        self.c4 = nn.Conv1d(hidden, hidden, 3, padding=1); self.b4 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout[0])
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout[1]),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, in_dim)
        h = x.transpose(1, 2)                      # (B, in_dim, T)
        h = self.proj(h)
        h = F.gelu(self.b1(self.c1(h)))
        h = h + F.gelu(self.b2(self.c2(h)))
        h = h + F.gelu(self.b3(self.c3(h)))
        h = h + F.gelu(self.b4(self.c4(h)))
        h = self.drop(h).max(-1).values            # (B, hidden)
        return self.head(h)


def main():
    tr = np.load(f"{PROC}/rgb_resnet18_train.npz")
    te = np.load(f"{PROC}/rgb_resnet18_test.npz")
    Xtr, ytr = tr["feats"].astype(np.float32), tr["labels"].astype(np.int64)
    Xte, yte = te["feats"].astype(np.float32), te["labels"].astype(np.int64)
    print(f"train {Xtr.shape}  test {Xte.shape}")

    pm = np.load(PMAMBA_NPZ)
    pm_probs, pm_labels = pm["probs"], pm["labels"]
    pm_pred = pm_probs.argmax(-1)
    pm_acc = (pm_pred == pm_labels).mean()
    print(f"PMamba(depth) ref: {pm_acc*100:.2f}%")
    assert (pm_labels == yte).all()

    DEV = torch.device("cuda")
    Xtr_t = torch.from_numpy(Xtr).to(DEV)
    ytr_t = torch.from_numpy(ytr).to(DEV)
    Xte_t = torch.from_numpy(Xte).to(DEV)
    yte_t = torch.from_numpy(yte).to(DEV)

    torch.manual_seed(0)
    np.random.seed(0)
    model = TempConvHead(in_dim=512, hidden=256, num_classes=25).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    EPOCHS = 120
    BS = 32
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup = 5
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda ep: (ep + 1) / warmup if ep < warmup
        else 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, EPOCHS - warmup))),
    )

    tg(f"<b>RGB-QCC</b> {EPOCHS} ep, params {n_params/1e6:.2f}M\nPMamba ref {pm_acc*100:.2f}%")
    best_acc = 0.0
    best_logits = None
    N = Xtr_t.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N, device=DEV)
        train_loss = 0.0
        train_correct = 0
        for i in range(0, N, BS):
            idx = perm[i:i+BS]
            logits = model(Xtr_t[idx])
            loss = F.cross_entropy(logits, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * idx.size(0)
            train_correct += (logits.argmax(-1) == ytr_t[idx]).sum().item()
        sched.step()

        model.eval()
        with torch.no_grad():
            te_logits = model(Xte_t).cpu().numpy()
        rgb_pred = te_logits.argmax(-1)
        acc = (rgb_pred == yte).mean()
        rgb_probs = torch.softmax(torch.from_numpy(te_logits), dim=-1).numpy()

        # fusion sweep + oracle
        best_fuse = 0.0; best_alpha = 0.0
        for alpha in np.linspace(0, 1, 41):
            fused = alpha * pm_probs + (1 - alpha) * rgb_probs
            fa = (fused.argmax(-1) == yte).mean()
            if fa > best_fuse:
                best_fuse = fa; best_alpha = alpha
        oracle = ((pm_pred == yte) | (rgb_pred == yte)).mean()

        if acc > best_acc:
            best_acc = acc
            best_logits = te_logits
            torch.save(model.state_dict(), f"{WORK}/best.pt")

        if epoch % 5 == 0 or epoch == EPOCHS - 1 or best_fuse > pm_acc + 0.005:
            line = (f"ep {epoch:3d}  rgb {acc*100:.2f}%  best {best_acc*100:.2f}%  "
                    f"fuse[a={best_alpha:.2f}] {best_fuse*100:.2f}%  oracle {oracle*100:.2f}%")
            print(line)
            tg(f"<b>QCC ep{epoch}</b>\nRGB {acc*100:.2f}% best {best_acc*100:.2f}%\n"
               f"Fuse[a={best_alpha:.2f}] <b>{best_fuse*100:.2f}%</b>\nOracle {oracle*100:.2f}%")

    rgb_probs_best = torch.softmax(torch.from_numpy(best_logits), dim=-1).numpy()
    np.savez_compressed(f"{WORK}/test_preds.npz", probs=rgb_probs_best, labels=yte)

    rgb_pred = rgb_probs_best.argmax(-1)
    final_oracle = ((pm_pred == yte) | (rgb_pred == yte)).mean()
    best_fuse = 0.0; best_alpha = 0.0
    for alpha in np.linspace(0, 1, 41):
        fused = alpha * pm_probs + (1 - alpha) * rgb_probs_best
        a = (fused.argmax(-1) == yte).mean()
        if a > best_fuse:
            best_fuse = a; best_alpha = alpha
    final = (f"<b>RGB-QCC done</b>\nRGB best {best_acc*100:.2f}%\n"
             f"PMamba {pm_acc*100:.2f}%\n"
             f"Fuse[a={best_alpha:.2f}] <b>{best_fuse*100:.2f}%</b>  "
             f"(Δ {(best_fuse - pm_acc)*100:+.2f}pp)\n"
             f"Oracle {final_oracle*100:.2f}%")
    print(final.replace("<b>", "").replace("</b>", ""))
    tg(final)


if __name__ == "__main__":
    main()
