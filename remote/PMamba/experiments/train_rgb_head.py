"""Train tiny MLP head on frozen ResNet18 RGB features.

Reads:
  /notebooks/PMamba/dataset/Nvidia/Processed/rgb_resnet18_{train,test}.npz
  (each: feats (N, 16, 512), labels (N,))

Late-fuses with PMamba(depth) baseline:
  /notebooks/PMamba/experiments/work_dir/pmamba_branch/pmamba_test_preds.npz
  (probs (482, 25), labels (482,))

Reports: RGB solo acc, PMamba solo acc, fusion (alpha sweep) best alpha + acc.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda")
PROC = "/notebooks/PMamba/dataset/Nvidia/Processed"
PMAMBA = "/notebooks/PMamba/experiments/work_dir/pmamba_branch/pmamba_test_preds.npz"

tr = np.load(f"{PROC}/rgb_resnet18_train.npz")
te = np.load(f"{PROC}/rgb_resnet18_test.npz")
Xtr, ytr = tr["feats"], tr["labels"]
Xte, yte = te["feats"], te["labels"]
print("train", Xtr.shape, "test", Xte.shape)

# mean-pool over time -> (N, 512)
Xtr_p = torch.from_numpy(Xtr.mean(axis=1)).float().to(DEV)
Xte_p = torch.from_numpy(Xte.mean(axis=1)).float().to(DEV)
ytr_t = torch.from_numpy(ytr).long().to(DEV)
yte_t = torch.from_numpy(yte).long().to(DEV)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 25),
        )
    def forward(self, x):
        return self.net(x)


torch.manual_seed(0)
np.random.seed(0)

m = Head().to(DEV)
opt = torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-4)

best_acc = 0.0
best_logits = None
N = Xtr_p.shape[0]
B = 32
for epoch in range(60):
    m.train()
    perm = torch.randperm(N, device=DEV)
    for i in range(0, N, B):
        idx = perm[i:i+B]
        logits = m(Xtr_p[idx])
        loss = F.cross_entropy(logits, ytr_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    m.eval()
    with torch.no_grad():
        logits = m(Xte_p)
        acc = (logits.argmax(-1) == yte_t).float().mean().item()
    if acc > best_acc:
        best_acc = acc
        best_logits = logits.detach().cpu().numpy()
    if epoch % 5 == 0 or epoch == 59:
        print(f"ep {epoch:2d}  test_acc {acc*100:.2f}%  best {best_acc*100:.2f}%")

print(f"\nRGB solo best acc: {best_acc*100:.2f}%")
rgb_probs = torch.softmax(torch.from_numpy(best_logits), dim=-1).numpy()
np.savez_compressed(f"{PROC}/rgb_resnet18_test_preds.npz",
                    probs=rgb_probs, labels=yte)

# late-fuse with PMamba
pm = np.load(PMAMBA)
pm_probs, pm_labels = pm["probs"], pm["labels"]
assert (pm_labels == yte).all(), "label order mismatch"
pm_acc = (pm_probs.argmax(-1) == pm_labels).mean()
print(f"PMamba(depth) solo: {pm_acc*100:.2f}%")

print("alpha sweep — alpha = weight on PMamba (1-alpha on RGB):")
best_fuse = 0.0
best_alpha = 0.0
for alpha in np.linspace(0, 1, 21):
    fused = alpha * pm_probs + (1 - alpha) * rgb_probs
    acc = (fused.argmax(-1) == yte).mean()
    if acc > best_fuse:
        best_fuse = acc; best_alpha = alpha
    print(f"  alpha={alpha:.2f}  acc={acc*100:.2f}%")
print(f"\nBest fusion: alpha={best_alpha:.2f}, acc={best_fuse*100:.2f}%")
print(f"vs PMamba alone: {pm_acc*100:.2f}%  (delta {(best_fuse - pm_acc)*100:+.2f}pp)")
