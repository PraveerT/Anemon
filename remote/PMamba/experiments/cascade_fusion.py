"""Cascade fusion analysis.

For each test sample:
  - Take PMamba ep110's prediction
  - If PMamba's top-2 predictions form a known confusion pair, defer to that
    pair's specialist (Cfbq fine-tuned on the 2 classes)
  - Otherwise keep PMamba's prediction
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/notebooks/PMamba/experiments")
import nvidia_dataloader as nd
import models.motion as MM


PAIRS = [(3, 16), (18, 9), (5, 4), (1, 0)]
WORK = "/notebooks/PMamba/experiments/work_dir"
PMAMBA_NPZ = f"{WORK}/pmamba_branch/pmamba_test_preds.npz"
DEV = torch.device("cuda")


def specialist_workdir(a, b):
    return f"{WORK}/pmamba_cfbq_pair{a}_{b}"


def load_specialist(ckpt_path):
    model = MM.MotionCfbq(num_classes=25, pts_size=256, knn=[32, 24, 48, 24], topk=8).to(DEV).eval()
    sd = torch.load(ckpt_path, map_location=DEV)
    if isinstance(sd, dict):
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model


@torch.no_grad()
def specialist_logits_on_full_test(model):
    # Need to use full test list (not the filtered one currently in train_depth_list.txt)
    # Temporarily restore test_depth_list.txt to full
    import shutil
    proc = "/notebooks/PMamba/dataset/Nvidia/Processed"
    shutil.copy(f"{proc}/test_depth_list_full.txt", f"{proc}/test_depth_list.txt")
    ds = nd.NvidiaLoader(framerate=32, phase="test")
    logits_all = np.zeros((len(ds), 25), dtype=np.float32)
    for i in range(len(ds)):
        x, _, _ = ds[i]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEV)
        logits = model(x).squeeze(0).cpu().numpy()
        logits_all[i] = logits
        if i % 100 == 0:
            print(f"  {i}/{len(ds)}")
    return logits_all


def main():
    pm = np.load(PMAMBA_NPZ)
    pm_probs, labels = pm["probs"], pm["labels"]
    pm_pred = pm_probs.argmax(-1)
    pm_acc = (pm_pred == labels).mean()
    print(f"PMamba ep110 test acc: {pm_acc*100:.2f}%")

    # Get top-2 predictions per sample
    top2 = np.argsort(-pm_probs, axis=-1)[:, :2]                # (482, 2)

    # Run each specialist on full test set
    specialist_probs = {}
    for a, b in PAIRS:
        wdir = specialist_workdir(a, b)
        ckpt = f"{wdir}/best_model.pt"
        if not os.path.exists(ckpt):
            print(f"[skip] {wdir} (no best_model.pt)")
            continue
        print(f"\nLoading specialist for pair ({a}, {b})...")
        m = load_specialist(ckpt)
        print(f"Inferring on full test ({a},{b})...")
        logits = specialist_logits_on_full_test(m)
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        specialist_probs[(a, b)] = probs

    # Cascade routing
    cascade_pred = pm_pred.copy()
    routed = 0
    for i in range(len(labels)):
        t = set(top2[i].tolist())
        for (a, b), spec_probs in specialist_probs.items():
            if t == {a, b}:
                # restrict specialist's prediction to the pair classes
                pair_logits = np.array([spec_probs[i, a], spec_probs[i, b]])
                cascade_pred[i] = a if pair_logits[0] > pair_logits[1] else b
                routed += 1
                break

    cascade_acc = (cascade_pred == labels).mean()
    print(f"\n=== CASCADE RESULT ===")
    print(f"PMamba alone:  {pm_acc*100:.2f}%")
    print(f"Cascade:       {cascade_acc*100:.2f}%   (delta {(cascade_acc-pm_acc)*100:+.2f}pp)")
    print(f"Routed: {routed}/{len(labels)}")

    # Per-pair: how many of the routed samples flipped? Also check what fraction got correct.
    print(f"\nPer-pair routing analysis:")
    for (a, b), spec_probs in specialist_probs.items():
        in_pair_mask = ((labels == a) | (labels == b)) & (
            np.isin(top2.flatten(), [a, b]).reshape(top2.shape).all(axis=-1)
        )
        # Actually compute: of test samples where pmamba's top-2 = {a,b}, how many true label match each side
        # For each such sample, did specialist get it right?
        n_routed_pair = 0
        n_pmamba_right = 0
        n_specialist_right = 0
        for i in range(len(labels)):
            if set(top2[i].tolist()) == {a, b}:
                n_routed_pair += 1
                if pm_pred[i] == labels[i]:
                    n_pmamba_right += 1
                if cascade_pred[i] == labels[i]:
                    n_specialist_right += 1
        delta = n_specialist_right - n_pmamba_right
        print(f"  ({a:2d},{b:2d}): {n_routed_pair} routed | pmamba {n_pmamba_right}/{n_routed_pair} | specialist {n_specialist_right}/{n_routed_pair} | delta {delta:+d}")


if __name__ == "__main__":
    main()
