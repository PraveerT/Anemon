"""Extract per-frame mid-stage features from PMamba ep110 checkpoint.

For each clip, runs the encoder up to stage5 output (B, 1024, T, P_s3),
mean-pools over P -> (B, T, 1024). Saves to .npz.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, "/notebooks/PMamba/experiments")
import nvidia_dataloader as nd
import models.motion as MM

CKPT = "/notebooks/PMamba/experiments/work_dir/pmamba_branch/epoch110_model.pt"
OUT = "/notebooks/PMamba/dataset/Nvidia/Processed"


def extract_for_split(model, ds, split):
    DEV = torch.device("cuda")
    feats_all = []
    labels_all = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, _ = ds[i]                                              # (T, N, 8) numpy
            x = torch.from_numpy(x).float().unsqueeze(0).to(DEV)         # (1, T, N, 8)
            coords = model._sample_points(x)                              # (1, 4, T, P)
            fea3 = model._encode_sampled_points(coords)                  # (1, 260, T, P_s3)
            output = model.stage5(fea3)                                   # (1, 1024, T, P_s3)
            # mean-pool over P only
            per_frame = output.mean(dim=-1).squeeze(0)                    # (1024, T)
            per_frame = per_frame.transpose(0, 1).cpu().numpy()           # (T, 1024)
            feats_all.append(per_frame)
            labels_all.append(int(y))
            if i % 100 == 0:
                print(f"  {split} {i}/{len(ds)}")
    feats = np.stack(feats_all, axis=0).astype(np.float32)               # (N, T, 1024)
    labels = np.array(labels_all, dtype=np.int64)
    np.savez_compressed(f"{OUT}/pmamba_feats_{split}.npz", feats=feats, labels=labels)
    print(f"{split} wrote: {feats.shape}")


def main():
    DEV = torch.device("cuda")
    model = MM.Motion(num_classes=25, pts_size=96, knn=[32, 24, 48, 24], topk=8).to(DEV)
    sd = torch.load(CKPT, map_location=DEV)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    print("loaded pmamba ep110")

    for split in ("train", "test"):
        ds = nd.NvidiaLoader(framerate=32, phase=split)
        # NvidiaLoader internally uses pts.npy; force pts_size to fixed value via model
        model.pts_size = 256
        extract_for_split(model, ds, split)


if __name__ == "__main__":
    main()
