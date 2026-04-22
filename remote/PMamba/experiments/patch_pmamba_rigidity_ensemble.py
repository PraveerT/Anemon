"""Add NvidiaRigidityLoader (wraps NvidiaLoader + rigidity .npy) and
PMambaRigidityEnsemble (PMamba + RigidityOnlyClassifier, fused logits).

Both live in new file modules so the existing NvidiaLoader / Motion stay
untouched.
"""
from pathlib import Path

# 1) loader
LOADER_PATH = Path("nvidia_rigidity_loader.py")
LOADER_PATH.write_text('''"""Pair NvidiaLoader point clouds with precomputed rigidity per-point stats.

Returns: ((pmamba_input, rigidity_tensor), label, line). pmamba_input has the
same shape/dtype as NvidiaLoader's output so PMamba accepts it unchanged.
"""
import numpy as np
import torch

from nvidia_dataloader import NvidiaLoader


class NvidiaRigidityLoader(NvidiaLoader):
    def __init__(self, *args, rigidity_per_point=True, rigidity_sort=True,
                 rigidity_norm_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rigidity_per_point = rigidity_per_point
        self.rigidity_sort = rigidity_sort
        self.rigidity_norm_scale = rigidity_norm_scale

    def __getitem__(self, index):
        pts, label, line = super().__getitem__(index)
        # Resolve rigidity path — pts.npy / depth.npy share the same stem.
        rel = self.r.split(line)[1][1:-4]                    # ./Nvidia/... stem
        suffix = "_rigidity_pp.npy" if self.rigidity_per_point else "_rigidity.npy"
        rig_path = f"../dataset/{rel}{suffix}"
        rig = np.load(rig_path).astype(np.float32)            # (T, P) or (T, K)
        if self.rigidity_per_point and self.rigidity_sort:
            rig = np.sort(rig, axis=-1)[:, ::-1].copy()
        rig = rig * self.rigidity_norm_scale
        return (pts, torch.from_numpy(rig).float()), label, line
''', encoding="utf-8")

# 2) ensemble model: append to models/motion.py
MOTION_PATH = Path("models/motion.py")
src = MOTION_PATH.read_text(encoding="utf-8")
if "class PMambaRigidityEnsemble" in src:
    print("PMambaRigidityEnsemble already present — skipping")
else:
    snippet = '''

class PMambaRigidityEnsemble(nn.Module):
    """Late-fusion ensemble: PMamba (point-cloud) + RigidityOnlyClassifier
    (per-frame sorted residuals). Logits fused via softmax(alpha) weighted mean
    of softmax probs. alpha starts biased toward PMamba (init logit 2.0
    -> softmax ~0.88 weight on PMamba).
    """

    def __init__(self, num_classes=25, pts_size=256,
                 knn=(32, 24, 48, 24), topk=8,
                 rigidity_dim=256, rigidity_hidden=128, rigidity_lstm_layers=2,
                 rigidity_dropout=0.3,
                 pmamba_weights=None, rigidity_weights=None,
                 freeze_pmamba=False, freeze_rigidity=False,
                 init_alpha_logit=2.0, **kwargs):
        super().__init__()
        self.pmamba = Motion(num_classes=num_classes, pts_size=pts_size,
                             knn=list(knn), topk=topk)
        from depth_branch.model import RigidityOnlyClassifier
        self.rigidity = RigidityOnlyClassifier(
            num_classes=num_classes, rigidity_dim=rigidity_dim,
            hidden=rigidity_hidden, lstm_layers=rigidity_lstm_layers,
            dropout=rigidity_dropout,
        )
        # Learnable fusion scalar (softmax of two logits).
        self.fusion_logits = nn.Parameter(torch.tensor([init_alpha_logit, 0.0]))

        if pmamba_weights:
            sd = torch.load(pmamba_weights, map_location='cpu')
            sd = sd.get('model_state_dict', sd)
            missing, unexpected = self.pmamba.load_state_dict(sd, strict=False)
            print(f"PMamba weights: missing={len(missing)} unexpected={len(unexpected)}")
        if rigidity_weights:
            sd = torch.load(rigidity_weights, map_location='cpu')
            sd = sd.get('model_state_dict', sd)
            missing, unexpected = self.rigidity.load_state_dict(sd, strict=False)
            print(f"Rigidity weights: missing={len(missing)} unexpected={len(unexpected)}")

        if freeze_pmamba:
            for p in self.pmamba.parameters(): p.requires_grad_(False)
        if freeze_rigidity:
            for p in self.rigidity.parameters(): p.requires_grad_(False)

    def forward(self, inputs):
        # Expect a tuple (pmamba_input, rigidity_tensor).
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            pm_in, rig = inputs
        else:
            raise ValueError("PMambaRigidityEnsemble expects (pts, rigidity) tuple")
        pm_logits = self.pmamba(pm_in)
        rig_logits = self.rigidity(rig)
        weights = torch.softmax(self.fusion_logits, dim=0)              # (2,)
        pm_p = torch.softmax(pm_logits, dim=-1)
        rg_p = torch.softmax(rig_logits, dim=-1)
        fused = weights[0] * pm_p + weights[1] * rg_p
        # Return log of fused probs so cross-entropy -> NLL works as usual.
        return torch.log(fused.clamp(min=1e-9))
'''
    src = src.rstrip() + snippet + "\n"
    MOTION_PATH.write_text(src, encoding="utf-8")
    print("appended PMambaRigidityEnsemble to models/motion.py")

print(f"loader -> {LOADER_PATH}")
