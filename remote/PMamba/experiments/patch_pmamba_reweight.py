"""Add PMambaRigidityReweight (Motion subclass with v9c-style CE reweighting).

Batch-normalized softplus weighting from rigidity clip-std, same formula as
depth_branch.model.DepthCNNLSTM. Exposes self.latest_sample_weights picked up
by main.py's already-patched weighted-CE loop.

Also defines a companion loader if NvidiaRigidityLoader isn't already present
(it is).
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class PMambaRigidityReweight" in src:
    print("PMambaRigidityReweight already present")
else:
    snippet = '''

class PMambaRigidityReweight(Motion):
    """PMamba Motion with v9c-clean per-clip CE reweighting.

    Accepts forward input as either a tensor (no reweighting) or a
    (pts, rigidity_tensor) tuple. Rigidity shape (B, T, K) or (B, T, P).
    Sample weight:
        clip_std_i = std(rigidity_mean_per_frame_i across frames)
        z_i = (clip_std_i - mean(clip_std)) / (std(clip_std) + eps)
        w_i = clip(softplus(beta * z_i), max=2) ; normalized to mean 1
    Stored on self.latest_sample_weights; main.py's weighted-CE path uses it.
    """

    def __init__(self, *args, clip_reweight_beta=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_reweight_beta = clip_reweight_beta
        self.latest_sample_weights = None

    def forward(self, inputs):
        rig = None
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            pts, rig = inputs
        else:
            pts = inputs
        if self.training and self.clip_reweight_beta != 0 and rig is not None:
            r = rig.float()
            if r.dim() == 3:
                per_frame = r.mean(dim=-1) if r.shape[-1] != 6 else r[..., 0]
            else:
                per_frame = r
            clip_std = per_frame.std(dim=-1)
            mu = clip_std.mean()
            sd = clip_std.std() + 1e-6
            z = (clip_std - mu) / sd
            w = torch.nn.functional.softplus(self.clip_reweight_beta * z)
            w = torch.clamp(w, max=2.0)
            w = w * (w.numel() / (w.sum() + 1e-8))
            self.latest_sample_weights = w
        else:
            self.latest_sample_weights = None
        return super().forward(pts)
'''
    src = src.rstrip() + snippet + "\n"
    MOTION.write_text(src, encoding="utf-8")
    print("appended PMambaRigidityReweight to models/motion.py")
