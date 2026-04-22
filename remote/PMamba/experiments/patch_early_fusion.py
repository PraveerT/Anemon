"""Add NvidiaMultiLoader (pts + depth+tops + rigidity stats) and
PMambaDepthEarlyFusion (feature-level fusion of Motion and DepthCNNLSTM)."""
from pathlib import Path

# 1) multimodal loader
Path("nvidia_multi_loader.py").write_text('''"""Yield pmamba pts, depth+tops tensor, and rigidity stats from one sample."""
import torch
from nvidia_dataloader import NvidiaLoader
from depth_branch.dataloader import DepthVideoLoader


class NvidiaMultiLoader(NvidiaLoader):
    """Composes NvidiaLoader (pts) with DepthVideoLoader (depth+tops+rigidity).

    Returns: ((pts, depth_tensor, rigidity_tensor), label, line)
    """

    def __init__(self, *args, img_size=112, use_tops=True, use_rigidity=True,
                 rigidity_per_point=False, rigidity_norm_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Share phase / valid_subject / framerate with depth side.
        self._depth_loader = DepthVideoLoader(
            framerate=kwargs.get("framerate", args[0] if args else 32),
            valid_subject=kwargs.get("valid_subject"),
            phase=kwargs.get("phase", "train"),
            img_size=img_size,
            use_tops=use_tops,
            use_rigidity=use_rigidity,
            rigidity_per_point=rigidity_per_point,
            rigidity_norm_scale=rigidity_norm_scale,
            # no train augments here — augmenting pm vs depth independently breaks pairing
            hflip_prob=0.0,
            time_cutout_prob=0.0,
        )

    def __getitem__(self, index):
        pts, label, line = super().__getitem__(index)
        out_d, _, _ = self._depth_loader[index]
        depth_tensor, rigidity_tensor = out_d
        return (pts, depth_tensor, rigidity_tensor), label, line
''', encoding="utf-8")

# 2) early-fusion model appended to models/motion.py
MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class PMambaDepthEarlyFusion" in src:
    print("PMambaDepthEarlyFusion already present — skipping")
else:
    snippet = '''

class PMambaDepthEarlyFusion(nn.Module):
    """Feature-level early-fusion of Motion (PMamba) + DepthCNNLSTM (v9c-style).

    pm_feat:  (B, 1024)  = Motion.extract_features
    dpt_feat: (B, 1024)  = DepthCNNLSTM.extract_features  (lstm_hidden * 4)
    fused = concat -> MLP -> num_classes.
    """

    def __init__(self, num_classes=25, pts_size=256,
                 knn=(32, 24, 48, 24), topk=8,
                 depth_in_channels=4, depth_feat_dim=256, depth_lstm_hidden=256,
                 depth_lstm_layers=2, depth_bidir=True, depth_dropout=0.3,
                 clip_reweight_beta=1.5,
                 pmamba_weights=None, depth_weights=None,
                 freeze_pmamba=False, freeze_depth=False,
                 pmamba_feat_dim=1024, fusion_hidden=512, fusion_dropout=0.3,
                 **kwargs):
        super().__init__()
        from depth_branch.model import DepthCNNLSTM
        self.pmamba = Motion(num_classes=num_classes, pts_size=pts_size,
                             knn=list(knn), topk=topk)
        self.depth = DepthCNNLSTM(
            num_classes=num_classes, in_channels=depth_in_channels,
            feat_dim=depth_feat_dim, lstm_hidden=depth_lstm_hidden,
            lstm_layers=depth_lstm_layers, bidirectional=depth_bidir,
            dropout=depth_dropout,
            rigidity_dim=0, rigidity_aux_dim=0, clip_reweight_beta=clip_reweight_beta,
        )

        if pmamba_weights:
            sd = torch.load(pmamba_weights, map_location='cpu')
            sd = sd.get('model_state_dict', sd)
            m, u = self.pmamba.load_state_dict(sd, strict=False)
            print(f"PMamba weights: missing={len(m)} unexpected={len(u)}")
        if depth_weights:
            sd = torch.load(depth_weights, map_location='cpu')
            sd = sd.get('model_state_dict', sd)
            m, u = self.depth.load_state_dict(sd, strict=False)
            print(f"Depth weights: missing={len(m)} unexpected={len(u)}")

        if freeze_pmamba:
            for p in self.pmamba.parameters(): p.requires_grad_(False)
        if freeze_depth:
            for p in self.depth.parameters(): p.requires_grad_(False)

        mult = 2 if depth_bidir else 1
        depth_feat_out = depth_lstm_hidden * mult * 2
        total = pmamba_feat_dim + depth_feat_out
        self.fusion_head = nn.Sequential(
            nn.Linear(total, fusion_hidden),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, inputs):
        # inputs: (pts, depth_tensor, rigidity_tensor) tuple
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            pts, depth, rig = inputs
        else:
            raise ValueError("PMambaDepthEarlyFusion expects (pts, depth, rigidity) tuple")
        pm_feat = self.pmamba.extract_features(pts)
        dp_feat = self.depth.extract_features((depth, rig))
        return self.fusion_head(torch.cat([pm_feat, dp_feat], dim=1))
'''
    src = src.rstrip() + snippet + "\n"
    MOTION.write_text(src, encoding="utf-8")
    print("appended PMambaDepthEarlyFusion to models/motion.py")
