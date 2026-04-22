"""Add PMambaFlowAux: Motion + auxiliary per-frame flow-stats prediction head.

Paper-inspired (Mittal 2020 scene flow self-supervision) but adapted to
supervised gesture classification:
- Tap PMamba's per-point features (fea3 = 260 ch) right before stage5's pool.
- Pool over points -> per-frame 260-dim vector.
- Tiny MLP predicts per-frame K=6 rigidity summary (Kabsch-residual stats
  from the precomputed {stem}_rigidity.npy files).
- Loss: CE + lambda * MSE(pred, target). Target is observable; no collapse.

Hypothesis: learning per-frame flow/non-rigidity stats should shape features
to encode articulation, helping solo classification.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class PMambaFlowAux" in src:
    print("PMambaFlowAux already present")
else:
    snippet = '''

class PMambaFlowAux(Motion):
    """Motion with auxiliary per-frame rigidity-summary prediction head.

    Input accepted as either pts tensor or (pts, rigidity_tensor) tuple.
    rigidity_tensor shape (B, T, K=6). During training we minimise
    CE + flow_aux_weight * MSE(head(per_frame_feat), rigidity_tensor).
    """

    def __init__(self, *args, flow_aux_weight=0.1, flow_aux_dim=6,
                 flow_aux_hidden=64, flow_feat_channels=260, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_aux_weight = flow_aux_weight
        self.flow_aux_dim = flow_aux_dim
        self.flow_head = nn.Sequential(
            nn.Linear(flow_feat_channels, flow_aux_hidden),
            nn.GELU(),
            nn.Linear(flow_aux_hidden, flow_aux_dim),
        )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        self._aux_target = None

    def extract_features(self, inputs):
        # Handle (pts, rig_target) tuple; stash target for aux loss.
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            pts, self._aux_target = inputs
        else:
            pts = inputs
            self._aux_target = None

        coords = self._sample_points(pts)
        fea3 = self._encode_sampled_points(coords)        # (B, 260, T, P)

        # Auxiliary head on per-frame pooled features.
        if self.training and self.flow_aux_weight > 0 and self._aux_target is not None:
            per_frame = fea3.mean(dim=-1)                 # (B, 260, T)
            pred = self.flow_head(per_frame.transpose(1, 2))  # (B, T, K)
            target = self._aux_target.float()
            # Adjust for T mismatch (shouldn't happen with framerate=32 everywhere).
            if target.shape[1] != pred.shape[1]:
                # crop or interp
                target = target[:, :pred.shape[1]]
            aux = torch.nn.functional.mse_loss(pred, target)
            self.latest_aux_loss = self.flow_aux_weight * aux
            self.latest_aux_metrics = {
                'qcc_raw': aux.detach(),
                'qcc_forward': aux.detach(),
                'qcc_backward': aux.detach(),
                'qcc_valid_ratio': torch.ones(1, device=aux.device),
            }
        else:
            self.latest_aux_loss = None
            self.latest_aux_metrics = {}

        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics
'''
    src = src.rstrip() + snippet + "\n"
    MOTION.write_text(src, encoding="utf-8")
    print("appended PMambaFlowAux to models/motion.py")
