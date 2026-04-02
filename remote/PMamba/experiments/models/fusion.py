import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.reqnn_motion import BearingQCCFeatureMotion


class MotionDualBranchFusion(nn.Module):
    """Logit-gated fusion of frozen PMamba and quaternion branches.

    Branches are weight-frozen but kept in train mode for random point
    sampling. During eval, K internal TTA passes average probabilities
    (arithmetic mean) for robust predictions. During training, single
    pass provides data augmentation for the gate.
    """

    def __init__(
        self,
        num_classes,
        pts_size,
        temporal_model_args=None,
        spatial_model_args=None,
        aux_weight=0.0,
        tta_k=10,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pts_size = pts_size
        self.aux_weight = aux_weight
        self.tta_k = tta_k

        # Temporal branch (PMamba)
        self.temporal_branch = Motion(
            num_classes=num_classes,
            pts_size=pts_size,
            **(temporal_model_args or {}),
        )
        self.temporal_feat_dim = self.temporal_branch.feature_dim

        # Spatial branch (Quaternion)
        self.spatial_branch = BearingQCCFeatureMotion(
            num_classes=num_classes,
            pts_size=pts_size,
            **(spatial_model_args or {}),
        )
        self.spatial_feat_dim = self.spatial_branch.feature_dim

        # Freeze branch weights
        for param in self.temporal_branch.parameters():
            param.requires_grad = False
        for param in self.spatial_branch.parameters():
            param.requires_grad = False

        # Logit-based gate
        self.gate = nn.Linear(num_classes * 2, num_classes)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 1.4)

        self.temporal_logits = None
        self.spatial_logits = None

    def _stabilize_branches(self):
        for m in list(self.temporal_branch.modules()) + list(self.spatial_branch.modules()):
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
                m.eval()

    def train(self, mode=True):
        super().train(mode)
        # Branches always in train mode for random point sampling
        self.temporal_branch.train()
        self.spatial_branch.train()
        self._stabilize_branches()
        return self

    def _single_pass(self, points, aux_input):
        """One forward pass with current random point sampling."""
        with torch.no_grad():
            t_feat = self.temporal_branch.extract_features(points)
            s_feat = self.spatial_branch.extract_features(points, aux_input=aux_input)
            t_logits = self.temporal_branch.classify_features(t_feat)
            s_logits = self.spatial_branch.classifier(s_feat)
        return t_logits, s_logits

    def forward(self, inputs):
        if isinstance(inputs, dict):
            points = inputs['points']
            aux_input = inputs
        else:
            points = inputs
            aux_input = None

        K = self.tta_k if not self.training else 1

        if K == 1:
            t_logits, s_logits = self._single_pass(points, aux_input)
            self.temporal_logits = t_logits
            self.spatial_logits = s_logits
        else:
            # Average logits and probabilities over K random point samples
            all_t = []
            all_s = []
            for _ in range(K):
                t_logits, s_logits = self._single_pass(points, aux_input)
                all_t.append(t_logits)
                all_s.append(s_logits)
            # Average logits for gate input (smoother signal)
            t_logits = torch.stack(all_t).mean(0)
            s_logits = torch.stack(all_s).mean(0)
            self.temporal_logits = t_logits
            self.spatial_logits = s_logits

        # Gate from (averaged) logits
        gate_input = torch.cat([t_logits, s_logits], dim=1)
        gate_weights = torch.sigmoid(self.gate(gate_input))

        t_probs = F.softmax(t_logits, dim=1)
        s_probs = F.softmax(s_logits, dim=1)

        fused = gate_weights * t_probs + (1 - gate_weights) * s_probs
        return torch.log(fused + 1e-8)
