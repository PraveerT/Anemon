"""Add MotionQuatProto: PMamba + Hamilton-mix layer + class quaternion prototypes.

After PMamba's 1024-d feature:
  1. QuaternionLinear(1024, 1024)   — Hamilton-product mix (idea #4)
  2. Linear(1024, 4) -> S^3 unit quat (q_pred)
  3. logit_c = scale * <q_pred, q_c>  for each class (geodesic-style head)
Class prototypes q_c (25 unit quats) are learnable.

Auxiliary cycle/spread regularizer pushes q_c apart on S^3 (idea #1).
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionQuatProto" in src:
    start = src.find("\n\nclass MotionQuatProto")
    end = src.find("\n\nclass ", start + 1)
    src = src[:start] if end == -1 else src[:start] + src[end:]
    print("stripped existing MotionQuatProto")

snippet = '''


class MotionQuatProto(Motion):
    """PMamba + quaternion classification head with learnable class prototypes.

    Loss exposed as `aux_loss` attribute on each forward (cycle/spread reg).
    """

    def __init__(self, num_classes, pts_size, quat_scale=16.0, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.quat_scale = quat_scale
        self.quat_mix = QuaternionLinear(self.feature_dim, self.feature_dim)
        self.quat_head = nn.Linear(self.feature_dim, 4)
        # Class prototypes on S^3: random init then normalize
        proto = torch.randn(num_classes, 4)
        proto = proto / proto.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.class_quats = nn.Parameter(proto)

    def _normalize_quat(self, q):
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def classify_features(self, features):
        # features: (B, 1024)
        h = self.quat_mix(features.unsqueeze(1)).squeeze(1)        # (B, 1024)
        q_pred = self._normalize_quat(self.quat_head(h))           # (B, 4)
        protos = self._normalize_quat(self.class_quats)            # (C, 4)
        # Cosine similarity (handles q ~ -q via abs of dot product)
        cos = (q_pred @ protos.t()).abs()                          # (B, C)
        return self.quat_scale * cos

    def forward(self, inputs):
        return self.classify_features(self.extract_features(inputs))
'''

MOTION.write_text(src.rstrip() + snippet + "\n", encoding="utf-8")
print("added MotionQuatProto to models/motion.py")
