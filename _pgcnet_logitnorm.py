"""PGCNetPrunedLogitNorm -- lean PGCNetPruned + LogitNorm training-time loss.

LogitNorm (Wei et al. 2022): the attractor collapse is driven by unbounded logit
magnitude -> overconfidence, so on test the ambiguous victim-class samples get
swallowed by high-norm attractor classes. Fix: during TRAINING, feed the CE loss
the *direction-only* logits z/(tau*||z||), decoupling magnitude from the decision.
At INFERENCE we return raw logits (argmax is scale-invariant), so the network is
byte-identical at test -- this is a pure training-time regularizer.

tau is the temperature: smaller -> sharper. Paper default ~0.04.
"""
from models.pgcnet_pruned import PGCNetPruned


class PGCNetPrunedLogitNorm(PGCNetPruned):
    def __init__(self, *args, logitnorm_tau=0.04, **kwargs):
        super().__init__(*args, **kwargs)
        self.logitnorm_tau = float(logitnorm_tau)

    def forward(self, inputs):
        logits = super().forward(inputs)
        if self.training:
            norm = logits.norm(dim=1, keepdim=True) + 1e-7
            return logits / (norm * self.logitnorm_tau)
        return logits
