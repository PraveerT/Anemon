"""CN-XXL student distilled from the 91.08 quat-head teacher, plus a reflection-
EQUIVARIANCE term (NOT invariance): f(mirror x) = P f(x), where P is the empirical
mirror-twin class permutation. Handedness is a feature, so the student must map a
mirrored gesture to its TWIN class, not agree with itself. The equivariance weight
lambda decays to 0 (constant lambda collapses it -4pp; it helps only as an early
nudge). KD soft-labels are the stable regularizer that lifts the fragile ~89 floor.

loss (added via main.py's aux hook) = kd_weight * KD(T) + lambda(epoch) * equiv.
Inference = a plain CN-XXL quat-head forward (== baseline), so eval is unchanged.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead
from models.motion_cleanest_quat import _inertia_quat

_TEACHER_MARGS = dict(num_classes=25, pts_size=172, topk=8, knn=[32, 24, 48, 24],
                      multi_scale_num_scales=5, lxl_hidden_dim=256, lxl_mlp_dim=512,
                      lxl_num_layers=4, lxl_dropout=0.3, lxl_bidirectional=True,
                      lxl_residual_scale=0.7)


class MotionCleanestLinXLKD(MotionCleanestLinXLQuatHead):
    def __init__(self, *args,
                 teacher_ckpt='/notebooks/Manta/experiments/work_dir/pgcnet_quat_head/best_model.pt',
                 P_path='mirror_P.npy', kd_temp=4.0, kd_weight=1.0,
                 eq_lambda0=0.5, eq_decay_end=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_temp = float(kd_temp)
        self.kd_weight = float(kd_weight)
        self.eq_lambda0 = float(eq_lambda0)
        self.eq_decay_end = int(eq_decay_end)
        self.cur_epoch = 0
        self.aux_loss = None
        # frozen teacher, hidden from nn.Module registration (list) so it is not
        # in parameters()/state_dict()/.train(); lazily moved to device in forward
        t = MotionCleanestLinXLQuatHead(**_TEACHER_MARGS)
        ck = torch.load(teacher_ckpt, map_location='cpu')
        sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
        t.load_state_dict(sd, strict=False)
        t.eval(); t.pts_size = 172
        for p in t.parameters():
            p.requires_grad_(False)
        self._teacher = [t]
        # P matrix: Pmat[P[i], i] = 1  -> target = softdist @ Pmat.T
        P = np.load(P_path)
        Pmat = torch.zeros(self.num_classes, self.num_classes)
        for i, j in enumerate(P):
            Pmat[int(j), i] = 1.0
        self.register_buffer('Pmat', Pmat)
        # twin classes = where P[i] != i; equivariance is applied ONLY here
        # (non-twin classes are masked out, never forced toward invariance)
        twin = torch.tensor([1.0 if int(P[i]) != i else 0.0 for i in range(self.num_classes)])
        self.register_buffer('twin_mask', twin)

    def _logits_from_coords(self, coords):          # replicate quat-head head
        B, _, T, _ = coords.shape
        quat = _inertia_quat(coords[:, :3]).reshape(B, T * 4)
        if self.quat_zero_input:
            quat = torch.zeros_like(quat)
        aux_logits = self.quat_head(quat)
        fea3 = self._encode_sampled_points(coords)
        g = self.global_bn(self.pool5(self.stage5(fea3))).flatten(1)
        main_logits = self.classify_features(g)
        if self.quat_no_aux:
            return main_logits
        return main_logits + self.quat_head_scale * aux_logits

    @staticmethod
    def _mirror_coords(coords):                     # flip u (ch0) about per-frame centroid
        c = coords.clone()
        u = c[:, 0]                                  # (B,T,N)
        um = u.mean(dim=-1, keepdim=True)
        c[:, 0] = 2 * um - u
        return c

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        coords = self._sample_points(inputs)         # random (train) / linspace (eval)
        s = self._logits_from_coords(coords)
        if not self.training:
            self.aux_loss = None
            return s
        # KD soft labels from teacher (skip entirely if kd_weight==0)
        if self.kd_weight > 0.0:
            teacher = self._teacher[0]
            if next(teacher.parameters()).device != inputs.device:
                teacher.to(inputs.device)
            teacher.eval()
            with torch.no_grad():
                t_logits = teacher(inputs)
            T = self.kd_temp
            kd = F.kl_div(F.log_softmax(s / T, dim=1),
                          F.softmax(t_logits / T, dim=1),
                          reduction='batchmean') * (T * T)
        else:
            kd = s.new_zeros(())
        # reflection EQUIVARIANCE on the SAME sampled points: f(mirror) = P f
        lam = self.eq_lambda0 * max(0.0, 1.0 - self.cur_epoch / max(1, self.eq_decay_end))
        sel = self.twin_mask[s.argmax(dim=1)] > 0 if lam > 0.0 else None
        if lam > 0.0 and bool(sel.any()):
            s_m = self._logits_from_coords(self._mirror_coords(coords))
            tgt = F.softmax(s.detach(), dim=1) @ self.Pmat.t()  # permute by twin map
            eq = F.kl_div(F.log_softmax(s_m[sel], dim=1), tgt[sel], reduction='batchmean')
            self.aux_loss = self.kd_weight * kd + lam * eq
        else:
            self.aux_loss = self.kd_weight * kd
        return s
