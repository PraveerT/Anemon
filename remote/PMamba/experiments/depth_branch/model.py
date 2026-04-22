"""Depth CNN-LSTM models.

DepthCNNLSTM:     per-frame 2D CNN encoder -> BiLSTM over time -> mean+max pool -> classifier
DepthCNNLSTMQCC:  same backbone + quaternion-cycle-consistency aux loss on per-frame CNN feats

Input:  (B, T, C, H, W) float32 in [0, 1], C = 1 or 4
Output: (B, num_classes) logits  (aux loss fetched via get_auxiliary_loss())
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DepthCNN(nn.Module):
    """Small per-frame CNN. For 112x112 input -> 7x7 feature -> global pool."""

    def __init__(self, in_channels=1, feat_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block1 = ConvBlock(32, 64, stride=2)
        self.block2 = ConvBlock(64, 128, stride=2)
        self.block3 = ConvBlock(128, feat_dim, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x                                         # (B, D, 7, 7)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x)
        return x.flatten(1)


class DepthCNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes=25,
        in_channels=1,
        feat_dim=256,
        lstm_hidden=256,
        lstm_layers=2,
        bidirectional=True,
        dropout=0.3,
        rigidity_dim=0,                     # K; 0 disables concat-to-LSTM path (v6)
        rigidity_aux_dim=0,                 # K; 0 disables aux-predict head (v7)
        rigidity_aux_hidden=64,
        rigidity_aux_weight=0.1,
        rigidity_aux_loss='mse',           # 'mse' (v7) or 'bce_median' (v8)
        clip_reweight_beta=0.0,            # v9: CE weighting by clip rigidity std (0 disables)
        **kwargs,
    ):
        super().__init__()
        self.cnn = DepthCNN(in_channels=in_channels, feat_dim=feat_dim)
        self.rigidity_dim = rigidity_dim
        lstm_input = feat_dim + rigidity_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        mult = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * mult * 2, num_classes)

        # Auxiliary rigidity-prediction head (v7). Predicts per-frame rigidity
        # stats from per-frame CNN features. MSE against observed stats.
        self.rigidity_aux_dim = rigidity_aux_dim
        self.rigidity_aux_weight = rigidity_aux_weight
        self.rigidity_aux_loss = rigidity_aux_loss
        if rigidity_aux_dim > 0:
            out_dim = 1 if rigidity_aux_loss == 'bce_median' else rigidity_aux_dim
            self.rigidity_aux_head = nn.Sequential(
                nn.Linear(feat_dim, rigidity_aux_hidden),
                nn.GELU(),
                nn.Linear(rigidity_aux_hidden, out_dim),
            )
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        self.clip_reweight_beta = clip_reweight_beta
        self.latest_sample_weights = None

    def forward(self, inputs):
        # Accept: tensor (legacy), or (tensor, rigidity_tensor) tuple when
        # rigidity_dim > 0 or rigidity_aux_dim > 0.
        rigidity = None
        if isinstance(inputs, (tuple, list)):
            if len(inputs) == 2:
                inputs, rigidity = inputs
            else:
                inputs = inputs[0]
        if isinstance(inputs, dict):
            inputs = inputs["depth"]
        x = inputs.float()
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)
        feat = feat.view(B, T, -1)                       # (B, T, feat_dim)

        # v9: per-clip sample weights for CE reweighting. Uses rigidity tensor's
        # mean channel (channel 0) across frames; higher within-clip std ->
        # more "articulation dynamics" in the clip -> more CE weight.
        if self.clip_reweight_beta > 0 and rigidity is not None:
            mean_per_frame = rigidity[:, :, 0].float()                    # (B, T)
            clip_std = mean_per_frame.std(dim=1)                          # (B,)
            self.latest_sample_weights = 1.0 + self.clip_reweight_beta * clip_std
        else:
            self.latest_sample_weights = None

        # Auxiliary rigidity prediction (v7 mse / v8 bce_median) from per-frame feat.
        if self.training and self.rigidity_aux_dim > 0 and self.rigidity_aux_weight > 0:
            assert rigidity is not None, "rigidity_aux_dim>0 but no rigidity tensor supplied"
            rig_true = rigidity.float()                                          # (B, T, K)
            if self.rigidity_aux_loss == 'bce_median':
                # Target: is rigidity_mean_t above the per-clip median? (B, T) bool.
                mean_per_frame = rig_true[:, :, 0]                               # (B, T)
                median_per_clip = mean_per_frame.median(dim=1, keepdim=True).values
                target = (mean_per_frame > median_per_clip).float()              # (B, T)
                logits = self.rigidity_aux_head(feat).squeeze(-1)                # (B, T)
                aux = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            else:
                rig_pred = self.rigidity_aux_head(feat)                          # (B, T, K)
                aux = torch.nn.functional.mse_loss(rig_pred, rig_true)
            self.latest_aux_loss = self.rigidity_aux_weight * aux
            self.latest_aux_metrics = {
                'qcc_raw': aux.detach(),
                'qcc_forward': aux.detach(),
                'qcc_backward': aux.detach(),
                'qcc_valid_ratio': torch.ones(1, device=aux.device),
            }
        else:
            self.latest_aux_loss = None
            self.latest_aux_metrics = {}

        if self.rigidity_dim > 0:
            assert rigidity is not None, "rigidity_dim>0 but no rigidity tensor supplied"
            feat = torch.cat([feat, rigidity.float()], dim=-1)   # (B, T, feat_dim+K)

        seq, _ = self.lstm(feat)
        t_mean = seq.mean(dim=1)
        t_max = seq.max(dim=1).values
        pooled = torch.cat([t_mean, t_max], dim=-1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics


def quat_mul(q1, q2):
    """Hamilton product of two batched unit quaternions. Layout [w, x, y, z]."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


class QCCHead(nn.Module):
    """Predicts unit quaternion q_{t->t+1} from (feat_t, feat_{t+1}) pair."""

    def __init__(self, feat_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        # Bias near identity quaternion so early output starts near [1,0,0,0];
        # small random weights so gradient can flow from the first step.
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def forward(self, feats):
        # feats: (B, T, D). Build pairs (t, t+1 mod T) around cyclic sequence.
        nxt = torch.roll(feats, shifts=-1, dims=1)
        pair = torch.cat([feats, nxt], dim=-1)                          # (B, T, 2D)
        q = self.mlp(pair)                                              # (B, T, 4)
        q = F.normalize(q, dim=-1, eps=1e-6)
        return q                                                        # (B, T, 4)


class DepthCNNLSTMQCC(DepthCNNLSTM):
    """DepthCNNLSTM + QCC aux loss over per-frame CNN feats.

    Predicts q_t^{t+1} for all t (cyclically), composes around the full cycle,
    and pushes the result toward +/- identity. Aux loss weighted by qcc_weight.
    """

    def __init__(
        self,
        num_classes=25,
        in_channels=1,
        feat_dim=256,
        lstm_hidden=256,
        lstm_layers=2,
        bidirectional=True,
        dropout=0.3,
        qcc_hidden=64,
        qcc_weight=0.1,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, in_channels=in_channels, feat_dim=feat_dim,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            bidirectional=bidirectional, dropout=dropout,
        )
        self.qcc_head = QCCHead(feat_dim=feat_dim, hidden=qcc_hidden)
        self.qcc_weight = qcc_weight
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs["depth"]
        x = inputs.float()
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)                                              # (B*T, D)
        feat = feat.view(B, T, -1)                                      # (B, T, D)

        # --- QCC aux on CNN features ---
        if self.training and self.qcc_weight > 0:
            q = self.qcc_head(feat)                                     # (B, T, 4)
            q_comp = q[:, 0]
            for t in range(1, T):
                q_comp = quat_mul(q_comp, q[:, t])
            # Push composed |w| toward 1 (rotation angle toward 0 or 2*pi).
            w_sq = q_comp[:, 0] ** 2                                    # (B,)
            qcc_raw = (1.0 - w_sq).mean()
            self.latest_aux_loss = self.qcc_weight * qcc_raw
            # Reported angle (rad) for logging: 2 * arccos(|w|)
            with torch.no_grad():
                abs_w = q_comp[:, 0].abs().clamp(0.0, 1.0)
                angle_mean = (2.0 * torch.arccos(abs_w)).mean()
                self.latest_aux_metrics = {
                    'qcc_raw': qcc_raw.detach(),
                    'qcc_forward': qcc_raw.detach(),
                    'qcc_backward': angle_mean.detach(),
                    'qcc_valid_ratio': torch.ones(1, device=qcc_raw.device),
                }
        else:
            self.latest_aux_loss = None
            self.latest_aux_metrics = {}

        # --- classification path unchanged ---
        seq, _ = self.lstm(feat)
        t_mean = seq.mean(dim=1)
        t_max = seq.max(dim=1).values
        pooled = torch.cat([t_mean, t_max], dim=-1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics


class PartQCCHead(nn.Module):
    """Partwise quaternion predictor.

    Pool 7x7 feat map into K=6 parts (2 rows x 3 cols), predict q_k^{t->t+1} per
    part per frame-pair via a shared MLP.
    """

    def __init__(self, feat_dim, hidden=64, part_grid=(2, 3)):
        super().__init__()
        self.part_grid = part_grid
        self.num_parts = part_grid[0] * part_grid[1]
        self.part_pool = nn.AdaptiveAvgPool2d(part_grid)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def extract_parts(self, feat_map):
        # feat_map: (B*T, D, 7, 7) -> (B*T, D, Ph, Pw) -> (B*T, K, D)
        pooled = self.part_pool(feat_map)                       # (B*T, D, Ph, Pw)
        b_t, D, Ph, Pw = pooled.shape
        return pooled.view(b_t, D, Ph * Pw).transpose(1, 2).contiguous()

    def forward(self, part_feats):
        # part_feats: (B, T, K, D)
        nxt = torch.roll(part_feats, shifts=-1, dims=1)         # (B, T, K, D)
        pair = torch.cat([part_feats, nxt], dim=-1)             # (B, T, K, 2D)
        q = self.mlp(pair)                                      # (B, T, K, 4)
        return F.normalize(q, dim=-1, eps=1e-6)


def quat_from_vectors(v1, v2, eps=1e-6):
    """Shortest-arc unit quaternion rotating v1 to v2. Inputs (..., 3).

    Antipodal case (v1 ~ -v2) is left unresolved; F.normalize folds it to the
    identity, which is fine as a rare degenerate. For our use case (mean-tops
    directions in adjacent frames) antipodal never occurs in practice.
    """
    v1 = F.normalize(v1, dim=-1, eps=eps)
    v2 = F.normalize(v2, dim=-1, eps=eps)
    d = (v1 * v2).sum(-1, keepdim=True)                         # cos(theta)
    axis = torch.cross(v1, v2, dim=-1)                          # sin(theta) * n
    w = 1.0 + d                                                 # 2 * cos^2(theta/2)
    q = torch.cat([w, axis], dim=-1)                            # (..., 4) [w, x, y, z]
    return F.normalize(q, dim=-1, eps=eps)


class DepthCNNLSTMTopsQCC(DepthCNNLSTM):
    """Option-C QCC: tops-anchored quaternion supervision.

    Observable target: per frame, mean tops direction m_t (unit vector over
    hand mask). Shortest-arc rotation m_t -> m_{t+1} gives q_obs_t. QCCHead
    predicts q_pred_t from (CNN feat_t, CNN feat_{t+1}) and is supervised to
    match q_obs_t via 1 - (q_pred . q_obs)^2 (sign-ambiguous, unit-quat safe).

    Because q_obs is a real rotation signal and usually non-identity, the
    trivial identity-output collapse of Options A/B is not a valid minimum.

    Requires in_channels >= 4 (depth + tops 3ch, uses channels 1:4 as tops).
    """

    def __init__(
        self,
        num_classes=25,
        in_channels=4,
        feat_dim=256,
        lstm_hidden=256,
        lstm_layers=2,
        bidirectional=True,
        dropout=0.3,
        qcc_hidden=64,
        qcc_weight=0.1,
        **kwargs,
    ):
        assert in_channels >= 4, "tops-QCC needs tops channels (input 1:4)"
        super().__init__(
            num_classes=num_classes, in_channels=in_channels, feat_dim=feat_dim,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            bidirectional=bidirectional, dropout=dropout,
        )
        self.qcc_head = QCCHead(feat_dim=feat_dim, hidden=qcc_hidden)
        self.qcc_weight = qcc_weight
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def _mean_tops_per_frame(self, inputs):
        """inputs: (B, T, C, H, W) with channels 1:4 = tops. Returns (B, T, 3)."""
        tops = inputs[:, :, 1:4]                                # (B, T, 3, H, W)
        m = tops.sum(dim=(-2, -1))                              # (B, T, 3), zero outside mask
        return m                                                # unnormalized (ok for quat_from_vectors)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs["depth"]
        x = inputs.float()
        B, T, C, H, W = x.shape

        # Classifier path
        x_flat = x.view(B * T, C, H, W)
        feat = self.cnn(x_flat).view(B, T, -1)                  # (B, T, D)

        # --- Option-C QCC aux ---
        if self.training and self.qcc_weight > 0:
            # Head prediction
            q_pred = self.qcc_head(feat)                        # (B, T, 4)

            # Observable target: rotation from m_t to m_{t+1} (cyclic)
            with torch.no_grad():
                m = self._mean_tops_per_frame(x)                # (B, T, 3)
                m_next = torch.roll(m, shifts=-1, dims=1)
                # frames with zero tops magnitude -> mark invalid
                valid = (m.norm(dim=-1) > 1e-4) & (m_next.norm(dim=-1) > 1e-4)
                q_obs = quat_from_vectors(m, m_next)            # (B, T, 4)

            # Loss: 1 - (q_pred . q_obs)^2  (sign-invariant cos^2)
            dot = (q_pred * q_obs).sum(-1)                      # (B, T)
            pair_loss = 1.0 - dot ** 2                          # (B, T)
            valid_f = valid.float()
            denom = valid_f.sum().clamp(min=1.0)
            qcc_raw = (pair_loss * valid_f).sum() / denom
            self.latest_aux_loss = self.qcc_weight * qcc_raw

            with torch.no_grad():
                # Angle between predicted and observed (rad)
                abs_dot = dot.abs().clamp(0.0, 1.0)
                angle_mean = (2.0 * torch.arccos(abs_dot)) * valid_f
                angle_mean = angle_mean.sum() / denom
                valid_ratio = valid_f.mean()
                # How non-trivial the observed rotations are (sanity check)
                obs_nontriv = (1.0 - q_obs[..., 0] ** 2) * valid_f
                obs_nontriv = obs_nontriv.sum() / denom
                self.latest_aux_metrics = {
                    'qcc_raw': qcc_raw.detach(),
                    'qcc_forward': qcc_raw.detach(),
                    'qcc_backward': angle_mean.detach(),
                    'qcc_valid_ratio': obs_nontriv.detach(),
                }
        else:
            self.latest_aux_loss = None
            self.latest_aux_metrics = {}

        # Classifier continues
        seq, _ = self.lstm(feat)
        t_mean = seq.mean(dim=1)
        t_max = seq.max(dim=1).values
        pooled = torch.cat([t_mean, t_max], dim=-1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics


class DepthCNNLSTMPartQCC(DepthCNNLSTM):
    """Option-B QCC: per-part cycle consistency on CNN 7x7 feat grid.

    Partition spatial feat into K=6 parts, predict q_k^{t->t+1} per part with a
    shared MLP, compose per-part cycle around T frames, push composed toward
    identity (|w|^2 -> 1). Classifier path unchanged (uses global AvgPool).
    """

    def __init__(
        self,
        num_classes=25,
        in_channels=1,
        feat_dim=256,
        lstm_hidden=256,
        lstm_layers=2,
        bidirectional=True,
        dropout=0.3,
        qcc_hidden=64,
        qcc_weight=0.1,
        part_grid=(2, 3),
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, in_channels=in_channels, feat_dim=feat_dim,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            bidirectional=bidirectional, dropout=dropout,
        )
        self.qcc_head = PartQCCHead(feat_dim=feat_dim, hidden=qcc_hidden, part_grid=tuple(part_grid))
        self.num_parts = self.qcc_head.num_parts
        self.qcc_weight = qcc_weight
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs["depth"]
        x = inputs.float()
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat_map = self.cnn.forward_features(x)                 # (B*T, D, 7, 7)

        # Classifier path: global pool -> LSTM -> classifier
        pooled = self.cnn.pool(feat_map).flatten(1)             # (B*T, D)
        feat = pooled.view(B, T, -1)                            # (B, T, D)

        # QCC path: partition -> per-part quat -> cycle
        if self.training and self.qcc_weight > 0:
            parts = self.qcc_head.extract_parts(feat_map)       # (B*T, K, D)
            K = parts.shape[1]
            parts = parts.view(B, T, K, -1)                     # (B, T, K, D)
            q = self.qcc_head(parts)                            # (B, T, K, 4)

            # Compose per part around the cycle
            q_comp = q[:, 0]                                    # (B, K, 4)
            for t in range(1, T):
                q_comp = quat_mul(q_comp, q[:, t])

            w_sq = q_comp[..., 0] ** 2                          # (B, K)
            qcc_raw = (1.0 - w_sq).mean()
            self.latest_aux_loss = self.qcc_weight * qcc_raw

            with torch.no_grad():
                abs_w = q_comp[..., 0].abs().clamp(0.0, 1.0)
                angle_mean = (2.0 * torch.arccos(abs_w)).mean()
                # Diagnostic: per-step avg non-identity pressure (diagnostic only,
                # not added to loss). High -> individual q's depart from [1,0,0,0].
                step_nontrivial = (1.0 - q[..., 0] ** 2).mean()
                self.latest_aux_metrics = {
                    'qcc_raw': qcc_raw.detach(),
                    'qcc_forward': qcc_raw.detach(),
                    'qcc_backward': angle_mean.detach(),
                    'qcc_valid_ratio': step_nontrivial.detach(),
                }
        else:
            self.latest_aux_loss = None
            self.latest_aux_metrics = {}

        # Classifier continues from pooled features
        seq, _ = self.lstm(feat)
        t_mean = seq.mean(dim=1)
        t_max = seq.max(dim=1).values
        pooled_time = torch.cat([t_mean, t_max], dim=-1)
        pooled_time = self.dropout(pooled_time)
        return self.classifier(pooled_time)

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics
