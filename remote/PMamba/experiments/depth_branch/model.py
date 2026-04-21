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

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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
        **kwargs,
    ):
        super().__init__()
        self.cnn = DepthCNN(in_channels=in_channels, feat_dim=feat_dim)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        mult = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * mult * 2, num_classes)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs["depth"]
        x = inputs.float()
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)
        feat = feat.view(B, T, -1)

        seq, _ = self.lstm(feat)
        t_mean = seq.mean(dim=1)
        t_max = seq.max(dim=1).values
        pooled = torch.cat([t_mean, t_max], dim=-1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


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
