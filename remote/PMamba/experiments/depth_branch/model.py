"""Depth CNN-LSTM model.

Per-frame 2D CNN encoder -> BiLSTM over time -> mean+max pool -> classifier.
Input:  (B, T, 1, H, W) float32 in [0, 1]
Output: (B, num_classes) logits
"""
import torch
import torch.nn as nn


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
        )                                                    # 112 -> 56 -> 28
        self.block1 = ConvBlock(32, 64, stride=2)            # 28 -> 14
        self.block2 = ConvBlock(64, 128, stride=2)           # 14 -> 7
        self.block3 = ConvBlock(128, feat_dim, stride=1)     # 7 -> 7
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return x.flatten(1)                                  # (B, feat_dim)


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
        self.classifier = nn.Linear(lstm_hidden * mult * 2, num_classes)  # mean+max

    def forward(self, inputs):
        # Accept dict or tensor; depth branch always uses tensor
        if isinstance(inputs, dict):
            inputs = inputs["depth"]
        x = inputs.float()                                   # (B, T, 1, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)                                   # (B*T, feat_dim)
        feat = feat.view(B, T, -1)                           # (B, T, feat_dim)

        seq, _ = self.lstm(feat)                             # (B, T, 2*hidden)
        t_mean = seq.mean(dim=1)
        t_max = seq.max(dim=1).values
        pooled = torch.cat([t_mean, t_max], dim=-1)          # (B, 4*hidden)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
