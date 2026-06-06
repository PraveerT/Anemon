"""PGCNetGridT2S -- decoupled TEMPORAL-then-SPATIAL on a grid feature-video.

Raw depth points have no cross-frame correspondence (verified: index-matched
frame-to-frame displacement > distance to a random same-frame point), so a
per-point temporal trajectory is noise. We instead get frame-consistent
structure by binning (u,v) into an HxW grid per CLIP (same cell = same spatial
region across all T frames). Each cell aggregates its points' embeddings ->
a sparse feature-video (B, T, d, H, W). Then:
  1) TEMPORAL FIRST: Conv1d over time, independently per cell.
  2) SPATIAL SECOND: Conv2d over the HxW grid of time-pooled features.
-> global pool -> classifier. Tests whether processing motion before structure
(the opposite order of the kNN-spatial-first backbone) helps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGCNetGridT2S(nn.Module):
    def __init__(self, num_classes=25, grid=16, embed_dim=32, t_dim=64,
                 s_dim=128, dropout=0.3, **kwargs):
        super().__init__()
        self.num_classes = int(num_classes)
        self.H = self.W = int(grid)
        self.embed = nn.Sequential(
            nn.Linear(3, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))
        self.temporal = nn.Sequential(                       # per-cell, over T
            nn.Conv1d(embed_dim, t_dim, 3, padding=1), nn.BatchNorm1d(t_dim), nn.ReLU(inplace=True),
            nn.Conv1d(t_dim, t_dim, 3, padding=1), nn.BatchNorm1d(t_dim), nn.ReLU(inplace=True))
        self.spatial = nn.Sequential(                        # over HxW grid
            nn.Conv2d(t_dim, s_dim, 3, padding=1), nn.BatchNorm2d(s_dim), nn.ReLU(inplace=True),
            nn.Conv2d(s_dim, s_dim, 3, padding=1), nn.BatchNorm2d(s_dim), nn.ReLU(inplace=True))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(s_dim, self.num_classes)

    def _bin(self, c):                                        # c: (B,T,N) -> cell idx per axis
        lo = c.amin(dim=(1, 2), keepdim=True)
        hi = c.amax(dim=(1, 2), keepdim=True)
        n = (c - lo) / (hi - lo + 1e-6)                       # [0,1] per clip (frame-consistent)
        return (n * self.H).long().clamp(0, self.H - 1)

    def _grid(self, x):                                       # x: (B,T,N,C) -> (B,T,HW,d)
        B, T, N, C = x.shape
        H = W = self.H
        iu = self._bin(x[..., 0])
        iv = self._bin(x[..., 1])
        cell = (iv * W + iu)                                  # (B,T,N) in [0,HW)
        feat = self.embed(x[..., :3])                         # (B,T,N,d)
        d = feat.shape[-1]
        gsum = feat.new_zeros(B, T, H * W, d)
        gsum.scatter_add_(2, cell.unsqueeze(-1).expand(-1, -1, -1, d), feat)
        cnt = feat.new_zeros(B, T, H * W, 1)
        cnt.scatter_add_(2, cell.unsqueeze(-1), feat.new_ones(B, T, N, 1))
        return gsum / cnt.clamp(min=1.0)                      # (B,T,HW,d) mean-pool, empty->0

    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        x = inputs
        B, T, N, C = x.shape
        H = W = self.H
        grid = self._grid(x)                                  # (B,T,HW,d)
        d = grid.shape[-1]
        z = grid.permute(0, 2, 3, 1).reshape(B * H * W, d, T) # (B*HW, d, T)
        z = self.temporal(z)                                 # TEMPORAL FIRST -> (B*HW, t_dim, T)
        z = z.max(dim=2).values                              # pool over time -> (B*HW, t_dim)
        z = z.reshape(B, H, W, -1).permute(0, 3, 1, 2)       # (B, t_dim, H, W)
        z = self.spatial(z)                                  # SPATIAL SECOND -> (B, s_dim, H, W)
        z = z.amax(dim=(2, 3))                               # global pool -> (B, s_dim)
        return self.fc(self.drop(z))

    def extract_features(self, inputs):
        return self.forward(inputs)
