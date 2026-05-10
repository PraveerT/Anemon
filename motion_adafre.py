"""PMamba + AdaFreBlock (HMSFT-inspired adaptive frequency gating).

Inserts a DCT-based adaptive frequency gating module after PMamba's Mamba
temporal stage. Operates on (B, C, T, P) features:
  - DCT over T dim -> frequency representation (B, C, F=T, P)
  - Spatial frequency gating: per-channel sigmoid gates from avg+max pool over (T,P)
  - Temporal frequency gating: per-frequency sigmoid gates from avg+max pool over (C,P)
  - Element-wise gate application
  - IDCT back to time domain

Theoretical claim: aux-loss prediction of frequency targets degraded N2
(86.51 < 88.59 baseline). Architectural integration via in-place gating
preserves the backbone's CE objective while exploiting frequency information.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.motion import Motion


def _dct_matrix(N, device, dtype):
    """Type-II orthonormal DCT matrix of size NxN."""
    n = torch.arange(N, device=device, dtype=dtype)
    k = n.unsqueeze(1)
    M = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
    M[0] *= 1 / math.sqrt(N)
    M[1:] *= math.sqrt(2 / N)
    return M  # (N, N)


def dct1d(x, dim=-2):
    """Apply DCT along dim. x can be any rank."""
    x = x.transpose(dim, -1)  # bring target dim to last
    N = x.shape[-1]
    M = _dct_matrix(N, x.device, x.dtype)
    out = x @ M.T
    return out.transpose(dim, -1)


def idct1d(x, dim=-2):
    """Inverse DCT (orthonormal: M^T inverts M)."""
    x = x.transpose(dim, -1)
    N = x.shape[-1]
    M = _dct_matrix(N, x.device, x.dtype)
    out = x @ M
    return out.transpose(dim, -1)


class AdaFreBlock(nn.Module):
    """Adaptive frequency gating block over the time dimension.

    Args:
        channels: number of channels C
        T: time dim length (sequence length on which DCT is applied)
        hidden: hidden dim of the gate small-MLPs
    """
    def __init__(self, channels, T, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = max(channels // 4, 16)
        # Spatial (per-channel) gating MLPs — share weights across avg/max via SE-style
        self.spatial_gate = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )
        # Temporal (per-frequency) gating MLPs
        self.temporal_gate = nn.Sequential(
            nn.Conv1d(T, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, T, kernel_size=1),
        )
        # Learnable mix weights between avg-pool and max-pool branches
        self.alpha_spatial = nn.Parameter(torch.tensor(0.5))
        self.alpha_temporal = nn.Parameter(torch.tensor(0.5))
        # Residual scale (zero-init: at start, AdaFreBlock is identity)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """x: (B, C, T, P) — features from PMamba's mamba stage."""
        Bz, C, T, P = x.shape
        # DCT over T (treat freq as F=T)
        f = dct1d(x, dim=2)            # (B, C, F, P)

        # --- spatial frequency gating: per-channel weights ---
        # pool over (F, P) -> per-channel scalar
        avg_c = f.mean(dim=(2, 3), keepdim=True).squeeze(-1)  # (B, C, 1)
        max_c = f.amax(dim=(2, 3), keepdim=True).squeeze(-1)
        sg_avg = self.spatial_gate(avg_c)  # (B, C, 1)
        sg_max = self.spatial_gate(max_c)
        a = torch.sigmoid(self.alpha_spatial)
        spatial_w = torch.sigmoid(a * sg_avg + (1 - a) * sg_max)  # (B, C, 1)
        f = f * spatial_w.unsqueeze(-1)  # broadcast over (F, P)

        # --- temporal frequency gating: per-frequency weights ---
        # pool over (C, P) -> per-frequency scalar
        avg_f = f.mean(dim=(1, 3), keepdim=True).permute(0, 2, 1, 3).squeeze(-1)  # (B, F, 1)
        max_f = f.amax(dim=(1, 3), keepdim=True).permute(0, 2, 1, 3).squeeze(-1)
        tg_avg = self.temporal_gate(avg_f)  # (B, F, 1)
        tg_max = self.temporal_gate(max_f)
        b = torch.sigmoid(self.alpha_temporal)
        temporal_w = torch.sigmoid(b * tg_avg + (1 - b) * tg_max)  # (B, F, 1)
        # broadcast over C and P; need (B, 1, F, 1)
        # temporal_w: (B, F, 1) -> (B, 1, F, 1) for broadcast over (C, P)
        f = f * temporal_w.permute(0, 2, 1).unsqueeze(-1)

        # IDCT back
        y = idct1d(f, dim=2)  # (B, C, T, P)
        # zero-init residual
        return x + self.gamma * (y - x)


class MotionAdaFre(Motion):
    """PMamba with AdaFreBlock inserted after the Mamba temporal stage."""
    def __init__(self, *args, T_for_adafre=32, **kwargs):
        super().__init__(*args, **kwargs)
        # mamba output dim is 256 in stock PMamba
        # We'll lazy-init AdaFreBlock at first forward when we know shape
        self._T_for_adafre = T_for_adafre
        self.adafre = None

    def _encode_sampled_points(self, coords):
        # Override the encode pipeline: same as parent but insert AdaFreBlock
        batchsize, in_dims, timestep, pts_num = coords.shape
        ret_array1 = self.group.group_points(distance_dim=[0, 1, 2], array1=coords, array2=coords, knn=self.knn[0], dim=3)
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1 = self.pool1(self.stage1(ret_array1)).reshape(batchsize, -1, timestep, pts_num)
        fea1 = torch.cat((coords, fea1), dim=1)
        in_dims = fea1.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[0]
        ret_group_array2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3, coord_dim=self.coord_channels)
        ret_array2, coords = self.select_ind(ret_group_array2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret_array2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)
        in_dims = fea2.shape[1] * 2 - self.coord_channels
        pts_num //= self.downsample[1]
        ret_group_array3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3, coord_dim=self.coord_channels)
        ret_array3, coords = self.select_ind(ret_group_array3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret_array3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)

        # Lazy init AdaFreBlock once we know channel count and T
        if self.adafre is None:
            C = fea3_mamba.shape[1]
            T = fea3_mamba.shape[2]
            self.adafre = AdaFreBlock(C, T).to(fea3_mamba.device)

        # Apply AdaFreBlock
        fea3_freq = self.adafre(fea3_mamba)

        return torch.cat((coords, fea3_freq), dim=1)
