"""BDN-Q as upstream preprocessor on raw input + RD as the temporal encoder.

Architecture:
  raw (B, T, N, C_raw)
    -> permute to (B, C_raw, T, N)
    -> BDN-Q (per-point temporal scan, hidden=H_pre, output_dim=C_raw)
    -> permute back to (B, T, N, C_raw)
    -> PMamba spatial pipeline (_sample_points + _encode_sampled_points)
    -> RD temporal encoder (self.mamba)
    -> classifier

The BDN-Q preprocessor sees length-T per-point trajectories on raw 8-channel
features (no spatial encoding). Subsequent stages get a "smoothed" version of
the raw input. RD handles the actual temporal recurrence as usual.

Tests whether the buffer hybrid contributes orthogonal signal when applied to
raw features (different role than replacing the temporal encoder).
"""
import torch
import torch.nn as nn

from models.motion import Motion
from models.motion_bdn_q import BDeltaQTemporalEncoder
from models.motion_realdeltanet import RealDeltaNetTemporalEncoder


class MotionBDNQUpstreamRD(Motion):
    """PMamba with BDN-Q preprocessor on raw input + RD as temporal encoder."""
    def __init__(
        self,
        *args,
        # Upstream BDN-Q config (small: operates on raw 8-channel features)
        bdnq_pre_hidden_dim=32,
        bdnq_pre_num_layers=1,
        bdnq_pre_num_heads=4,
        bdnq_pre_n_q=2,
        bdnq_pre_n_v=2,
        bdnq_pre_buffer_size=1,
        bdnq_pre_dropout=0.3,
        bdnq_pre_bidirectional=True,
        bdnq_pre_raw_channels=8,
        # Downstream RD config (same defaults as motion_realdeltanet)
        rd_hidden_dim=128,
        rd_num_layers=2,
        rd_num_heads=4,
        rd_n_q=4,
        rd_n_v=8,
        rd_dropout=0.3,
        rd_bidirectional=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Upstream BDN-Q operating on raw features
        self.bdnq_pre = BDeltaQTemporalEncoder(
            in_channels=bdnq_pre_raw_channels,
            hidden_dim=bdnq_pre_hidden_dim,
            output_dim=bdnq_pre_raw_channels,
            num_layers=bdnq_pre_num_layers,
            num_heads=bdnq_pre_num_heads,
            n_q=bdnq_pre_n_q,
            n_v=bdnq_pre_n_v,
            buffer_size=bdnq_pre_buffer_size,
            dropout=bdnq_pre_dropout,
            bidirectional=bdnq_pre_bidirectional,
            scan_axis='T',
        )
        # Replace self.mamba with RD
        old = self.mamba
        self.mamba = RealDeltaNetTemporalEncoder(
            in_channels=old.in_channels,
            hidden_dim=rd_hidden_dim,
            output_dim=old.output_dim,
            num_layers=rd_num_layers,
            num_heads=rd_num_heads,
            n_q=rd_n_q,
            n_v=rd_n_v,
            dropout=rd_dropout,
            bidirectional=rd_bidirectional,
        )
        self._raw_C = bdnq_pre_raw_channels

    def _preprocess_raw(self, inputs):
        """inputs: (B, T, N, C_raw). Returns same shape after BDN-Q."""
        # BDeltaQTemporalEncoder expects (B, C, T, N).
        x = inputs.permute(0, 3, 1, 2).contiguous()      # (B, C, T, N)
        # If channel count differs (e.g. dataset gives 4 channels), pad/truncate.
        # Standard NVGesture loader emits 8 channels.
        if x.shape[1] != self._raw_C:
            # Truncate or pad with zeros along channel dim
            B, C, T, N = x.shape
            if C > self._raw_C:
                x = x[:, : self._raw_C]
            else:
                pad = torch.zeros(B, self._raw_C - C, T, N, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
        x = self.bdnq_pre(x)                              # (B, C_raw, T, N)
        x = x.permute(0, 2, 3, 1).contiguous()            # (B, T, N, C_raw)
        return x

    def extract_features(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs['points']
        # Upstream BDN-Q on raw frames
        inputs = self._preprocess_raw(inputs)
        coords = self._sample_points(inputs)
        fea3 = self._encode_sampled_points(coords)
        output = self.stage5(fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)
