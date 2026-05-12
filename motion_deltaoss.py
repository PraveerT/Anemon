"""DeltaOSS: alternating Gated DeltaNet (state-tracking) + LinOSS (oscillatory) blocks.

Hypothesis: DeltaNet captures fine-grained state transitions, LinOSS captures
periodic / rhythmic patterns. Alternating gives both biases per pair of layers.

Stack pattern (D-L-D-L by default): each pair = (DeltaNet layer, LinOSS layer).
Both bidirectional.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import Motion
from models.motion_deltanet_v2 import GatedDeltaNetV2Block
from models.motion_linoss import LinOSSBlock


class DeltaOSSTemporalEncoder(nn.Module):
    """Alternating Delta + LinOSS stack, bidirectional. Drop-in for MambaTemporalEncoder."""
    def __init__(self, in_channels, hidden_dim=256, output_dim=None, num_pairs=2,
                 num_heads=4, head_dim=64, expand_v=2, n_state=128,
                 use_short_conv=True, conv_size=4, dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.bidirectional = bidirectional
        self.num_pairs = num_pairs
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        def build_stack():
            layers = []
            for _ in range(num_pairs):
                layers.append(GatedDeltaNetV2Block(hidden_dim, num_heads=num_heads,
                                                   head_dim=head_dim, expand_v=expand_v,
                                                   use_short_conv=use_short_conv,
                                                   conv_size=conv_size, dropout=dropout))
                layers.append(LinOSSBlock(hidden_dim, n_state=n_state, dropout=dropout))
            return nn.ModuleList(layers)

        self.fwd_layers = build_stack()
        self.fwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2 * num_pairs)])
        if bidirectional:
            self.bwd_layers = build_stack()
            self.bwd_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2 * num_pairs)])
        self.dropout = nn.Dropout(dropout)
        proj_in = 2 * hidden_dim if bidirectional else hidden_dim
        self.final_norm = nn.LayerNorm(proj_in)
        self.output_proj = nn.Linear(proj_in, self.output_dim)

    def _stack(self, x, layers, norms):
        for blk, norm in zip(layers, norms):
            r = x
            x = norm(x)
            x = blk(x)
            x = self.dropout(x) + r
        return x

    def forward(self, x):
        B, C, T, N = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * N, T, C)
        x = self.input_proj(x)
        fwd = self._stack(x, self.fwd_layers, self.fwd_norms)
        if self.bidirectional:
            bwd = self._stack(torch.flip(x, dims=[1]), self.bwd_layers, self.bwd_norms)
            bwd = torch.flip(bwd, dims=[1])
            x = torch.cat([fwd, bwd], dim=-1)
        else:
            x = fwd
        x = self.final_norm(x)
        x = self.output_proj(x)
        x = x.reshape(B, N, T, self.output_dim).permute(0, 3, 2, 1)
        return x


class MotionDeltaOSS(Motion):
    def __init__(self, *args, do_hidden_dim=256, do_num_pairs=2, do_num_heads=4,
                 do_head_dim=64, do_expand_v=2, do_n_state=128,
                 do_dropout=0.3, do_bidirectional=True, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.mamba
        in_c = old.in_channels
        out_d = old.output_dim
        self.mamba = DeltaOSSTemporalEncoder(
            in_channels=in_c,
            hidden_dim=do_hidden_dim,
            output_dim=out_d,
            num_pairs=do_num_pairs,
            num_heads=do_num_heads,
            head_dim=do_head_dim,
            expand_v=do_expand_v,
            n_state=do_n_state,
            dropout=do_dropout,
            bidirectional=do_bidirectional,
        )
