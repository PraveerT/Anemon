"""TemporalFirstMamba model: per-point Mamba across time FIRST, then spatial pool.

Structurally inverts PMamba's spatial-first-then-temporal bias. Same polar
input as v27 polar model (tops + magnitude + time) so we can compare the
pure architectural effect.

Adds class TemporalFirstMambaMotion to models/motion.py.
"""
from pathlib import Path

PATH = Path("models/motion.py")
src = PATH.read_text(encoding="utf-8")

new_class = '''

class TemporalFirstMambaMotion(nn.Module):
    """Per-point Mamba over time first, then set pool across points."""

    def __init__(self, num_classes=25, pts_size=96, in_channels=5,
                 hidden=128, mamba_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.pts_size = pts_size
        self.in_channels = in_channels
        self.hidden = hidden

        # Input projection: 5 -> hidden per (point, time)
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
        )

        # Per-point temporal Mamba. Uses MambaTemporalEncoder already in the file.
        self.temporal = MambaTemporalEncoder(
            in_channels=hidden, hidden_dim=hidden, output_dim=hidden,
            num_layers=mamba_layers,
        )

        # Spatial set aggregation: shared MLP per point + mean+max pool.
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 4, num_classes)  # mean + max = 2*hidden*2

    def _sample_points(self, inputs):
        # inputs: (B, T, P, C). Sample pts_size points (uniform stride, same-order across frames).
        B, T, P, C = inputs.shape
        device = inputs.device
        sample_size = min(self.pts_size, P)
        if self.training:
            indices = torch.randperm(P, device=device)[:sample_size]
        else:
            indices = torch.linspace(0, P - 1, sample_size, device=device).long()
        return inputs[:, :, indices, :]

    def _polar_input(self, inputs):
        """inputs: (B, T, P, C>=4) with xyz+time. Returns (B, T, P, 5) polar."""
        xyz = inputs[..., :3]
        time_ch = inputs[..., 3:4]
        centroid = xyz.mean(dim=2, keepdim=True)
        rel = xyz - centroid
        mag = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = (rel / mag).detach()
        return torch.cat([direction, mag, time_ch], dim=-1)             # (B, T, P, 5)

    def forward(self, inputs):
        # Handle dict inputs (quaternion dataloader) and raw tensor (NvidiaLoader).
        if isinstance(inputs, dict):
            inputs = inputs['points']
        if inputs.dim() == 3:  # (B, T*P, C)
            B, N, C = inputs.shape
            T = N // self.pts_size
            inputs = inputs.view(B, T, self.pts_size, C)

        sampled = self._sample_points(inputs[..., :4].float())          # (B, T, P, 4)
        polar = self._polar_input(sampled)                               # (B, T, P, 5)
        B, T, P, _ = polar.shape

        # Per-point projection
        x = self.input_proj(polar)                                       # (B, T, P, hidden)

        # Per-point temporal Mamba: (B*P, hidden, T)
        x = x.permute(0, 2, 3, 1).contiguous()                           # (B, P, hidden, T)
        x = x.view(B * P, self.hidden, T)
        # MambaTemporalEncoder expects 4D input (B, C, T, N). Emulate by wrapping.
        # Simpler: call directly with 4D shape (BP, C, T, 1).
        x_4d = x.unsqueeze(-1)                                           # (B*P, hidden, T, 1)
        x_4d = self.temporal(x_4d)                                       # (B*P, hidden, T, 1)
        x = x_4d.squeeze(-1)                                             # (B*P, hidden, T)

        # Pool over time: mean + max
        t_mean = x.mean(dim=-1)                                          # (B*P, hidden)
        t_max = x.max(dim=-1).values                                     # (B*P, hidden)
        per_point = torch.cat([t_mean, t_max], dim=-1)                   # (B*P, 2*hidden)
        per_point = self.spatial_mlp(per_point)                          # (B*P, 2*hidden)
        per_point = per_point.view(B, P, -1)                             # (B, P, 2*hidden)

        # Spatial set aggregation: mean + max
        s_mean = per_point.mean(dim=1)
        s_max = per_point.max(dim=1).values
        feat = torch.cat([s_mean, s_max], dim=-1)                        # (B, 4*hidden)
        feat = self.dropout(feat)
        return self.classifier(feat)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("added TemporalFirstMambaMotion")
