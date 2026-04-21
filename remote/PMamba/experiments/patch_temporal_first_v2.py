"""Fix TemporalFirstMambaMotion: correspondence-guided sampling so per-point
temporal Mamba sees a coherent trajectory (point i at frame t = same physical
point across t=0..T-1)."""
from pathlib import Path

PATH = Path("models/motion.py")
src = PATH.read_text(encoding="utf-8")

# Remove old class (from 'class TemporalFirstMambaMotion' to end of file).
anchor = "\nclass TemporalFirstMambaMotion(nn.Module):"
idx = src.find(anchor)
assert idx > 0, "anchor missing"
src = src[:idx].rstrip() + "\n"

new_class = '''

class TemporalFirstMambaMotion(nn.Module):
    """Per-point Mamba over time first, then set pool across points.

    Requires correspondence-aware dataloader (NvidiaQuaternionQCCParityLoader)
    so point i at frame t is the same physical point at frame t+1. Without
    correspondence, per-point temporal signal is noise (stays at chance).
    """

    def __init__(self, num_classes=25, pts_size=96, in_channels=5,
                 hidden=128, mamba_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.pts_size = pts_size
        self.in_channels = in_channels
        self.hidden = hidden

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
        )
        self.temporal = MambaTemporalEncoder(
            in_channels=hidden, hidden_dim=hidden, output_dim=hidden,
            num_layers=mamba_layers,
        )
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden * 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 4, num_classes)

    def _correspondence_guided_sample(self, points, aux_input):
        """Sample points following correspondence chains across frames.

        Duplicated from BearingQCCFeatureMotion so this class stays standalone.
        Frame 0 sampled random (train) or uniform (eval); each next frame
        follows the correspondence target of the previous frame's sample.
        """
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device

        if sample_size == pts_per_frame:
            return points

        orig_flat_idx = aux_input['orig_flat_idx']
        corr_target = aux_input['corr_full_target_idx']
        corr_weight = aux_input['corr_full_weight']
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames

        sampled = torch.zeros(batch_size, num_frames, sample_size, channels,
                              device=device, dtype=points.dtype)

        for b in range(batch_size):
            if self.training:
                idx = torch.randperm(pts_per_frame, device=device)[:sample_size]
            else:
                idx = torch.linspace(0, pts_per_frame - 1, sample_size,
                                     device=device).long()
            sampled[b, 0] = points[b, 0, idx]
            current_prov = orig_flat_idx[b, 0, idx].long()

            for t in range(num_frames - 1):
                next_prov = orig_flat_idx[b, t + 1].long()
                reverse_map = torch.full((total_pts,), -1, dtype=torch.long, device=device)
                reverse_map[next_prov] = torch.arange(pts_per_frame, device=device)

                tgt_flat = corr_target[b, current_prov]
                tgt_w = corr_weight[b, current_prov]
                tgt_flat_safe = tgt_flat.clamp(min=0)
                tgt_frame = tgt_flat // raw_ppf
                tgt_pos = reverse_map[tgt_flat_safe]

                valid = ((tgt_flat >= 0) & (tgt_w > 0)
                         & (tgt_frame == t + 1) & (tgt_pos >= 0))

                next_idx = torch.randint(0, pts_per_frame, (sample_size,), device=device)
                next_idx[valid] = tgt_pos[valid]

                sampled[b, t + 1] = points[b, t + 1, next_idx]
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()

        return sampled

    def _fallback_sample(self, inputs):
        B, T, P, C = inputs.shape
        device = inputs.device
        sample_size = min(self.pts_size, P)
        if self.training:
            indices = torch.randperm(P, device=device)[:sample_size]
        else:
            indices = torch.linspace(0, P - 1, sample_size, device=device).long()
        return inputs[:, :, indices, :]

    def _polar_input(self, inputs):
        xyz = inputs[..., :3]
        time_ch = inputs[..., 3:4]
        centroid = xyz.mean(dim=2, keepdim=True)
        rel = xyz - centroid
        mag = rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = (rel / mag).detach()
        return torch.cat([direction, mag, time_ch], dim=-1)

    def forward(self, inputs):
        # Quaternion dataloader returns dict with correspondence aux.
        has_corr = False
        if isinstance(inputs, dict):
            aux = inputs
            points = inputs['points']
            has_corr = ('orig_flat_idx' in aux
                        and 'corr_full_target_idx' in aux
                        and 'corr_full_weight' in aux)
        else:
            points = inputs
            aux = None

        points = points.float()
        if points.dim() == 3:
            B, N, C = points.shape
            T = N // self.pts_size
            points = points.view(B, T, self.pts_size, C)

        if has_corr:
            sampled = self._correspondence_guided_sample(points[..., :4], aux)
        else:
            sampled = self._fallback_sample(points[..., :4])

        polar = self._polar_input(sampled)
        B, T, P, _ = polar.shape

        x = self.input_proj(polar)                               # (B, T, P, hidden)
        x = x.permute(0, 2, 3, 1).contiguous()                   # (B, P, hidden, T)
        x = x.view(B * P, self.hidden, T)
        x_4d = x.unsqueeze(-1)                                   # (B*P, hidden, T, 1)
        x_4d = self.temporal(x_4d)
        x = x_4d.squeeze(-1)                                     # (B*P, hidden, T)

        t_mean = x.mean(dim=-1)
        t_max = x.max(dim=-1).values
        per_point = torch.cat([t_mean, t_max], dim=-1)           # (B*P, 2h)
        per_point = self.spatial_mlp(per_point)
        per_point = per_point.view(B, P, -1)

        s_mean = per_point.mean(dim=1)
        s_max = per_point.max(dim=1).values
        feat = torch.cat([s_mean, s_max], dim=-1)                # (B, 4h)
        feat = self.dropout(feat)
        return self.classifier(feat)
'''

src = src.rstrip() + "\n" + new_class + "\n"
PATH.write_text(src, encoding="utf-8")
print("rewrote TemporalFirstMambaMotion with correspondence-guided sampling")
