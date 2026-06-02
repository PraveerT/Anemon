"""GPU-batched point-cloud augmentations.

Mirrors the CPU pipeline in utils.pts_transform but batched over (B, T*N, C).
Disabled when self.training is False (no-op for eval).
"""
import torch
import torch.nn as nn


class GpuAugmentor(nn.Module):
    def __init__(self,
                 scale_lo=0.85, scale_hi=1.15,
                 angle_sigma=0.08, angle_clip=0.2,
                 translate_range=0.1,
                 jitter_std=0.015, jitter_clip=0.06,
                 dropout_max=0.25,
                 speed_range=(0.85, 1.15), speed_prob=0.3,
                 tt_max_shift=0.2, tt_prob=0.4,
                 tc_max_ratio=0.2, tc_num_holes=(1, 4), tc_prob=0.5,
                 ts_window=7, ts_num=4, ts_prob=0.4):
        super().__init__()
        self.scale_lo, self.scale_hi = scale_lo, scale_hi
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip
        self.translate_range = translate_range
        self.jitter_std, self.jitter_clip = jitter_std, jitter_clip
        self.dropout_max = dropout_max
        self.speed_range, self.speed_prob = speed_range, speed_prob
        self.tt_max_shift, self.tt_prob = tt_max_shift, tt_prob
        self.tc_max_ratio = tc_max_ratio
        self.tc_num_holes = tc_num_holes
        self.tc_prob = tc_prob
        self.ts_window, self.ts_num, self.ts_prob = ts_window, ts_num, ts_prob

    @torch.no_grad()
    def forward(self, x):
        # x: (B, T, N, C); aug only during training
        if not self.training:
            return x
        B, T, N, C = x.shape
        flat = x.reshape(B, T * N, C).contiguous()

        flat = self._scale(flat)
        flat = self._rotate(flat)
        flat = self._translate(flat)
        flat = self._temporal_speed(flat)
        flat = self._temporal_translate(flat)
        flat = self._temporal_cutout(flat)
        flat = self._temporal_shuffle(flat)
        flat = self._jitter(flat)
        flat = self._dropout(flat)

        return flat.reshape(B, T, N, C)

    # ------------------ spatial augs (fully batched) ------------------

    def _scale(self, x):
        B = x.shape[0]
        s = torch.empty(B, 1, 1, device=x.device).uniform_(self.scale_lo, self.scale_hi)
        out = x.clone()
        out[..., :3] = out[..., :3] * s
        return out

    def _rotate(self, x):
        B, dev = x.shape[0], x.device
        angles = torch.clamp(
            self.angle_sigma * torch.randn(B, 3, device=dev),
            -self.angle_clip, self.angle_clip,
        )
        c = torch.cos(angles); s = torch.sin(angles)
        ones = torch.ones(B, device=dev); zeros = torch.zeros(B, device=dev)
        Rx = torch.stack([
            torch.stack([ones,       zeros,    zeros],    dim=1),
            torch.stack([zeros,      c[:, 0], -s[:, 0]],  dim=1),
            torch.stack([zeros,      s[:, 0],  c[:, 0]],  dim=1),
        ], dim=1)
        Ry = torch.stack([
            torch.stack([c[:, 1],    zeros,    s[:, 1]],  dim=1),
            torch.stack([zeros,      ones,     zeros],    dim=1),
            torch.stack([-s[:, 1],   zeros,    c[:, 1]],  dim=1),
        ], dim=1)
        Rz = torch.stack([
            torch.stack([c[:, 2],   -s[:, 2],  zeros],    dim=1),
            torch.stack([s[:, 2],    c[:, 2],  zeros],    dim=1),
            torch.stack([zeros,      zeros,    ones],     dim=1),
        ], dim=1)
        R = torch.bmm(torch.bmm(Rz, Ry), Rx)  # (B,3,3)
        out = x.clone()
        out[..., :3] = torch.bmm(out[..., :3], R.transpose(1, 2))
        return out

    def _translate(self, x):
        B = x.shape[0]
        t = torch.empty(B, 1, 3, device=x.device).uniform_(
            -self.translate_range, self.translate_range
        )
        out = x.clone()
        out[..., :3] = out[..., :3] + t
        return out

    def _jitter(self, x):
        out = x.clone()
        noise = torch.clamp(
            torch.randn_like(out[..., :3]) * self.jitter_std,
            -self.jitter_clip, self.jitter_clip,
        )
        out[..., :3] = out[..., :3] + noise
        return out

    def _dropout(self, x):
        # Replace dropped rows with prev-row values; per-sample mask.
        B, M, C = x.shape
        out = x.clone()
        for b in range(B):
            ratio = torch.rand(1, device=x.device).item() * self.dropout_max
            mask = torch.rand(M, device=x.device) <= ratio
            mask[0] = False
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                out[b, idx] = out[b, idx - 1]
        return out

    # ------------------ temporal augs (per-sample loop) ------------------

    def _temporal_speed(self, x):
        B, M, _ = x.shape
        lo, hi = self.speed_range
        out_list = []
        for b in range(B):
            xb = x[b]
            if torch.rand(1).item() > self.speed_prob:
                out_list.append(xb); continue
            speed = lo + torch.rand(1).item() * (hi - lo)
            if speed < 1.0:
                num_dup = int(M * (1 - speed) * 0.1)
                if num_dup > 0:
                    dup_idx = torch.randperm(M, device=x.device)[:num_dup]
                    xb = torch.cat([xb, xb[dup_idx]], dim=0)[:M]
            else:
                skip_ratio = min(0.2, speed - 1.0)
                num_skip = int(M * skip_ratio)
                if num_skip > 0:
                    skip_idx = torch.randperm(M, device=x.device)[:num_skip]
                    keep = torch.ones(M, dtype=torch.bool, device=x.device)
                    keep[skip_idx] = False
                    kept = xb[keep]
                    while kept.shape[0] < M:
                        pad = kept[: min(kept.shape[0], M - kept.shape[0])]
                        kept = torch.cat([kept, pad], dim=0)
                    xb = kept[:M]
            out_list.append(xb)
        return torch.stack(out_list, dim=0)

    def _temporal_translate(self, x):
        B, M, _ = x.shape
        max_shift = int(M * self.tt_max_shift)
        if max_shift == 0:
            return x
        out_list = []
        for b in range(B):
            xb = x[b]
            if torch.rand(1).item() > self.tt_prob:
                out_list.append(xb); continue
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            if shift > 0:
                xb = torch.cat([xb[:shift], xb[:-shift]], dim=0)
            elif shift < 0:
                s = abs(shift)
                xb = torch.cat([xb[s:], xb[-s:]], dim=0)
            out_list.append(xb)
        return torch.stack(out_list, dim=0)

    def _temporal_cutout(self, x):
        B, M, C = x.shape
        max_hole = max(1, int(M * self.tc_max_ratio))
        cap = M // 4
        out = x.clone()
        for b in range(B):
            if torch.rand(1).item() > self.tc_prob:
                continue
            num_holes = int(torch.randint(self.tc_num_holes[0],
                                          self.tc_num_holes[1] + 1,
                                          (1,)).item())
            for _ in range(num_holes):
                hole = int(torch.randint(1, max_hole + 1, (1,)).item())
                hole = min(hole, cap)
                if hole == 0:
                    continue
                start = int(torch.randint(0, M - hole + 1, (1,)).item())
                end = start + hole
                if start > 0 and end < M:
                    alpha = torch.linspace(0, 1, hole, device=x.device).unsqueeze(1)
                    before = out[b, start - 1: start]
                    after = out[b, end: end + 1]
                    out[b, start:end] = before * (1 - alpha) + after * alpha
                elif start == 0:
                    out[b, start:end] = out[b, end: end + hole] if end + hole <= M else out[b, end - 1: end].expand(hole, C)
                else:
                    out[b, start:end] = out[b, start - hole: start]
        return out

    def _temporal_shuffle(self, x):
        B, M, _ = x.shape
        out = x.clone()
        if M < self.ts_window:
            return out
        for b in range(B):
            if torch.rand(1).item() > self.ts_prob:
                continue
            for _ in range(self.ts_num):
                start = int(torch.randint(0, M - self.ts_window + 1, (1,)).item())
                end = start + self.ts_window
                perm = torch.randperm(self.ts_window, device=x.device)
                out[b, start:end] = out[b, start:end][perm]
        return out
