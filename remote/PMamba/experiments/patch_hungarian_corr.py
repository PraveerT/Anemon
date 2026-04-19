"""Add Hungarian (optimal 1-to-1) correspondence to nvidia_dataloader.

The default NvidiaQuaternionQCCParityLoader uses mutual-NN filtering, which drops
40-47% of valid matches across all motion regimes (see analysis in chat history on
2026-04-19).  This patch adds:

  - assignment_mode: "mutual" (default, unchanged) | "hungarian"
  - vectorized _match_dense_points_vec (the original is pure-Python triple loop)
  - _build_full_correspondence_hungarian using scipy.optimize.linear_sum_assignment
  - separate cache path suffix for Hungarian caches: _hu_r{radius}.npz
    (existing mutual cache keeps _bi_r{radius}.npz tag; no collision)

Empirical improvement on 24-clip stratified sample:
  low motion  mutual 53.1% kept -> hungarian 98.3%  (1.85x)
  mid motion  mutual 41.1% kept -> hungarian 72.1%  (1.75x)
  high motion mutual 31.3% kept -> hungarian 59.4%  (1.90x)

Usage from training config / loader args:
  assignment_mode: hungarian

Regenerate caches once after applying this patch:
  for pts in .../*_pts.npy:
      rp = np.load(pts); rd = loader._load_aligned_depth(...)
      corr = loader._build_full_correspondence_hungarian(rd, rp)
      np.savez_compressed(cache_path, **corr)
"""
import re
from pathlib import Path

PATH = Path('nvidia_dataloader.py')
src = PATH.read_text(encoding='utf-8')

# 1. Add assignment_mode to __init__ signature + validation
src = src.replace(
    "        bidirectional_correspondence=True,\n    ):\n        self.bidirectional_correspondence = bidirectional_correspondence\n",
    "        bidirectional_correspondence=True,\n        assignment_mode=\"mutual\",\n    ):\n"
    "        self.bidirectional_correspondence = bidirectional_correspondence\n"
    "        if assignment_mode not in (\"mutual\", \"hungarian\"):\n"
    "            raise ValueError(f\"assignment_mode must be 'mutual' or 'hungarian', got {assignment_mode}\")\n"
    "        self.assignment_mode = assignment_mode\n",
)

# 2. Route builder based on mode
src = src.replace(
    "        correspondence = self._build_full_correspondence(raw_depth, raw_points)\n        if self.correspondence_cache:\n",
    "        if self.assignment_mode == \"hungarian\":\n"
    "            correspondence = self._build_full_correspondence_hungarian(raw_depth, raw_points)\n"
    "        else:\n"
    "            correspondence = self._build_full_correspondence(raw_depth, raw_points)\n"
    "        if self.correspondence_cache:\n",
)

# 3. Add 'hu' suffix in cache path when Hungarian
src = src.replace(
    "    def _full_correspondence_cache_path(self, raw_depth_path):\n"
    "        direction_tag = \"bi\" if self.bidirectional_correspondence else \"uni\"\n"
    "        return \"{}_{}_{}_r{}.npz\".format(\n"
    "            raw_depth_path[:-4],\n"
    "            self.correspondence_cache_tag,\n"
    "            direction_tag,\n"
    "            self.correspondence_radius,\n"
    "        )\n",
    "    def _full_correspondence_cache_path(self, raw_depth_path):\n"
    "        direction_tag = \"bi\" if self.bidirectional_correspondence else \"uni\"\n"
    "        if self.assignment_mode == \"hungarian\":\n"
    "            return \"{}_{}_hu_r{}.npz\".format(\n"
    "                raw_depth_path[:-4],\n"
    "                self.correspondence_cache_tag,\n"
    "                self.correspondence_radius,\n"
    "            )\n"
    "        return \"{}_{}_{}_r{}.npz\".format(\n"
    "            raw_depth_path[:-4],\n"
    "            self.correspondence_cache_tag,\n"
    "            direction_tag,\n"
    "            self.correspondence_radius,\n"
    "        )\n",
)

# 4. Inject vectorized matcher + Hungarian builder above _frame_best_matches
insertion = '''    def _match_dense_points_vec(self, depth_frame, source_points):
        height, width = depth_frame.shape
        n = source_points.shape[0]
        r = self.correspondence_radius
        sr = np.clip(np.round(source_points[:, 0]).astype(np.int32), 0, height - 1)
        sc = np.clip(np.round(source_points[:, 1]).astype(np.int32), 0, width - 1)
        sd = source_points[:, 2].astype(np.float32)
        drs, dcs = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        drs = drs.ravel().astype(np.int32)
        dcs = dcs.ravel().astype(np.int32)
        rr = sr[:, None] + drs[None, :]
        cc = sc[:, None] + dcs[None, :]
        in_bounds = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        rr_c = np.clip(rr, 0, height - 1)
        cc_c = np.clip(cc, 0, width - 1)
        cand_depth = depth_frame[rr_c, cc_c].astype(np.float32)
        valid_pix = in_bounds & (cand_depth > 0)
        spatial = (drs * drs + dcs * dcs).astype(np.float32)
        score = spatial[None, :] + self.correspondence_depth_weight * np.abs(cand_depth - sd[:, None])
        score = np.where(valid_pix, score, np.inf)
        best = np.argmin(score, axis=1)
        best_score = score[np.arange(n), best]
        ok = np.isfinite(best_score)
        matched = np.stack(
            [rr[np.arange(n), best], cc[np.arange(n), best], cand_depth[np.arange(n), best]],
            axis=1,
        ).astype(np.float32)
        return matched, ok

    def _build_full_correspondence_hungarian(self, raw_depth, raw_points):
        from scipy.optimize import linear_sum_assignment
        frame_count, point_count, _ = raw_points.shape
        total_points = frame_count * point_count
        corr_full_target_idx = np.full(total_points, -1, dtype=np.int64)
        corr_full_weight = np.zeros(total_points, dtype=np.float32)
        big_cost = np.float32(1e8)

        for frame_idx in range(frame_count - 1):
            source_points = raw_points[frame_idx, :, :3]
            target_points = raw_points[frame_idx + 1, :, :3]
            matched, ok = self._match_dense_points_vec(raw_depth[frame_idx + 1], source_points)
            diff = target_points[None, :, :3] - matched[:, None, :3]
            cost = (
                diff[..., 0] ** 2
                + diff[..., 1] ** 2
                + self.correspondence_sample_weight * diff[..., 2] ** 2
            ).astype(np.float32)
            cost_masked = np.where(ok[:, None], cost, big_cost)
            row, col = linear_sum_assignment(cost_masked)
            assigned_cost = cost_masked[row, col]
            keep = (assigned_cost <= self.correspondence_max_dist) & ok[row]
            kept_row = row[keep]
            kept_col = col[keep]
            kept_cost = assigned_cost[keep].astype(np.float32)
            conf = 1.0 / (1.0 + kept_cost / max(self.correspondence_confidence_scale, 1e-6))
            conf = conf.astype(np.float32)
            source_flat = frame_idx * point_count + kept_row
            target_flat = (frame_idx + 1) * point_count + kept_col
            corr_full_target_idx[source_flat] = target_flat
            corr_full_weight[source_flat] = conf

        return {
            "corr_full_target_idx": corr_full_target_idx,
            "corr_full_weight": corr_full_weight,
        }

    def _frame_best_matches'''

src = src.replace("    def _frame_best_matches", insertion, 1)

PATH.write_text(src, encoding='utf-8')
print("patched nvidia_dataloader.py (Hungarian correspondence support)")
