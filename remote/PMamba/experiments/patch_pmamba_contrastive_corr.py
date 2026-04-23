"""Add MotionRigidityContrastiveCorr: PMamba + contrastive aux with proper
Hungarian correspondence sampling (no NN matching).

Uses NvidiaQuaternionQCCParityLoader (return_correspondence=True). Points
sampled via correspondence chains → same array index = same physical point
across frames. Per-point Kabsch residual is clean, not noisy.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionRigidityContrastiveCorr" in src:
    start = src.find("\n\nclass MotionRigidityContrastiveCorr")
    src = src[:start]

snippet = '''

class MotionRigidityContrastiveCorr(Motion):
    """PMamba + InfoNCE contrastive aux with clean correspondence-aligned
    per-point residuals."""

    def __init__(self, num_classes, pts_size, contrast_weight=0.05,
                 contrast_temp=0.1, contrast_num_anchors=16, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.contrast_weight = contrast_weight
        self.contrast_temp = contrast_temp
        self.contrast_num_anchors = contrast_num_anchors
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _correspondence_guided_sample(self, points, aux_input):
        """Copied from BearingQCCFeatureMotion. Returns (B, F, S, C) with
        same-index = same-physical-point across frames."""
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device

        if sample_size == pts_per_frame:
            corr_matched = torch.ones(batch_size, num_frames - 1, pts_per_frame,
                                      dtype=torch.bool, device=device)
            return points, corr_matched

        orig_flat_idx = aux_input['orig_flat_idx']
        corr_target = aux_input['corr_full_target_idx']
        corr_weight = aux_input['corr_full_weight']
        total_pts = corr_target.shape[-1]
        raw_ppf = total_pts // num_frames

        sampled = torch.zeros(batch_size, num_frames, sample_size, channels,
                              device=device, dtype=points.dtype)
        corr_matched = torch.zeros(batch_size, num_frames - 1, sample_size,
                                   dtype=torch.bool, device=device)

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
                corr_matched[b, t] = valid
                current_prov = orig_flat_idx[b, t + 1, next_idx].long()

        return sampled, corr_matched

    def extract_features(self, inputs):
        # inputs from NvidiaQuaternionQCCParityLoader is a dict.
        if isinstance(inputs, dict):
            points_raw = inputs["points"]                        # (B, T, P_raw, C)
            aux_input = inputs
            has_corr = ("orig_flat_idx" in aux_input
                        and "corr_full_target_idx" in aux_input
                        and "corr_full_weight" in aux_input)
        else:
            points_raw = inputs
            aux_input = None
            has_corr = False

        if has_corr:
            sampled, corr_matched = self._correspondence_guided_sample(
                points_raw[..., :4], aux_input,
            )
            coords = sampled.permute(0, 3, 1, 2).contiguous()   # (B, 4, T, P)
        else:
            coords = self._sample_points(points_raw)
            corr_matched = None

        batchsize, in_dims, timestep, pts_num = coords.shape

        ret_array1 = self.group.group_points(
            distance_dim=[0, 1, 2], array1=coords, array2=coords,
            knn=self.knn[0], dim=3,
        )
        ret_array1 = ret_array1.reshape(batchsize, in_dims, timestep * pts_num, -1)
        fea1_raw = self.pool1(self.stage1(ret_array1)).reshape(
            batchsize, -1, timestep, pts_num,
        )

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.contrast_weight > 0 and has_corr:
            with torch.no_grad():
                rigidity = self._compute_residual_corr(
                    coords[:, :3], corr_matched,
                )
            c_loss, c_metrics = self._contrastive_loss(
                fea1_raw, rigidity, corr_matched,
            )
            self.latest_aux_loss = self.contrast_weight * c_loss
            c_metrics["qcc_raw"] = c_loss.detach()
            c_metrics["qcc_forward"] = c_loss.detach()
            c_metrics["qcc_backward"] = c_loss.detach()
            c_metrics["qcc_valid_ratio"] = corr_matched.float().mean().detach()
            self.latest_aux_metrics = c_metrics

        fea1 = torch.cat((coords, fea1_raw), dim=1)

        in_dims = fea1.shape[1] * 2 - 4
        pts_num //= self.downsample[0]
        rg2 = self.group.st_group_points(fea1, 3, [0, 1, 2], self.knn[1], 3)
        ret2, coords = self.select_ind(rg2, coords, batchsize, in_dims, timestep, pts_num)
        fea2 = self.pool2(self.stage2(ret2)).reshape(batchsize, -1, timestep, pts_num)
        fea2 = torch.cat((coords, fea2), dim=1)
        fea2 = self.multi_scale(fea2)

        in_dims = fea2.shape[1] * 2 - 4
        pts_num //= self.downsample[1]
        rg3 = self.group.st_group_points(fea2, 3, [0, 1, 2], self.knn[2], 3)
        ret3, coords = self.select_ind(rg3, coords, batchsize, in_dims, timestep, pts_num)
        fea3 = self.pool3(self.stage3(ret3)).reshape(batchsize, -1, timestep, pts_num)
        fea3_mamba = self.mamba(fea3)
        coords_fea3 = torch.cat((coords, fea3_mamba), dim=1)

        output = self.stage5(coords_fea3)
        output = self.pool5(output)
        output = self.global_bn(output)
        return output.flatten(1)

    def _compute_residual_corr(self, xyz, corr_matched):
        """xyz: (B, 3, T, P). corr_matched: (B, T-1, P) bool.

        Per frame pair (t, t+1): u/v are correspondence-aligned (same index =
        same physical point). Kabsch using ONLY the corr_matched points for
        robustness, then residual magnitude for ALL points.
        """
        B, _, T, P = xyz.shape
        device = xyz.device
        xyz_p = xyz.permute(0, 2, 3, 1)                          # (B, T, P, 3)
        rig = torch.zeros(B, T, P, device=device, dtype=xyz.dtype)
        eye = torch.eye(3, device=device, dtype=xyz.dtype).unsqueeze(0).expand(B, 3, 3)

        for t in range(T - 1):
            p = xyz_p[:, t]
            q = xyz_p[:, t + 1]
            w = corr_matched[:, t].float()                       # (B, P) mask
            w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)

            # Centroids on matched subset only.
            c_p = (p * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum
            c_q = (q * w.unsqueeze(-1)).sum(dim=1, keepdim=True) / w_sum
            u = p - c_p                                           # (B, P, 3)
            v = q - c_q

            # Weighted Kabsch using matched mask.
            u_w = u * w.unsqueeze(-1)
            H = u_w.transpose(-1, -2) @ v                         # (B, 3, 3)
            H = H + 1e-5 * eye
            try:
                U, S, Vh = torch.linalg.svd(H)
            except Exception:
                R = eye
                rig[:, t] = (v - u).norm(dim=-1)
                continue

            V = Vh.transpose(-1, -2)
            det = torch.det(V @ U.transpose(-1, -2))
            D_diag = torch.stack(
                [torch.ones_like(det), torch.ones_like(det), det], dim=-1,
            )
            D = torch.diag_embed(D_diag)
            R = V @ D @ U.transpose(-1, -2)

            u_rot = u @ R.transpose(-1, -2)
            residual = v - u_rot
            rig[:, t] = residual.norm(dim=-1)

        rig[:, -1] = rig[:, -2]
        return rig

    def _contrastive_loss(self, features, rigidity, corr_matched=None):
        """features: (B, C, T, P), rigidity: (B, T, P). Split by per-frame
        median. Only use points that are corr_matched (where reliable)."""
        B, C, T, P = features.shape
        feat = features.permute(0, 2, 3, 1)
        feat = F.normalize(feat, dim=-1)

        med = rigidity.median(dim=-1, keepdim=True).values
        is_rigid = rigidity <= med

        tau = self.contrast_temp
        total = feat.new_zeros(())
        count = 0

        for b in range(B):
            for t in range(T):
                f = feat[b, t]
                r_mask = is_rigid[b, t]
                # Filter to corr-matched points (reliable residuals only).
                if corr_matched is not None and t < T - 1:
                    reliable = corr_matched[b, t]
                elif corr_matched is not None and t == T - 1:
                    reliable = corr_matched[b, t - 1]
                else:
                    reliable = torch.ones_like(r_mask)

                pos_idx = (r_mask & reliable).nonzero(as_tuple=True)[0]
                neg_idx = (~r_mask & reliable).nonzero(as_tuple=True)[0]
                if pos_idx.numel() < 2 or neg_idx.numel() < 1:
                    continue

                for anchor_set, other_set in [(pos_idx, neg_idx), (neg_idx, pos_idx)]:
                    N = min(self.contrast_num_anchors, anchor_set.numel() - 1)
                    if N < 1:
                        continue
                    sel = torch.randperm(anchor_set.numel(), device=f.device)[:N]
                    a_idx = anchor_set[sel]
                    anchors = f[a_idx]
                    pos_f = f[anchor_set]
                    neg_f = f[other_set]

                    sim_pos = anchors @ pos_f.T
                    sim_neg = anchors @ neg_f.T
                    self_mask = (anchor_set.unsqueeze(0) == a_idx.unsqueeze(1))
                    sim_pos = sim_pos.masked_fill(self_mask, -1e9)

                    exp_pos = (sim_pos / tau).exp()
                    exp_neg = (sim_neg / tau).exp()
                    num = exp_pos.sum(dim=-1) + 1e-8
                    den = num + exp_neg.sum(dim=-1)
                    loss = -(num / den).log().mean()
                    total = total + loss
                    count += 1

        if count == 0:
            return features.new_zeros((), requires_grad=True), {}
        loss = total / count
        return loss, {"contrast_raw": loss.detach()}
'''
SRC_NEW = src.rstrip() + snippet + "\n"
MOTION.write_text(SRC_NEW, encoding="utf-8")
print("added MotionRigidityContrastiveCorr to models/motion.py")
