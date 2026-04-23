"""Add MotionRigiditySegmentation: PMamba + per-point rigid/articulating
binary segmentation aux head.

Target per point = 1 if Kabsch residual > per-frame median else 0.
Cross-entropy on per-point features (fea1 output). Uses
NvidiaQuaternionQCCParityLoader + correspondence-guided sampling for clean
residuals.

Simpler than contrastive (no pair sampling, direct supervised gradient).
Still shapes features to distinguish rigid (palm/wrist) from articulating
(fingers).
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
if "class MotionRigiditySegmentation" in src:
    start = src.find("\n\nclass MotionRigiditySegmentation")
    src = src[:start]

snippet = '''

class MotionRigiditySegmentation(Motion):
    """PMamba + per-point rigid/articulating binary segmentation aux.

    Aux target: 1 if Kabsch residual > per-frame median, else 0. Tiny linear
    head on fea1 per-point features. Binary CE aux averaged over
    correspondence-matched points only.
    """

    def __init__(self, num_classes, pts_size, seg_weight=0.1, **kwargs):
        super().__init__(num_classes, pts_size, **kwargs)
        self.seg_weight = seg_weight
        feat_dim = 64                                            # stage1 out
        self.seg_head = nn.Linear(feat_dim, 2)
        self.latest_aux_loss = None
        self.latest_aux_metrics = {}

    def get_auxiliary_loss(self):
        return self.latest_aux_loss

    def get_auxiliary_metrics(self):
        return self.latest_aux_metrics

    def _correspondence_guided_sample(self, points, aux_input):
        batch_size, num_frames, pts_per_frame, channels = points.shape
        sample_size = min(self.pts_size, pts_per_frame)
        device = points.device
        if sample_size == pts_per_frame:
            corr_matched = torch.ones(batch_size, num_frames - 1, pts_per_frame,
                                      dtype=torch.bool, device=device)
            return points, corr_matched

        orig_flat_idx = aux_input["orig_flat_idx"]
        corr_target = aux_input["corr_full_target_idx"]
        corr_weight = aux_input["corr_full_weight"]
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
        if isinstance(inputs, dict):
            points_raw = inputs["points"]
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
            coords = sampled.permute(0, 3, 1, 2).contiguous()
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
        )                                                        # (B, 64, T, P)

        self.latest_aux_loss = None
        self.latest_aux_metrics = {}
        if self.training and self.seg_weight > 0 and has_corr:
            with torch.no_grad():
                rigidity = self._compute_residual_corr(coords[:, :3], corr_matched)
            s_loss, s_metrics = self._seg_loss(fea1_raw, rigidity, corr_matched)
            self.latest_aux_loss = self.seg_weight * s_loss
            s_metrics["qcc_raw"] = s_loss.detach()
            s_metrics["qcc_forward"] = s_loss.detach()
            s_metrics["qcc_backward"] = s_loss.detach()
            s_metrics["qcc_valid_ratio"] = corr_matched.float().mean().detach()
            self.latest_aux_metrics = s_metrics

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
        B, _, T, P = xyz.shape
        device = xyz.device
        xyz_p = xyz.permute(0, 2, 3, 1)
        rig = torch.zeros(B, T, P, device=device, dtype=xyz.dtype)
        eye = torch.eye(3, device=device, dtype=xyz.dtype).unsqueeze(0).expand(B, 3, 3)
        for t in range(T - 1):
            p = xyz_p[:, t]; q = xyz_p[:, t + 1]
            w = corr_matched[:, t].float()
            w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
            c_p = (p * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
            c_q = (q * w.unsqueeze(-1)).sum(1, keepdim=True) / w_sum
            u = p - c_p; v = q - c_q
            H = (u * w.unsqueeze(-1)).transpose(-1, -2) @ v
            H = H + 1e-5 * eye
            try:
                U, S, Vh = torch.linalg.svd(H)
            except Exception:
                rig[:, t] = (v - u).norm(dim=-1); continue
            V = Vh.transpose(-1, -2)
            det = torch.det(V @ U.transpose(-1, -2))
            D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1))
            R = V @ D @ U.transpose(-1, -2)
            rig[:, t] = (v - u @ R.transpose(-1, -2)).norm(dim=-1)
        rig[:, -1] = rig[:, -2]
        return rig

    def _seg_loss(self, features, rigidity, corr_matched):
        """Binary CE per-point: 1 if residual > per-frame median else 0.
        Mask to corr-matched points only."""
        B, C, T, P = features.shape
        feat = features.permute(0, 2, 3, 1).reshape(-1, C)       # (B*T*P, C)
        logits = self.seg_head(feat)                              # (B*T*P, 2)

        med = rigidity.median(dim=-1, keepdim=True).values
        target = (rigidity > med).long().reshape(-1)              # (B*T*P,)

        # Mask to corr-matched (valid) points. corr_matched is (B, T-1, P).
        # Expand last frame from previous.
        if corr_matched.shape[1] < T:
            last = corr_matched[:, -1:]
            mask_full = torch.cat([corr_matched, last], dim=1)    # (B, T, P)
        else:
            mask_full = corr_matched
        mask = mask_full.reshape(-1)                              # bool

        if mask.sum() < 1:
            return features.new_zeros((), requires_grad=True), {}

        loss = F.cross_entropy(logits[mask], target[mask])
        with torch.no_grad():
            acc = (logits[mask].argmax(-1) == target[mask]).float().mean()
        return loss, {"seg_raw": loss.detach(), "seg_acc": acc}
'''
SRC_NEW = src.rstrip() + snippet + "\n"
MOTION.write_text(SRC_NEW, encoding="utf-8")
print("added MotionRigiditySegmentation to models/motion.py")
