"""Add MotionRigidityContrastive: PMamba + InfoNCE contrastive aux on
per-point rigidity residuals computed ON-THE-FLY via nearest-neighbor Kabsch.

Design:
- Uses NvidiaLoader (no rigidity precompute needed, no correspondence data)
- Per frame pair (t, t+1): match points via NN (cdist argmin on centered xyz),
  Kabsch on matched pairs, compute per-point residual magnitude
- Split per-frame points by median residual into rigid vs articulating
- InfoNCE: pull same-set features together, push opposite apart, on fea1

NN-matching is noisy on articulating points but for binary rigid/artic split
(median threshold) the noise averages out. No preprocess or special loader.
"""
from pathlib import Path

MOTION = Path("models/motion.py")
src = MOTION.read_text(encoding="utf-8")
# Overwrite any existing class by rebuilding from scratch.
if "class MotionRigidityContrastive" in src:
    # crude strip
    start = src.find("\n\nclass MotionRigidityContrastive")
    if start > 0:
        src = src[:start]
snippet = '''

class MotionRigidityContrastive(Motion):
    """PMamba + InfoNCE contrastive aux on per-point rigidity residuals.

    Residuals computed on-the-fly via NN-matching Kabsch between consecutive
    frames. No preprocess, no correspondence data loader needed.
    """

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

    def extract_features(self, inputs):
        coords = self._sample_points(inputs)
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
        if self.training and self.contrast_weight > 0:
            with torch.no_grad():
                rigidity = self._compute_residual(coords[:, :3])  # (B, T, P)
            c_loss, c_metrics = self._contrastive_loss(fea1_raw, rigidity)
            self.latest_aux_loss = self.contrast_weight * c_loss
            c_metrics["qcc_raw"] = c_loss.detach()
            c_metrics["qcc_forward"] = c_loss.detach()
            c_metrics["qcc_backward"] = c_loss.detach()
            c_metrics["qcc_valid_ratio"] = torch.tensor(1.0, device=c_loss.device)
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

    def _compute_residual(self, xyz):
        """xyz: (B, 3, T, P). Returns per-frame-pair residual magnitude (B, T, P).

        Per frame pair (t, t+1): NN-match points via cdist argmin on centered
        xyz, Kabsch rotation, residual = |v_matched - R*u|. Last frame: copy
        previous (no pair).
        """
        B, _, T, P = xyz.shape
        device = xyz.device
        xyz_p = xyz.permute(0, 2, 3, 1)                          # (B, T, P, 3)

        rig = torch.zeros(B, T, P, device=device, dtype=xyz.dtype)
        eye = torch.eye(3, device=device, dtype=xyz.dtype).unsqueeze(0).expand(B, 3, 3)

        for t in range(T - 1):
            p = xyz_p[:, t]                                       # (B, P, 3)
            q = xyz_p[:, t + 1]
            c_p = p.mean(dim=1, keepdim=True)
            c_q = q.mean(dim=1, keepdim=True)
            u = p - c_p                                           # (B, P, 3)
            v = q - c_q

            # NN match: for each u_i, find nearest v_j in CENTERED space.
            dist = torch.cdist(u, v)                              # (B, P, P)
            nn = dist.argmin(dim=-1)                              # (B, P)
            v_m = torch.gather(v, 1, nn.unsqueeze(-1).expand(-1, -1, 3))

            H = u.transpose(-1, -2) @ v_m                         # (B, 3, 3)
            H = H + 1e-5 * eye
            try:
                U, S, Vh = torch.linalg.svd(H)
            except Exception:
                R = eye
                residual = v_m - u
                rig[:, t] = residual.norm(dim=-1)
                continue

            V = Vh.transpose(-1, -2)
            det = torch.det(V @ U.transpose(-1, -2))
            D_diag = torch.stack(
                [torch.ones_like(det), torch.ones_like(det), det], dim=-1,
            )
            D = torch.diag_embed(D_diag)
            R = V @ D @ U.transpose(-1, -2)

            u_rot = u @ R.transpose(-1, -2)                       # (B, P, 3)
            residual = v_m - u_rot
            rig[:, t] = residual.norm(dim=-1)

        rig[:, -1] = rig[:, -2]                                   # copy last
        return rig

    def _contrastive_loss(self, features, rigidity):
        """features: (B, C, T, P), rigidity: (B, T, P) scalar per-point."""
        B, C, T, P = features.shape
        feat = features.permute(0, 2, 3, 1)                       # (B, T, P, C)
        feat = F.normalize(feat, dim=-1)

        med = rigidity.median(dim=-1, keepdim=True).values
        is_rigid = rigidity <= med

        tau = self.contrast_temp
        total = feat.new_zeros(())
        count = 0
        pos_frac_sum = 0.0

        for b in range(B):
            for t in range(T):
                f = feat[b, t]
                r_mask = is_rigid[b, t]
                pos_idx = r_mask.nonzero(as_tuple=True)[0]
                neg_idx = (~r_mask).nonzero(as_tuple=True)[0]
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
                pos_frac_sum += pos_idx.numel() / P

        if count == 0:
            return features.new_zeros((), requires_grad=True), {}
        loss = total / count
        metrics = {
            "contrast_raw": loss.detach(),
            "pos_frac": torch.tensor(pos_frac_sum / max(B * T, 1), device=features.device),
        }
        return loss, metrics
'''
SRC_NEW = src.rstrip() + snippet + "\n"
MOTION.write_text(SRC_NEW, encoding="utf-8")
print("rewrote MotionRigidityContrastive (on-the-fly residual)")
