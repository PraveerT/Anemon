

class NvidiaDQHybridLoader(NvidiaDTWLoader):
    """N2 DTW loader + 20 bone points appended with dual-quaternion info per frame.
    Output shape: (T, P_depth + 20, 8)
    Bone channels: [mid_x, mid_y, mid_z, time_coord, qw, qx, qy, qz]
    Depth points already have 8 channels (xyz, time + 4 padding/extras).
    """
    LANDMARKS_PATH = '../dataset/Nvidia/Processed/skeleton_landmarks.npz'
    BONES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import numpy as _np
        import re as _re
        import torch as _torch

        self._landmarks_raw = dict(_np.load(self.LANDMARKS_PATH, allow_pickle=False))
        # Build sample-line -> landmark-key map
        self._sample_to_key = {}
        for line in self.inputs_list:
            parts = line.strip().split('\t')
            relpath = parts[1] if len(parts) > 1 else line
            m = _re.search(r'class_(\d+)/(subject\S+?)/', relpath)
            if m:
                k = './Video_data/class_{}/{}'.format(m.group(1), m.group(2))
                self._sample_to_key[line] = k

        # Precompute bone features per sample (all of them)
        def _fillnan(arr):
            valid = _np.isfinite(arr[..., 0]).all(axis=-1)
            last = None; out = arr.copy()
            for t in range(out.shape[0]):
                if valid[t]: last = out[t]
                elif last is not None: out[t] = last
            for t in range(out.shape[0]):
                if not _np.isfinite(out[t]).all():
                    for t2 in range(t+1, out.shape[0]):
                        if valid[t2]: out[t] = out[t2]; break
                    else: out[t] = 0
            return out

        def _vec_to_quat(V):
            n = _np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9
            u = V / n
            cos_h = _np.clip((1 + u[..., 2:3]) * 0.5, 1e-9, 1.0)
            w = _np.sqrt(cos_h)
            sin_h = _np.sqrt(_np.clip(1 - cos_h, 0, 1))
            axis = _np.zeros_like(u); axis[..., 0] = -u[..., 1]; axis[..., 1] = u[..., 0]
            s = _np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9
            return _np.concatenate([w, axis/s*sin_h], axis=-1).astype(_np.float32)

        # Encode all landmarks once
        self._bone_feats = {}
        all_mids = []
        for k, lm_raw in self._landmarks_raw.items():
            if lm_raw.shape[0] < 4: continue
            lm = _fillnan(lm_raw)
            T = lm.shape[0]
            f = _np.zeros((T, 20, 7), dtype=_np.float32)
            for b, (p, c) in enumerate(self.BONES):
                bone_vec = lm[:, c, :] - lm[:, p, :]
                rot_q = _vec_to_quat(bone_vec)
                mid = (lm[:, c, :] + lm[:, p, :]) / 2
                f[:, b, :3] = mid
                f[:, b, 3:] = rot_q
            self._bone_feats[k] = f
            all_mids.append(f[..., :3])

        if all_mids:
            cat = _np.concatenate([m.reshape(-1, 3) for m in all_mids])
            self._mid_mean = float(cat.mean())
            self._mid_std = float(cat.std()) + 1e-7
        else:
            self._mid_mean = 0.0; self._mid_std = 1.0
        self._import_torch = _torch
        self._import_np = _np

    def __getitem__(self, index):
        _torch = self._import_torch
        _np = self._import_np
        pts, lbl, name = super().__getitem__(index)
        # pts: (T, P_depth, 8)  (torch.Tensor or np.ndarray)
        if not _torch.is_tensor(pts):
            pts = _torch.from_numpy(pts).float()
        else:
            pts = pts.float()
        T_d, P_d, C_d = pts.shape

        key = self._sample_to_key.get(name, None)
        if key is None or key not in self._bone_feats:
            bone_aug = _np.zeros((T_d, 20, 7), dtype=_np.float32)
        else:
            bf = self._bone_feats[key]
            # Resample temporally
            T_b = bf.shape[0]
            idx = _np.linspace(0, T_b - 1, T_d).astype(_np.int64)
            bone_aug = bf[idx]

        # Build bone tensor (T, 20, 8)
        bone_t = _torch.zeros((T_d, 20, 8), dtype=_torch.float32)
        mid_norm = (bone_aug[..., :3] - self._mid_mean) / self._mid_std
        bone_t[..., :3] = _torch.from_numpy(mid_norm).float()
        time_coord = _torch.linspace(-1, 1, T_d)
        bone_t[..., 3] = time_coord.unsqueeze(-1).expand(T_d, 20)
        bone_t[..., 4:8] = _torch.from_numpy(bone_aug[..., 3:]).float()

        combined = _torch.cat([pts, bone_t], dim=1)  # (T, P_d + 20, 8)
        return combined, lbl, name
