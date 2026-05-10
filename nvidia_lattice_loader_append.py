

class NvidiaDTWLatticeLoader(NvidiaDTWLoader):
    """N2 DTW loader + 216 lattice points appended with quaternion arrow features.

    Output shape: (T, P_depth + 216, 8) where channels are
      [x, y, z, time, intensity, q_x, q_y, q_z]
    For depth points: q_x,y,z = 0
    For lattice points: q_x,y,z from precomputed lattice arrow field
                        intensity = q_w (real part of quaternion)

    Lattice coords are in [-1, 1] — same normalized space as depth points.
    """
    LATTICE_PATH = '../dataset/Nvidia/Processed/lattice_arrows_K6_v2.npz'
    LATTICE_K = 6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import numpy as _np
        import re as _re
        import torch as _torch

        self._lattice_field = dict(_np.load(self.LATTICE_PATH, allow_pickle=False))
        # Build lattice point coordinates (216 fixed positions in [-1, 1]^3)
        coords = _np.linspace(-1, 1, self.LATTICE_K)
        grid = _np.stack(_np.meshgrid(coords, coords, coords, indexing='ij'), axis=-1)
        self._lattice_xyz = grid.reshape(-1, 3).astype(_np.float32)  # (216, 3)
        # Map sample line -> lattice key
        self._sample_to_key = {}
        for line in self.inputs_list:
            parts = line.strip().split('\t')
            relpath = parts[1] if len(parts) > 1 else line
            m = _re.search(r'class_(\d+)/(subject\S+?)/', relpath)
            if m:
                k = './Video_data/class_{}/{}'.format(m.group(1), m.group(2))
                self._sample_to_key[line] = k
        self._import_torch = _torch
        self._import_np = _np

    def __getitem__(self, index):
        _torch = self._import_torch
        _np = self._import_np
        pts, lbl, name = super().__getitem__(index)
        if not _torch.is_tensor(pts):
            pts = _torch.from_numpy(pts).float()
        else:
            pts = pts.float()
        T_d, P_d, C_d = pts.shape  # depth: (T, P, 8)

        # Get lattice quaternion field for this sample
        key = self._sample_to_key.get(name, None)
        if key is None or key not in self._lattice_field:
            lat_q = _np.zeros((T_d, 216, 4), dtype=_np.float32)
            lat_q[..., 0] = 1.0  # identity quaternion
        else:
            field = self._lattice_field[key]  # (T, 216, 4)
            T_lat = field.shape[0]
            if T_lat != T_d:
                idx = _np.linspace(0, T_lat - 1, T_d).astype(_np.int64)
                field = field[idx]
            lat_q = field

        # Build lattice point tensor (T, 216, 8)
        # channels: [x, y, z, time, q_w, q_x, q_y, q_z]
        lat_xyz = self._lattice_xyz  # (216, 3) — broadcast to (T, 216, 3)
        lat_xyz_T = _np.broadcast_to(lat_xyz[None, :, :], (T_d, 216, 3)).copy()
        time_coord = _np.linspace(-1, 1, T_d).astype(_np.float32)[:, None]  # (T, 1)
        time_T = _np.broadcast_to(time_coord[:, None, :], (T_d, 216, 1)).copy()
        # Lattice 8 channels: xyz(3) + time(1) + quat(4) = 8
        lat_8 = _np.concatenate([lat_xyz_T, time_T, lat_q], axis=-1).astype(_np.float32)
        lat_8 = _torch.from_numpy(lat_8)

        # Pad depth points' last 4 channels (quat slots) with zeros + identity
        # Original depth channels (8): we'll set last 4 to (1, 0, 0, 0) = identity quat
        depth_8 = pts.clone()
        if depth_8.shape[-1] >= 8:
            # Replace channels 4-7 with identity quat
            depth_8[..., 4] = 1.0  # q_w
            depth_8[..., 5:8] = 0.0  # q_x, q_y, q_z
        else:
            # Pad to 8 channels
            pad = _torch.zeros((T_d, P_d, 8 - depth_8.shape[-1]), dtype=_torch.float32)
            pad[..., 0] = 1.0
            depth_8 = _torch.cat([depth_8, pad], dim=-1)

        combined = _torch.cat([depth_8, lat_8], dim=1)  # (T, P_d + 216, 8)
        return combined, lbl, name
