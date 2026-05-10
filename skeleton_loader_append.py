

class NvidiaSkeletonLoader(NvidiaLoader):
    """Returns MediaPipe hand landmarks as point cloud: (T=framesize, 21, 4) with
    [x, y, z, t]. Linear interpolation over NaN frames. Resampled in time to framesize.
    Output mimics standard point cloud loader so existing Motion model works.
    """
    LANDMARKS_PATH = '../dataset/Nvidia/Processed/skeleton_landmarks.npz'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import numpy as _np
        self._landmarks = dict(_np.load(self.LANDMARKS_PATH, allow_pickle=False))
        # Build mapping from list-line to landmark key
        self._sample_to_key = {}
        for line in self.inputs_list:
            # line: "0000\t./Nvidia/Processed/train/class_01/subject3_r0/sk_depth.avi/0000_depth_label_00.npy\t00\n"
            # we want './Video_data/class_01/subject3_r0'
            parts = line.strip().split('\t')
            relpath = parts[1] if len(parts) > 1 else line
            # Extract class_XX and subject_XX
            import re as _re
            m = _re.search(r'class_(\d+)/(subject\S+?)/', relpath)
            if m:
                key = f'./Video_data/class_{m.group(1)}/{m.group(2)}'
                self._sample_to_key[line] = key

    def __getitem__(self, index):
        import torch as _torch
        import numpy as _np
        # Get label normally
        label = int(self.r.split(self.inputs_list[index])[-2])
        line = self.inputs_list[index]
        key = self._sample_to_key.get(line, None)
        T_target = self.framerate
        if key is None or key not in self._landmarks:
            # No skeleton — return zeros
            pts = _np.zeros((T_target, 21, 4), dtype=_np.float32)
            return _torch.from_numpy(pts), label, line
        arr = self._landmarks[key].astype(_np.float32)  # (T, 21, 3)
        T = arr.shape[0]
        # Forward-fill NaN
        valid = _np.isfinite(arr[..., 0]).all(axis=-1)  # (T,)
        last_valid = None
        for t in range(T):
            if valid[t]:
                last_valid = arr[t]
            elif last_valid is not None:
                arr[t] = last_valid
        # If still NaN at start, back-fill
        for t in range(T):
            if not _np.isfinite(arr[t]).all():
                # Find next valid
                for t2 in range(t + 1, T):
                    if valid[t2]:
                        arr[t] = arr[t2]
                        break
                else:
                    arr[t] = 0
        # Resample temporally to T_target
        if T != T_target:
            idx = _np.linspace(0, T - 1, T_target).astype(_np.int64)
            arr = arr[idx]
        # Add t-channel
        out = _np.zeros((T_target, 21, 4), dtype=_np.float32)
        out[..., :3] = arr
        out[..., 3] = _np.linspace(-1, 1, T_target)[:, None]
        return _torch.from_numpy(out), label, line
