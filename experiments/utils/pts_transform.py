# This file is based on Pointnet2_Pytorch repo.
# https://github.com/erikwijmans/Pointnet2_PyTorch.git

from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import pdb
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pts):
        for t in self.transforms:
            pts = t(pts)
        return pts


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    axis = axis.numpy()
    angle = angle.numpy()
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    return R.float()


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = torch.FloatTensor(1).uniform_(self.lo, self.hi)
        points[:, :3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = torch.clamp(
            self.angle_sigma * torch.randn(3), -self.angle_clip, self.angle_clip
        )
        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], torch.FloatTensor([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], torch.FloatTensor([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], torch.FloatTensor([0.0, 0.0, 1.0]))
        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        points[:, :3] = torch.matmul(points[:, :3], rotation_matrix.t())
        return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        # points[:, :3] += torch.clamp(torch.randn(1) * self.std, -self.clip, self.clip)
        points[:, :3] += torch.clamp(torch.randn(points[:, :3].shape) * self.std, -self.clip, self.clip)
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = torch.FloatTensor(3).uniform_(-self.translate_range, self.translate_range)
        points[:, :3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        dropout_ratio = torch.rand(1) * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(torch.rand((points.shape[0])).numpy() <= dropout_ratio.numpy())[0]
        if len(drop_idx) > 0:
            points[drop_idx] = points[drop_idx - 1]  # set to the first point

        return points


class TemporalSpeedChange(object):
    """Randomly speed up or slow down temporal sequences."""
    def __init__(self, speed_range=(0.8, 1.2), prob=0.5):
        self.speed_range = speed_range
        self.prob = prob
        
    def __call__(self, points):
        if torch.rand(1) > self.prob:
            return points
            
        # Assume points are flattened: (T*N, C) where we need to identify frames
        # This is a simplified approach - may need adjustment based on actual data format
        total_points = points.shape[0]
        
        # Sample speed factor
        speed = torch.rand(1) * (self.speed_range[1] - self.speed_range[0]) + self.speed_range[0]
        
        if speed < 1.0:
            # Slow down: duplicate some frames
            num_duplicates = int(total_points * (1 - speed) * 0.1)  # Conservative duplication
            if num_duplicates > 0:
                duplicate_indices = torch.randperm(total_points)[:num_duplicates]
                duplicated_points = points[duplicate_indices]
                points = torch.cat([points, duplicated_points], dim=0)
                
                # Truncate to original length
                points = points[:total_points]
        else:
            # Speed up: skip some frames
            skip_ratio = min(0.2, (speed - 1.0))  # Conservative skipping
            num_skip = int(total_points * skip_ratio)
            if num_skip > 0:
                skip_indices = torch.randperm(total_points)[:num_skip]
                keep_mask = torch.ones(total_points, dtype=torch.bool)
                keep_mask[skip_indices] = False
                kept_points = points[keep_mask]
                
                # Pad back to original length by repeating some points
                while kept_points.shape[0] < total_points:
                    pad_size = min(kept_points.shape[0], total_points - kept_points.shape[0])
                    pad_points = kept_points[:pad_size]
                    kept_points = torch.cat([kept_points, pad_points], dim=0)
                
                points = kept_points[:total_points]
        
        return points


class TemporalTranslate(object):
    """Randomly shift the action in time by padding/cropping the sequence."""
    def __init__(self, max_shift_ratio=0.1, prob=0.4):
        self.max_shift_ratio = max_shift_ratio
        self.prob = prob
        
    def __call__(self, points):
        if torch.rand(1) > self.prob:
            return points
            
        total_points = points.shape[0]
        max_shift = int(total_points * self.max_shift_ratio)
        
        if max_shift == 0:
            return points
            
        # Random shift amount (positive = delay start, negative = early start)
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        
        if shift > 0:
            # Delay start: pad beginning, crop end
            # Duplicate first few points to pad the beginning
            pad_points = points[:shift]
            shifted_points = torch.cat([pad_points, points[:-shift]], dim=0)
        elif shift < 0:
            # Early start: crop beginning, pad end
            # Duplicate last few points to pad the end
            shift = abs(shift)
            pad_points = points[-shift:]
            shifted_points = torch.cat([points[shift:], pad_points], dim=0)
        else:
            shifted_points = points
            
        return shifted_points


class TemporalCutout(object):
    """Randomly mask temporal segments to prevent overfitting to specific temporal patterns."""
    def __init__(self, max_cutout_ratio=0.15, num_holes=(1, 3), prob=0.5):
        self.max_cutout_ratio = max_cutout_ratio
        self.num_holes = num_holes  # (min, max) number of holes
        self.prob = prob
        
    def __call__(self, points):
        if torch.rand(1) > self.prob:
            return points
            
        total_points = points.shape[0]
        num_holes = torch.randint(self.num_holes[0], self.num_holes[1] + 1, (1,)).item()
        
        result = points.clone()
        
        for _ in range(num_holes):
            # Random hole size
            hole_size = torch.randint(1, int(total_points * self.max_cutout_ratio) + 1, (1,)).item()
            hole_size = min(hole_size, total_points // 4)  # Don't mask more than 25% at once
            
            if hole_size == 0:
                continue
                
            # Random hole position
            start_idx = torch.randint(0, total_points - hole_size + 1, (1,)).item()
            end_idx = start_idx + hole_size
            
            # Strategy: Replace masked points with interpolated values or nearby points
            if start_idx > 0 and end_idx < total_points:
                # Linear interpolation between before and after
                alpha = torch.linspace(0, 1, hole_size).unsqueeze(1)
                before_point = result[start_idx - 1:start_idx]
                after_point = result[end_idx:end_idx + 1]
                interpolated = before_point * (1 - alpha) + after_point * alpha
                result[start_idx:end_idx] = interpolated
            else:
                # At boundaries, just duplicate nearby points
                if start_idx == 0:
                    # At beginning, use points from after the hole
                    result[start_idx:end_idx] = result[end_idx:end_idx + hole_size]
                else:
                    # At end, use points from before the hole
                    result[start_idx:end_idx] = result[start_idx - hole_size:start_idx]
        
        return result


class TemporalShuffle(object):
    """Shuffle small temporal windows to break temporal dependencies."""
    def __init__(self, window_size=5, num_shuffles=3, prob=0.3):
        self.window_size = window_size
        self.num_shuffles = num_shuffles
        self.prob = prob
        
    def __call__(self, points):
        if torch.rand(1) > self.prob:
            return points
            
        total_points = points.shape[0]
        result = points.clone()
        
        for _ in range(self.num_shuffles):
            if total_points < self.window_size:
                continue
                
            # Random window position
            start_idx = torch.randint(0, total_points - self.window_size + 1, (1,)).item()
            end_idx = start_idx + self.window_size
            
            # Shuffle points within the window
            window = result[start_idx:end_idx]
            shuffled_indices = torch.randperm(self.window_size)
            result[start_idx:end_idx] = window[shuffled_indices]
        
        return result
