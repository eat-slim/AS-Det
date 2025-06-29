from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import BasePoints


@TRANSFORMS.register_module()
class NuscenesPointSample(BaseTransform):
    """Point sample for NuScenes.

    Sampling data for spliced consecutive frames to a certain number.

    Args:
        num_points_keyframe (int): Number of points in the key frame to be sampled.
        num_points_others (int): Number of points in other frames to be sampled.
        voxel_size (tuple): Voxel size for sampling.
            Defaults to (0.1, 0.1, 0.1).
        replace (bool): Whether the sampling is with or without replacement.
            Defaults to False.
        timestamp_dim (int): Dimensions index used to indicate timestamp differences.
            Defaults to -1.
    """

    def __init__(self,
                 num_points_keyframe: int,
                 num_points_others: int,
                 voxel_size: Tuple[float] = (0.1, 0.1, 0.1),
                 replace: bool = False,
                 timestamp_dim: int = -1) -> None:
        self.num_points_keyframe = num_points_keyframe
        self.num_points_others = num_points_others
        self.voxel_size = voxel_size
        self.replace = replace
        self.timestamp_dim = timestamp_dim

    def _points_voxel_sampling(
        self,
        points: BasePoints,
        voxel_size: Tuple[float] = (0.1, 0.1, 0.1),
        return_choices: bool = False
    ) -> Union[Tuple[BasePoints, np.ndarray], BasePoints]:
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            voxel_size (tuple): Voxel size for sampling.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:

                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        pcd_xyz = points.coord.numpy()
        voxel_size = np.array(voxel_size, dtype=pcd_xyz.dtype)

        xyz_min = np.min(pcd_xyz, axis=0)
        xyz_max = np.max(pcd_xyz, axis=0)
        X, Y, Z = ((xyz_max - xyz_min) / voxel_size).astype(np.int32) + 1

        relative_xyz = pcd_xyz - xyz_min
        voxel_xyz = (relative_xyz / voxel_size).astype(np.int32)
        voxel_id = voxel_xyz[:, 0] + voxel_xyz[:, 1] * X + voxel_xyz[:, 2] * X * Y

        _, choices = np.unique(voxel_id, return_index=True)
        choices = torch.from_numpy(choices)

        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def _points_random_sampling(
        self,
        points: BasePoints,
        num_samples: Union[int, float],
        sample_range: Optional[float] = None,
        replace: bool = False,
        return_choices: bool = False
    ) -> Union[Tuple[BasePoints, np.ndarray], BasePoints]:
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int, float): Number of samples to be sampled. If
                float, we sample random fraction of points from num_points
                to 100%.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool): Sampling with or without replacement.
                Defaults to False.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:

                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if isinstance(num_samples, float):
            assert num_samples < 1
            num_samples = int(
                np.random.uniform(self.num_points, 1.) * points.shape[0])

        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            dist = np.linalg.norm(points.coord.numpy(), axis=1)
            far_inds = np.where(dist >= sample_range)[0]
            near_inds = np.where(dist < sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(
                    far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def transform(self, input_dict: dict) -> dict:
        """Transform function to sample points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']

        # split key frame and other frames
        keyframe_mask = points.tensor[:, self.timestamp_dim] == 0
        others_mask = ~keyframe_mask
        if 'lidar_sweeps' not in input_dict:
            # key frame without non-key frame
            others_mask[:] = True
        keyframe_points = points[keyframe_mask]
        other_points = points[others_mask]

        # sampling from key frame
        keyframe_points, choices1 = self._points_voxel_sampling(
            keyframe_points,
            self.voxel_size,
            return_choices=True
        )
        keyframe_points, choices2 = self._points_random_sampling(
            keyframe_points,
            self.num_points_keyframe,
            None,
            self.replace,
            return_choices=True)
        keyframe_choices = torch.where(keyframe_mask)[0][choices1][choices2]

        # sampling from other frames
        other_points, choices1 = self._points_voxel_sampling(
            other_points,
            self.voxel_size,
            return_choices=True
        )
        other_points, choices2 = self._points_random_sampling(
            other_points,
            self.num_points_others,
            None,
            self.replace,
            return_choices=True)
        other_choices = torch.where(others_mask)[0][choices1][choices2]

        # merge points and choices
        points = keyframe_points.cat([keyframe_points, other_points])
        choices = torch.cat([keyframe_choices, other_choices], dim=0)

        # other conventional operations
        input_dict['points'] = points

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            input_dict['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points_keyframe + self.num_points_others}' \
                    f'({self.num_points_keyframe} + {self.num_points_others}),'
        repr_str += f' voxel_size={self.voxel_size},'
        repr_str += f' replace={self.replace})'

        return repr_str
