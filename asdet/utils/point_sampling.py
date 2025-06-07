from typing import List
from torch import Tensor
import torch
import numpy as np


def voxel_sampling(points: List[Tensor], voxel_size) -> List[Tensor]:
    voxel_size = points[0].new_tensor(voxel_size).reshape(1, 3)
    sampling_points = []
    for p in points:
        pcd_xyz = p[:, :3]
        xyz_min = pcd_xyz.min(0, keepdim=True)[0]
        pcd_xyz = pcd_xyz - xyz_min

        voxel_id = (pcd_xyz // voxel_size).int()
        _, choices = np.unique(voxel_id.cpu(), axis=0, return_index=True)
        choices = torch.from_numpy(choices).to(p.device)
        sampling_p = p[choices]
        sampling_points.append(sampling_p)
    return sampling_points
