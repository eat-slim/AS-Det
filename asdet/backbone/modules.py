from typing import Tuple, List, Literal
import torch
import torch.nn as nn
from torch import Tensor as Tensor

from .utils import MultiQueryAndGroup, SampleAndGather, build_mlp, build_linear_mlp, coordinate_distance, knn_gather
from .utils import pooling_func


class SetAbstractionMSG(nn.Module):
    """
    local feature extraction module for points
    sampling - grouping - pointnet

    The main difference between our implementation and official code of PointNet++ is:
        we apply scatter operation to avoid the waste of computing resources caused by padding

    Args:
        in_channel (int): Input channels of point cloud.
        num_point (int): Number of points.
        radii (Tuple[float]): Tuple of radius in each query.
        num_samples (Tuple[int]): Number of samples in each query.
        mlp_channels (Tuple[Tuple[int]]): Number of channels in each P.
        aggregation_channel (int): Out channels of aggregation multi-scale grouping features
        fps_mod (dict): Type of sampling method, valid mod ['F-FPS', 'D-FPS', 'FS', 'AS'].
        query_mods (Tuple[str]): Type of query method, valid mod
            ['hybrid', 'ball']
            - hybrid: KNN and distance not exceeding radius.
            - ball: distance not exceeding radius but not necessarily the nearest.
        dilated_group (bool): Whether to use dilated query.
        normalize_xyz (bool): Whether to normalize xyz with radii in each SA module.
        norm (str): Config normalization layer. Defaults to 'BN'.
        bias (bool): Whether to use bias in mlps. Defaults to True.
    """

    def __init__(self,
                 in_channel: int,
                 num_point: int,
                 radii: Tuple[float],
                 num_samples: Tuple[int],
                 mlp_channels: Tuple[Tuple[int]],
                 aggregation_channel: int,
                 fps_mod: dict,
                 query_mods: Tuple[str],
                 dilated_group: bool = True,
                 normalize_xyz: bool = False,
                 norm: Literal['BN', 'LN'] = 'BN',
                 bias: bool = True,
                 pooling: str = 'max',
                 res_channels: List[int] = None,
                 runtime_cfg: dict = {},
                 **kwargs):
        super().__init__()
        assert len(radii) == len(num_samples) == len(query_mods) == len(mlp_channels)
        assert isinstance(num_point, int)
        assert isinstance(fps_mod, dict)

        self.in_channel = in_channel
        self.num_point = num_point
        self.radii = radii
        self.num_samples = num_samples
        self.mlp_channels = mlp_channels
        self.aggregation_channel = aggregation_channel

        self.fps_mod = fps_mod
        self.query_mods = query_mods
        self.dilated_group = dilated_group
        self.normalize_xyz = normalize_xyz

        self.sampler = SampleAndGather(num_point=self.num_point,
                                       in_channel=self.in_channel,
                                       sampling_mod=self.fps_mod,
                                       runtime_cfg=runtime_cfg)

        self.mlps = nn.ModuleList()
        self.groupers = nn.ModuleList()
        for i in range(len(radii)):
            num_sample = num_samples[i]
            radius = radii[i]
            min_radius = radii[i - 1] if dilated_group and i != 0 else 0
            grouper = MultiQueryAndGroup(
                mod=query_mods[i], max_radius=radius, sample_num=num_sample, min_radius=min_radius,
                normalize_xyz=normalize_xyz)
            mlp = build_linear_mlp(in_channel=in_channel + 3, channel_list=mlp_channels[i], norm=norm, bias=bias)
            self.groupers.append(grouper)
            self.mlps.append(mlp)
        if aggregation_channel is not None:
            inc = sum([i[-1] for i in mlp_channels]) if len(mlp_channels) > 0 else in_channel
            self.aggregation_mlp = build_mlp(in_channel=inc, channel_list=[aggregation_channel],
                                             dim=1, norm=norm, bias=bias)
        else:
            self.aggregation_mlp = nn.Identity()
        if res_channels is not None:
            self.res_mlp = build_mlp(in_channel=res_channels[0], channel_list=res_channels[1:],
                                     dim=1, norm=norm, bias=bias)
        else:
            self.res_mlp = None
        self.pooling = pooling_func[pooling]

    def forward(self, points_xyz: Tensor, points_fea: Tensor, target_xyz: Tensor = None, target_fea: Tensor = None
                ) -> Tuple[Tensor, Tensor]:
        """S-G-P

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            points_fea (Tensor): (B, C, N) Features of each point.
            target_xyz (Tensor): (B, S, 3) Sample and group target xyz
            target_fea (Tensor): (B, C', S) Sample and group target features

        Returns:
            Tuple[Tensor]:

                - new_xyz: (B, S, 3) Where S is the number of points. New features xyz.
                - new_fea: (B, D, S) Where S is the number of points. New feature descriptors.
        """
        '''Sampling'''
        if target_xyz is None:
            new_xyz, new_fea, indices = self.sampler(points_xyz, points_fea)
            B, S, _ = new_xyz.shape
        else:
            new_xyz, new_fea, indices = target_xyz, None, None
            B, S, _ = target_xyz.shape

        new_fea_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            '''Grouping'''
            grouped_features, center_idx = grouper(points_xyz, new_xyz, points_fea, new_fea)  # (M, C+3) | (M,)

            '''Pointnet'''
            # (M, C+3) -> (M, Di) -> (B*S, Di) -> (B, Di, S)
            grouped_features = mlp(grouped_features)
            grouped_features = self.pooling(grouped_features, center_idx)
            grouped_features = grouped_features.reshape(B, S, grouped_features.shape[-1]).transpose(-1, -2).contiguous()

            new_fea_list.append(grouped_features)

        '''aggregating multi-scale grouping features'''
        if len(new_fea_list) > 0:
            # (B, D1+D2+...+Dn, S) -> (B, D, S)
            new_fea = torch.cat(new_fea_list, dim=1)
        new_fea = self.aggregation_mlp(new_fea)

        '''fusing target feature if given'''
        if self.res_mlp is not None and target_fea is not None:
            # (B, D+C', S) -> (B, D', S)
            new_fea = torch.cat([new_fea, target_fea], dim=1)
            new_fea = self.res_mlp(new_fea)
        elif self.res_mlp is not None and target_fea is None:
            raise RuntimeWarning('res_mlp in SA is existing, but \'target_fea\' is not given')

        return new_xyz, new_fea


class FeaturePropagation(nn.Module):
    """
    feature propagation for up-sampling
    """

    def __init__(self,
                 in_channel: List[int],
                 mlp: List[int],
                 norm: Literal['BN', 'LN', 'IN'] = 'BN',
                 bias: bool = True):
        super(FeaturePropagation, self).__init__()
        self.mlp = build_mlp(in_channel=sum(in_channel), channel_list=mlp, dim=1, norm=norm, bias=bias)

    def forward(self, points_xyz1: Tensor, points_xyz2: Tensor, points_fea1: Tensor, points_fea2: Tensor) -> Tensor:
        """
        Args:
            points_xyz1: (B, N, 3)
            points_xyz2: (B, S, 3)
            points_fea1: (B, D1, N)
            points_fea2: (B, D2, S)

        Returns:
            (B, D, N)
        """
        # 3-nn points as candidates
        dists = coordinate_distance(points_xyz1[..., :3], points_xyz2[..., :3])
        dists, idx = torch.topk(dists, k=3, dim=-1, largest=False)

        # weighted-sum based on distance (linear interpolate)
        dist_recip = 1.0 / dists.clamp(min=1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(knn_gather(points_fea2.transpose(1, 2), idx) * weight.unsqueeze(-1), dim=2)
        # (B, N, D2) -> (B, D2, N)
        new_fea = interpolated_points.transpose(1, 2)

        # skip connection akin to U-Net (B, D2, N) -> (B, D1+D2, N) -> (B, D, N)
        new_fea = torch.cat((points_fea1, new_fea), dim=1)
        new_fea = self.mlp(new_fea)
        return new_fea
