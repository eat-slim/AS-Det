from typing import Tuple, Literal
import torch
import torch.nn as nn
from torch import Tensor as Tensor

from mmdet3d.registry import MODELS
from mmdet3d.models.backbones.base_pointnet import BasePointNet
from mmdet3d.utils import OptConfigType
from .modules import SetAbstractionMSG, FeaturePropagation

ThreeTupleIntType = Tuple[Tuple[Tuple[int, int, int]]]
TwoTupleIntType = Tuple[Tuple[int, int, int]]
TwoTupleStrType = Tuple[Tuple[str]]


@MODELS.register_module()
class PointNet2AS(BasePointNet):
    """PointNet++ with active sampling.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA module samples.
        radii (tuple[tuple[float]]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for query in each SA module.
        sa_channels (tuple[tuple[tuple[int]]]): Out channels of each mlp in SA module.
        aggregation_channels (tuple[int]): Out channels of aggregation multi-scale grouping features.
        fps_mods (tuple[dict]): kwargs of sampling for each SA module.
        query_mods (tuple[tuple[str]]): Mod of query for each SA and IRM module.
        dilated_group (tuple[bool]): Whether to use dilated query
        fp_channels (tuple[tuple[int]]): channels of each mlp in FP module.
        normalize_xyz (tuple[bool]): Whether to normalize xyz with radii in each SA module.
        norm (str): Config of normalization layer.
        bias (bool): Use bias in mlps
        out_indices (Sequence[int]): Output from which stages.

    """

    def __init__(self,
                 in_channels: int,
                 num_points: Tuple[int] = (4096, 1024, 512, 256),
                 radii: Tuple[Tuple[float, float, float]] = (
                         (0.2, 0.4, 0.8),
                         (0.4, 0.8, 1.6),
                         (1.6, 3.2, 4.8),
                         (4.8, 6.4)
                 ),
                 num_samples: TwoTupleIntType = ((32, 32, 64), (32, 32, 64), (32, 32, 32), (16, 32)),
                 sa_channels: ThreeTupleIntType = (((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                                                   ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                                                   ((128, 128, 256), (128, 192, 256), (128, 256, 256)),
                                                   ((256, 256, 512), (256, 512, 1024))),
                 aggregation_channels: Tuple[int] = (64, 128, 256, 256),
                 fps_mods: Tuple[dict] = (dict(type='D-FPS'), dict(type='AS'), dict(type='AS'), dict(type='AS')),
                 query_mods: TwoTupleStrType = (
                         ('ball', 'ball', 'ball'),
                         ('ball', 'ball', 'ball'),
                         ('ball', 'ball', 'ball'),
                         ('ball', 'ball')
                 ),
                 dilated_group: Tuple[bool] = (True, True, True, True),
                 fp_channels: Tuple[Tuple[int]] = ((256, 256), (256, 256)),
                 normalize_xyz: bool = False,
                 norm: Literal['BN', 'LN'] = 'BN',
                 bias: bool = False,
                 out_indices: Tuple[int] = (0, 1, 2, 3),
                 init_cfg: OptConfigType = None,
                 runtime_cfg: dict = {},
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        assert len(num_points) == len(radii) == len(num_samples) == len(sa_channels) == len(aggregation_channels)
        assert len(fp_channels) <= (len(num_points) - 1)
        self.in_channel = in_channels
        self.out_indices = out_indices
        self.downsample_layers = len(num_points)
        self.upsample_layers = len(fp_channels)
        self.runtime_cfg = runtime_cfg
        self.as_layers = []
        for i, mod in enumerate(fps_mods):
            if 'AS' in mod['type']:
                self.as_layers.append(i + 1)

        # ======================== Build Network ========================
        sa_out_channels = [in_channels - 3]
        self.downsampler = nn.ModuleList()
        for i in range(self.downsample_layers):
            sa_out_channel = sum([c[-1] for c in sa_channels[i]]) \
                if aggregation_channels[i] is None else aggregation_channels[i]
            sa_out_channels.append(sa_out_channel)
            self.downsampler.append(
                SetAbstractionMSG(
                    in_channel=sa_out_channels[i],
                    num_point=num_points[i],
                    radii=radii[i],
                    num_samples=num_samples[i],
                    mlp_channels=sa_channels[i],
                    aggregation_channel=aggregation_channels[i],
                    fps_mod=fps_mods[i],
                    query_mods=query_mods[i],
                    dilated_group=dilated_group[i],
                    normalize_xyz=normalize_xyz,
                    norm=norm,
                    bias=bias,
                    runtime_cfg=self.runtime_cfg))

        self.upsampler = nn.ModuleList()
        for i in range(self.upsample_layers):
            self.upsampler.append(
                FeaturePropagation(
                    in_channel=[sa_out_channels[-1 - i], sa_out_channels[-2 - i]],
                    mlp=fp_channels[i], norm=norm, bias=bias))

    def forward(self, points: Tensor) -> dict:
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features, with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the input points.
        """
        xyz, features = self._split_point_feats(points)

        sa_xyz = [xyz]
        sa_features = [features]

        out_sa_xyz = [xyz]
        out_sa_features = [features]
        out_sa_indices = [None]

        # downsample stage
        for i in range(self.downsample_layers):
            cur_xyz, cur_features = self.downsampler[i](sa_xyz[-1], sa_features[-1])
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            if i in self.out_indices:
                out_sa_xyz.append(sa_xyz[-1])
                out_sa_features.append(sa_features[-1])
                out_sa_indices.append(None)

        # upsample stage
        fp_xyz, fp_features = [], []
        for i in range(self.upsample_layers):
            points_xyz1, points_xyz2 = sa_xyz[-2 - i], sa_xyz[-1 - i]
            points_fea1, points_fea2 = sa_features[-2 - i], sa_features[-1 - i]
            new_fea = self.upsampler[i](points_xyz1, points_xyz2, points_fea1, points_fea2)
            fp_xyz.append(points_xyz1)
            fp_features.append(new_fea)

        return dict(
            sa_xyz=out_sa_xyz,
            sa_features=out_sa_features,
            sa_indices=out_sa_indices,
            fp_xyz=fp_xyz,
            fp_features=fp_features,
            as_layers=self.as_layers,
            runtime=self.runtime_cfg
        )
