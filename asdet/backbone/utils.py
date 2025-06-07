from typing import List, Tuple, Union, Literal
import torch
import torch.nn as nn
from torch import Tensor
import torch_scatter
from functools import lru_cache
from mmcv.ops import PointsSampler, gather_points
from mmcv.ops import ball_query as ball_query_mm

from asdet.sampling.active_sampling import ActiveSampling
from asdet.sampling.knn_mm import knn as knn_mm


class MultiQueryAndGroup(nn.Module):

    def __init__(self, mod: str, max_radius: float, sample_num: int, min_radius: float = 0.0,
                 normalize_xyz: bool = True, return_grouped_mask: bool = False):
        assert mod in ['hybrid', 'hybrid-t3d', 'ball', 'ball-t3d']
        super().__init__()
        self.radius = max_radius
        self.sample_num = sample_num
        self.normalize_xyz = normalize_xyz
        self.min_radius = min_radius
        self.return_grouped_mask = return_grouped_mask
        if min_radius > 0 and mod != 'ball':
            print(f'[UserWarning] dilated group is only available for \'ball\' currently, '
                  f'so the parameter \'min_radius\' is invalid for the \'{mod}\' you selected')
        if mod == 'ball':
            self._forward = self._ball_query
        elif mod == 'ball-t3d':
            self._forward = self._ball_query_t3d
        elif mod == 'hybrid':
            self._forward = self._hybrid_query
        elif mod == 'hybrid-t3d':
            self._forward = self._hybrid_query_t3d
        else:
            raise NotImplementedError

    def __repr__(self):
        desc = f'{self._forward.__name__}(R=({self.min_radius}, {self.radius}), ' \
               f'K={self.sample_num}, normalize_xyz={self.normalize_xyz})'
        return desc

    def forward(self, points_xyz: Tensor, center_xyz: Tensor, points_fea: Tensor, center_fea: Tensor = None
                ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            points_xyz (torch.Tensor): (B, N, 3) xyz coordinates of the points.
            center_xyz (torch.Tensor): (B, S, 3) coordinates of the centriods.
            points_fea (torch.Tensor): (B, C, N) The features of grouped points.
            center_fea (torch.Tensor): (B, C, S) The features of the centriods.

        Returns:
            Tuple | torch.Tensor: (B, 3 + C, S, K) Grouped concatenated coordinates and features of points.
        """
        grouped_features, grouped_idx, center_idx = self._forward(points_xyz, center_xyz, points_fea, center_fea)
        return grouped_features.contiguous(), center_idx

    def _ball_query(self, points_xyz: Tensor, center_xyz: Tensor, points_fea: Tensor, *args) -> Tuple[Tensor]:
        B, S, N, C = center_xyz.shape[0], center_xyz.shape[1], points_xyz.shape[1], points_fea.shape[1]
        K, radius = self.sample_num, self.radius
        device = points_xyz.device
        grouped_idx = ball_query_mm(self.min_radius, radius, K, points_xyz.contiguous(), center_xyz.contiguous())  # (B, S, K)
        mask = grouped_idx == grouped_idx[..., :1]
        mask[..., 0] = False

        # merge B & N and remove padding items  (B, S, K) -> (M,)
        points_xyz = points_xyz.reshape(-1, 3)  # (B, N, 3) -> (B*N, 3)
        points_fea = points_fea.transpose(-1, -2).reshape(-1, C)  # (B, C, N) -> (B*N, C)
        center_xyz = center_xyz.reshape(-1, 3)  # (B, S, 3) -> (B*S, 3)

        idx_offset = make_arange_tensor(B, device=device)[:, None, None] * N  # (B, 1, 1)
        grouped_idx = grouped_idx + idx_offset
        center_idx = make_arange_tensor(S, device=device)[None, :, None].repeat(B, 1, K)
        center_idx = center_idx + make_arange_tensor(B, device=device)[:, None, None] * S
        grouped_idx = grouped_idx[~mask]
        center_idx = center_idx[~mask]

        # get grouped features
        grouped_fea = points_fea[grouped_idx]  # (M, C)
        grouped_xyz = points_xyz[grouped_idx] - center_xyz[center_idx]  # (M, 3)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        grouped_features = torch.cat([grouped_xyz, grouped_fea], dim=1)  # (M, 3+C)
        return grouped_features, grouped_idx, center_idx

    def _ball_query_t3d(self, points_xyz: Tensor, center_xyz: Tensor, points_fea: Tensor, center_fea: Tensor
                        ) -> Tuple[Tensor, Tensor, Tensor]:
        B, S, N, C = center_xyz.shape[0], center_xyz.shape[1], points_xyz.shape[1], points_fea.shape[1]
        from pytorch3d.ops import ball_query
        K, radius = self.sample_num, self.radius
        device = points_xyz.device
        result = ball_query(p1=center_xyz, p2=points_xyz, K=K, radius=radius, return_nn=False)  # (B, S, K)

        # replace padding items with center points
        grouped_idx = result.idx
        mask = grouped_idx == -1
        if mask[..., 0].any():  # There is a center that groups nothing
            if center_fea is None:
                center_fea = torch.zeros(size=(B, points_fea.shape[1], S), dtype=torch.float, device=device)
            points_xyz = torch.cat([points_xyz, center_xyz], dim=1)  # (B, N+S, 3)
            points_fea = torch.cat([points_fea, center_fea], dim=-1)  # (B, C, N+S)
            padding_idx = torch.arange(start=N, end=N + S, device=device, dtype=torch.long)
            padding_idx = padding_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, K)  # (B, S, K)
            grouped_idx[mask] = padding_idx[mask]
            mask[..., 0] = False
            N += S

        # merge B & N and remove padding items  (B, S, K) -> (M,)
        points_xyz = points_xyz.reshape(-1, 3)  # (B, N, 3) -> (B*N, 3)
        points_fea = points_fea.transpose(-1, -2).reshape(-1, C)  # (B, C, N) -> (B*N, C)
        center_xyz = center_xyz.reshape(-1, 3)  # (B, S, 3) -> (B*S, 3)

        idx_offset = make_arange_tensor(B, device=device)[:, None, None] * N  # (B, 1, 1)
        grouped_idx = grouped_idx + idx_offset
        center_idx = make_arange_tensor(S, device=device)[None, :, None].repeat(B, 1, K)
        center_idx = center_idx + make_arange_tensor(B, device=device)[:, None, None] * S
        grouped_idx = grouped_idx[~mask]
        center_idx = center_idx[~mask]

        # get grouped features
        grouped_fea = points_fea[grouped_idx]  # (M, C)
        grouped_xyz = points_xyz[grouped_idx] - center_xyz[center_idx]  # (M, 3)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        grouped_features = torch.cat([grouped_xyz, grouped_fea], dim=1)  # (M, 3+C)
        return grouped_features, grouped_idx, center_idx

    def _hybrid_query(self, points_xyz: Tensor, center_xyz: Tensor, points_fea: Tensor, center_fea: Tensor
                      ) -> Tuple[Tensor, Tensor, Tensor]:
        B, S, N, C = center_xyz.shape[0], center_xyz.shape[1], points_xyz.shape[1], points_fea.shape[1]
        K, radius = self.sample_num, self.radius
        device = points_xyz.device
        grouped_idx, dist2 = knn_mm(K, points_xyz.contiguous(), center_xyz.contiguous(), False)

        # distance mask
        mask = dist2 > (self.radius ** 2)
        if mask[..., 0].any():
            if center_fea is None:
                center_fea = torch.zeros(size=(B, points_fea.shape[1], S), dtype=torch.float, device=device)
            points_xyz = torch.cat([points_xyz, center_xyz], dim=1)  # (B, N+S, 3)
            points_fea = torch.cat([points_fea, center_fea], dim=-1)  # (B, C, N+S)
            padding_idx = torch.arange(start=N, end=N + S, device=device, dtype=torch.int)
            padding_idx = padding_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, K)  # (B, S, K)
            grouped_idx[mask] = padding_idx[mask]
            mask[..., 0] = False
            N += S

        # merge B & N and remove padding items  (B, S, K) -> (M,)
        points_xyz = points_xyz.reshape(-1, 3)  # (B, N, 3) -> (B*N, 3)
        points_fea = points_fea.transpose(-1, -2).reshape(-1, C)  # (B, C, N) -> (B*N, C)
        center_xyz = center_xyz.reshape(-1, 3)  # (B, S, 3) -> (B*S, 3)

        idx_offset = make_arange_tensor(B, device=device)[:, None, None] * N  # (B, 1, 1)
        grouped_idx = grouped_idx + idx_offset
        center_idx = make_arange_tensor(S, device=device)[None, :, None].repeat(B, 1, K)
        center_idx = center_idx + make_arange_tensor(B, device=device)[:, None, None] * S
        grouped_idx = grouped_idx[~mask]
        center_idx = center_idx[~mask]

        # get grouped features
        grouped_fea = points_fea[grouped_idx]  # (M, C)
        grouped_xyz = points_xyz[grouped_idx] - center_xyz[center_idx]  # (M, 3)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        grouped_features = torch.cat([grouped_xyz, grouped_fea], dim=1)  # (M, 3+C)
        return grouped_features, grouped_idx, center_idx

    def _hybrid_query_t3d(self, points_xyz: Tensor, center_xyz: Tensor, points_fea: Tensor, center_fea: Tensor
                          ) -> Tuple[Tensor, Tensor, Tensor]:
        B, S, N, C = center_xyz.shape[0], center_xyz.shape[1], points_xyz.shape[1], points_fea.shape[1]
        from pytorch3d.ops import knn_points
        K, radius = self.sample_num, self.radius
        device = points_xyz.device
        result = knn_points(p1=center_xyz, p2=points_xyz, K=K, return_nn=False)

        # distance mask
        grouped_idx = result.idx
        mask = result.dists > (self.radius ** 2)
        if mask[..., 0].any():
            if center_fea is None:
                center_fea = torch.zeros(size=(B, points_fea.shape[1], S), dtype=torch.float, device=device)
            points_xyz = torch.cat([points_xyz, center_xyz], dim=1)  # (B, N+S, 3)
            points_fea = torch.cat([points_fea, center_fea], dim=-1)  # (B, C, N+S)
            padding_idx = torch.arange(start=N, end=N + S, device=device, dtype=torch.long)
            padding_idx = padding_idx.unsqueeze(0).unsqueeze(-1).repeat(B, 1, K)  # (B, S, K)
            grouped_idx[mask] = padding_idx[mask]
            mask[..., 0] = False
            N += S

        # merge B & N and remove padding items  (B, S, K) -> (M,)
        points_xyz = points_xyz.reshape(-1, 3)  # (B, N, 3) -> (B*N, 3)
        points_fea = points_fea.transpose(-1, -2).reshape(-1, C)  # (B, C, N) -> (B*N, C)
        center_xyz = center_xyz.reshape(-1, 3)  # (B, S, 3) -> (B*S, 3)

        idx_offset = make_arange_tensor(B, device=device)[:, None, None] * N  # (B, 1, 1)
        grouped_idx = grouped_idx + idx_offset
        center_idx = make_arange_tensor(S, device=device)[None, :, None].repeat(B, 1, K)
        center_idx = center_idx + make_arange_tensor(B, device=device)[:, None, None] * S
        grouped_idx = grouped_idx[~mask]
        center_idx = center_idx[~mask]

        # get grouped features
        grouped_fea = points_fea[grouped_idx]  # (M, C)
        grouped_xyz = points_xyz[grouped_idx] - center_xyz[center_idx]  # (M, 3)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        grouped_features = torch.cat([grouped_xyz, grouped_fea], dim=1)  # (M, 3+C)
        return grouped_features, grouped_idx, center_idx


class SampleAndGather(nn.Module):

    def __init__(self,
                 num_point: int,
                 in_channel: int,
                 sampling_mod: dict,
                 runtime_cfg: dict = {}) -> None:
        super().__init__()
        self.num_point = num_point
        self.in_channel = in_channel
        self.sampling_mod = sampling_mod
        self.sampling_type = sampling_mod['type']
        if self.sampling_type == 'AS':
            self.sampler = ActiveSampling(
                num_point=self.num_point,
                in_channel=in_channel,
                runtime_cfg=runtime_cfg,
                **self.sampling_mod,
            )
            self._forward = self.active_sampling
        elif self.sampling_type == 'FPS-t3d':
            self._forward = self.fps_t3d
        else:
            self.sampler = PointsSampler(
                num_point=[self.num_point],
                fps_mod_list=[self.sampling_type],
                fps_sample_range_list=[self.sampling_mod.get('fps_sample_range_list', -1)])
            self._forward = self.fps

    def __repr__(self):
        if self.sampling_type != 'AS':
            desc = f'{self.__class__.__name__}(S={self.num_point}, type={self.sampling_type})'
        else:
            desc = super().__repr__()
            insert_idx = len(self.__class__.__name__) + 1
            desc = desc[:insert_idx] + f'S={self.num_point}' + desc[insert_idx:]
        return desc

    def forward(self, points_xyz: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            points_xyz (torch.Tensor): (B, N, 3) xyz coordinates of the points.
            features (torch.Tensor): (B, C, N) features of the points.

        Returns:
            torch.Tensor: (B, S, 3) XYZ of sampled points.
            torch.Tensor: (B, C, S) features of sampled points.
            torch.Tensor: (B, S) Indices of sampled points.
        """
        return self._forward(points_xyz, features)

    def fps(self, points_xyz: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        indices = self.sampler(points_xyz, features)  # (B, S)
        new_xyz = gather_points(xyz_flipped, indices).transpose(1, 2).contiguous()  # (B, S, 3)
        new_fea = gather_points(features, indices).contiguous()  # (B, C, S)
        return new_xyz, new_fea, indices

    def active_sampling(self, points_xyz: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.sampler(points_xyz, features)

    def fps_t3d(self, points_xyz: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        from pytorch3d.ops import sample_farthest_points
        from pytorch3d.ops.utils import masked_gather
        new_xyz, indices = sample_farthest_points(points=points_xyz, K=self.num_point)
        new_fea = masked_gather(features.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()
        return new_xyz, new_fea, indices


@lru_cache()
def make_arange_tensor(end, device) -> Tensor:
    return torch.arange(end, device=device)


def coordinate_distance(src: Tensor, dst: Tensor) -> Tensor:
    """
    distance between two point sets
    ** This function is not compatible with FP16 and usually causes significant numerical errors under FP16 **

    Args:
        src: <torch.Tensor> (B, M, C)
        dst: <torch.Tensor> (B, N, C)
    Returns:
        <torch.Tensor> (B, M, N)
    """
    B, M, _ = src.shape
    _, N, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).view(B, M, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, N)
    return dist


def knn_gather(
    x: torch.Tensor, idx: torch.Tensor, lengths: Union[torch.Tensor, None] = None
):
    """
    A helper function for knn that allows indexing a tensor x with the indices `idx`
    returned by `knn_points`.

    For example, if `dists, idx = knn_points(p, x, lengths_p, lengths, K)`
    where p is a tensor of shape (N, L, D) and x a tensor of shape (N, M, D),
    then one can compute the K nearest neighbors of p with `p_nn = knn_gather(x, idx, lengths)`.
    It can also be applied for any tensor x of shape (N, M, U) where U != D.

    Args:
        x: Tensor of shape (B, N, C) containing C-dimensional features to
            be gathered.
        idx: LongTensor of shape (B, M, K) giving the indices returned by `knn_points`.
        lengths: LongTensor of shape (B,) of values in the range [0, N], giving the
            length of each example in the batch in x. Or None to indicate that every
            example has length M.
    Returns:
        x_out: Tensor of shape (B, M, K, C) resulting from gathering the elements of x
            with idx, s.t. `x_out[n, l, k] = x[n, idx[n, l, k]]`.
            If `k > lengths[n]` then `x_out[n, l, k]` is filled with 0.0.
    """
    N, M, U = x.shape
    _N, L, K = idx.shape

    if N != _N:
        raise ValueError("x and idx must have same batch dimension.")

    if lengths is None:
        lengths = torch.full((x.shape[0],), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    # idx_expanded has shape [N, L, K, U]

    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)
    # p2_nn has shape [N, L, K, U]

    needs_mask = lengths.min() < K
    if needs_mask:
        # mask has shape [N, K], true where idx is irrelevant because
        # there is less number of points in p2 than K
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]

        # expand mask to shape [N, L, K, U]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0

    return x_out


def build_mlp(in_channel: int, channel_list: List[int], dim: int = 2, bias: bool = False, drop_last_act: bool = False,
              norm: Literal['BN', 'LN', 'IN'] = 'BN', act: Literal['relu', 'elu'] = 'relu') -> nn.Sequential:
    """
    build MLPs with Conv1x1

    Args:
        in_channel: <int> input channels
        channel_list: <list[int]> channels of inner layers
        dim: <int> dim of tensor, 1 or 2
        bias: <bool> whether apply bias in convolution layers
        drop_last_act: <bool> whether drop the last activation function
        norm: config of normalization layer
        act: config of activation layer
    Returns:
        <torch.nn.ModuleList[torch.nn.Sequential]>
    """
    norm_1d = {'bn': nn.BatchNorm1d,
               'in': nn.InstanceNorm1d,
               'ln': LayerNorm1d}
    norm_2d = {'bn': nn.BatchNorm2d,
               'in': nn.InstanceNorm2d,
               'ln': LayerNorm2d}
    acts = {'relu': nn.ReLU,
            'elu': nn.ELU}

    if dim == 1:
        CONV = nn.Conv1d
        NORM = norm_1d.get(norm.lower(), nn.BatchNorm1d)
    else:
        CONV = nn.Conv2d
        NORM = norm_2d.get(norm.lower(), nn.BatchNorm2d)
    ACT = acts.get(act.lower(), nn.ReLU)

    mlp = []
    for channel in channel_list:
        # conv-norm-act
        mlp.append(CONV(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=bias))
        mlp.append(NORM(channel))
        mlp.append(ACT(inplace=True))
        in_channel = channel

    if drop_last_act:
        mlp = mlp[:-1]

    return nn.Sequential(*mlp)


def build_linear_mlp(in_channel: int, channel_list: List[int], bias: bool = False, drop_last_act: bool = False,
                     norm: Literal['BN', 'LN', 'IN'] = 'BN', act: Literal['relu', 'elu'] = 'relu') -> nn.Sequential:
    """
    build MLPs with Linear

    Args:
        in_channel: <int> input channels
        channel_list: <list[int]> channels of inner layers
        bias: <bool> whether apply bias in linear layers
        drop_last_act: <bool> whether drop the last activation function
        norm: config of normalization layer
        act: config of activation layer
    Returns:
        <torch.nn.ModuleList[torch.nn.Sequential]>
    """
    norms = {'bn': nn.BatchNorm1d,
             'in': nn.InstanceNorm1d,
             'ln': LayerNorm1d}
    acts = {'relu': nn.ReLU,
            'elu': nn.ELU}

    NORM = norms.get(norm.lower(), nn.BatchNorm1d)
    ACT = acts.get(act.lower(), nn.ReLU)

    mlp = []
    for channel in channel_list:
        # linear-norm-act
        mlp.append(nn.Linear(in_features=in_channel, out_features=channel, bias=bias))
        mlp.append(NORM(channel))
        mlp.append(ACT(inplace=True))
        in_channel = channel

    if drop_last_act:
        mlp = mlp[:-1]

    return nn.Sequential(*mlp)


class LayerNorm1d(nn.Module):
    """LayerNorm wrapper which supports tensor with shape of (B, C, N) as input"""

    def __init__(self, channel):
        super().__init__()
        self.ln = nn.LayerNorm(channel)

    def forward(self, x):
        """(B, C, N)"""
        out = self.ln(x.transpose(1, 2)).transpose(1, 2).contiguous()
        return out


class LayerNorm2d(nn.Module):
    """LayerNorm wrapper which supports tensor with shape of (B, C, H, W) as input"""

    def __init__(self, channel):
        super().__init__()
        self.ln = nn.LayerNorm(channel)

    def forward(self, x):
        """(B, C, H, W)"""
        out = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return out


def _max_pooling(points_fea: Tensor, center_idx: Tensor) -> Tensor:
    return torch_scatter.scatter_max(points_fea, center_idx, dim=0)[0]


def _mean_pooling(points_fea: Tensor, center_idx: Tensor) -> Tensor:
    return torch_scatter.scatter_mean(points_fea, center_idx, dim=0)


def _sum_pooling(points_fea: Tensor, center_idx: Tensor) -> Tensor:
    return torch_scatter.scatter_sum(points_fea, center_idx, dim=0)


pooling_func = dict(
    max=_max_pooling,
    mean=_mean_pooling,
    sum=_sum_pooling,
)
