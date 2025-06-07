from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import lru_cache


class ActiveSampling(nn.Module):
    """
    differentiable sampling
    """
    def __init__(self, num_point: int, in_channel: int, runtime_cfg: dict, use_st: bool = True, **kwargs):
        super().__init__()
        self.num_point = num_point
        self.use_st = use_st
        self.runtime_cfg = runtime_cfg

        self.W = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channel, out_channels=1, kernel_size=1),
        )
        self.W[-1].bias.data.fill_(0.0)

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
        B, N, S = points_xyz.shape[0], points_xyz.shape[1], self.num_point

        # predict sampling distribution (B, 1, N)
        logits = self.W(features)
        logits_act = F.softplus(logits)
        sampling_soft = logits_act / (logits_act.sum(-1, keepdim=True) + 1e-8)  # (B, 1, N)

        # sampling
        sampling_list = torch.multinomial(sampling_soft.squeeze(1), num_samples=S, replacement=False).unsqueeze(-1)

        if self.use_st and self.training:
            # straight-through ensures the sampling generating gradients
            sampling_hard = torch.zeros(size=(B, S, N), device=logits.device).scatter_(-1, sampling_list, 1.0)
            score = sampling_hard - sampling_soft.detach() + sampling_soft
            new_xyz = score @ points_xyz  # (B, S, N) @ (B, N, 3) => (B, S, 3)
            new_fea = score @ features.transpose(1, 2).contiguous()  # (B, S, N) @ (B, N, C) => (B, S, C)
            new_fea = new_fea.transpose(1, 2).contiguous()  # (B, S, C) => (B, C, S)
            indices = sampling_list.squeeze(-1)  # (B, S)
        else:
            # gradients are not required during inference
            sampling_list = sampling_list.squeeze(-1)
            batch_ids = make_batch_tensor(B, S, logits.device)
            new_xyz = points_xyz[batch_ids, sampling_list]
            new_fea = features.transpose(1, 2)[batch_ids, sampling_list]
            new_fea = new_fea.transpose(1, 2).contiguous()
            indices = sampling_list

        # recording middle results for loss estimation
        if self.training:
            if self.runtime_cfg.get('sa_logits_act', None) is None:
                self.runtime_cfg['sa_logits_act'] = []
            self.runtime_cfg['sa_logits_act'].append(logits_act)

        return new_xyz, new_fea, indices


@lru_cache()
def make_batch_tensor(B, S, device) -> Tensor:
    return torch.arange(B, device=device)[:, None].repeat(1, S)

