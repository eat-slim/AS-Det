from typing import List
import torch
import torch.nn as nn
from torch import Tensor


class PointSeparateHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 separate_head_cfg: dict,
                 init_bias: float = -2.19,
                 **kwargs) -> None:
        super().__init__()
        self.heads = nn.ModuleList()
        for i, single_head_cfg in enumerate(separate_head_cfg):
            out_channels = single_head_cfg['out_channels']
            num_conv = single_head_cfg['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list += [
                    nn.Linear(in_channels, in_channels, bias=False),
                    nn.BatchNorm1d(in_channels),
                    nn.ReLU(True),
                ]
            fc_list.append(nn.Linear(in_channels, out_channels, bias=True))
            fc = nn.Sequential(*fc_list)

            # i = 0 -> classification head, init bias = -2.19 -> sigmoid(-2.19) = 0.1
            if i == 0:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Linear):
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.heads.append(fc)

    def forward(self, p: Tensor):
        cls = self.heads[0](p)

        bbox = []
        for head in self.heads[1:]:
            bbox.append(head(p))
        bbox = torch.cat(bbox, dim=-1)  # (N, [x, y, z, l, w, h, sin, cos, vx, vy])

        return cls, bbox


class PointCBGHead(nn.Module):
    """

    Args:
        tasks (list[dict]): Task information including class number and class names.
        pred_layer_cfg (dict): Config of classification and regression prediction layers.
    """

    def __init__(self,
                 in_channels: int,
                 tasks: List[dict],
                 pred_layer_cfg: dict,
                 **kwargs) -> None:
        super().__init__()

        # predict each set of object independently
        self.task_heads = nn.ModuleList()
        for task in tasks:
            pred_layer_cfg['separate_head_cfg'][0]['out_channels'] = len(task['class_names'])
            self.task_heads.append(
                PointSeparateHead(in_channels=in_channels, **pred_layer_cfg)
            )

    def forward(self, p: Tensor):
        """Forward.

        Args:
            p (Tensor): Input Tensor

        Returns:
            List[Tensor]: Class scores predictions from multi tasks
            List[Tensor]: Regression predictions from multi tasks
        """
        # separate branches for multi tasks
        cls_score_list, bbox_pred_list = [], []
        for task_head in self.task_heads:
            cls_score, bbox_pred = task_head(p)
            cls_score_list.append(cls_score)
            bbox_pred_list.append(bbox_pred)

        return cls_score_list, bbox_pred_list
