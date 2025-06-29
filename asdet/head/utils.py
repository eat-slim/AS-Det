import torch
from torch import Tensor
from functools import lru_cache
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes


def _topk_1d(scores: Tensor, batch_size: int, batch_idx: Tensor, obj: Tensor, K: int = 40, nuscenes: bool = False):
    # scores: (N, num_classes)
    topk_score_list = []
    topk_inds_list = []
    topk_classes_list = []

    for bs_idx in range(batch_size):
        batch_inds = batch_idx == bs_idx
        if obj.shape[-1] == 1 and not nuscenes:
            b_obj = scores[batch_inds].permute(1, 0)
            topk_scores, topk_inds = torch.topk(b_obj, K)
            topk_score, topk_ind = torch.topk(obj[topk_inds.view(-1)].squeeze(-1), K)
        else:
            b_obj = obj[batch_inds].permute(1, 0)
            topk_scores, topk_inds = torch.topk(b_obj, min(K, b_obj.shape[-1]))
            topk_score, topk_ind = torch.topk(topk_scores.view(-1), min(K, topk_scores.view(-1).shape[-1]))

        topk_classes = (topk_ind // K).int()
        topk_inds = topk_inds.view(-1).gather(0, topk_ind)

        if not obj is None and obj.shape[-1] == 1:
            topk_score_list.append(obj[batch_inds][topk_inds])
        else:
            topk_score_list.append(topk_score)
        topk_inds_list.append(topk_inds)
        topk_classes_list.append(topk_classes)

    topk_score = torch.stack(topk_score_list)
    topk_inds = torch.stack(topk_inds_list)
    topk_classes = torch.stack(topk_classes_list)

    return topk_score, topk_inds, topk_classes


def gather_feat_idx(feats: Tensor, inds: Tensor, batch_size: int, batch_idx: Tensor):
    feats_list = []
    dim = feats.shape[-1]
    _inds = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), dim)

    for bs_idx in range(batch_size):
        batch_inds = batch_idx == bs_idx
        feat = feats[batch_inds]
        feats_list.append(feat.gather(0, _inds[bs_idx]))
    feats = torch.stack(feats_list)
    return feats


@lru_cache()
def make_arange_tensor(end, device) -> Tensor:
    return torch.arange(end, device=device)


def judge_points_in_boxes(points: Tensor, bboxes_3d: LiDARInstance3DBoxes) -> Tensor:
    """
    judge which bbox the point is located in

    Args:
        points: (N, 3)
        bboxes_3d: (K,)

    Returns:
        (N, K)
    """
    corners = bboxes_3d.corners  # (K, 8, 3)

    min_z = corners[..., -1].min(1, keepdim=True)[0].T  # (1, K)
    max_z = corners[..., -1].max(1, keepdim=True)[0].T  # (1, K)
    inlier_z = (min_z <= points[:, 2:3]) & (points[:, 2:3] <= max_z)  # (N, K)

    A, B, C = corners[:, 0, :2][None], corners[:, 2, :2][None], corners[:, 4, :2][None]  # (1, K, 2)
    P = points[:, :2].unsqueeze(1)  # (N, 1, 2)
    AB, AC = B - A, C - A
    PA, PB, PC = A - P, B - P, C - P  # (N, K, 2)
    inlier_xy = (((PA * AB).sum(-1) * (PB * AB).sum(-1)) <= 0) & (((PA * AC).sum(-1) * (PC * AC).sum(-1)) <= 0)

    inlier_mask = inlier_xy & inlier_z  # (N, K)
    return inlier_mask
