import torch
from torch import Tensor


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

