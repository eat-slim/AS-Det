from typing import List, Optional, Tuple, Union, Dict
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.utils import multi_apply
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmdet3d.models.utils import clip_sigmoid

from asdet.backbone.modules import SetAbstractionMSG
from .base_head import BaseASHead
from .cbg_head import PointCBGHead as CBGHead
from .utils import make_arange_tensor, judge_points_in_boxes, _topk_1d, gather_feat_idx


@MODELS.register_module()
class ASCBGNusHead(BaseASHead):
    """
    VoteHead with multi-scale center feature aggregation and CBG head for nuScenes

    Args:
        tasks (list[dict], optional): Task information including class number and class names.
        pred_layer_cfg (dict): Config of classfication and regression prediction layers.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and decoding boxes.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center regression loss.
        size_res_loss (dict): Config of size regression loss.
        dir_res_loss (dict): Config of direction regression loss.
        velocity_loss (dict): Config of velocity regression loss.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
    """

    def __init__(self,
                 in_channels: int,
                 decoder_layer_cfg: dict,
                 tasks: List[dict],
                 pred_layer_cfg: dict,
                 bbox_coder: Union[ConfigDict, dict],
                 objectness_loss: dict,
                 center_loss: dict,
                 size_res_loss: dict,
                 dir_res_loss: dict,
                 velocity_loss: dict,
                 vote_limit: Optional[tuple] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        num_classes = [len(t['class_names']) for t in tasks]
        super(ASCBGNusHead, self).__init__(
            num_classes=sum(num_classes),
            bbox_coder=bbox_coder,
            objectness_loss=objectness_loss,
            center_loss=center_loss,
            size_res_loss=size_res_loss,
            dir_res_loss=dir_res_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

        # parse config
        if vote_limit is None:
            self.vote_limit = (math.inf, math.inf, math.inf)
        else:
            self.vote_limit = vote_limit
            assert len(self.vote_limit) == 3
        self.num_classes = [len(t['class_names']) for t in tasks]
        num_cnt = 0
        self.class_id_mapping_each_head = []
        for num_sub_cls in self.num_classes:
            self.class_id_mapping_each_head.append(torch.arange(num_sub_cls) + num_cnt)
            num_cnt += num_sub_cls
        self.loss_velocity = MODELS.build(velocity_loss)

        # proposals generator
        self.proposal_head = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=3, kernel_size=1, bias=True),
        )

        # decoder layers & prediction heads
        self.src_idx = decoder_layer_cfg['src_idx']
        decoder_in_channels = decoder_layer_cfg['in_channels']
        decoder_radii = decoder_layer_cfg['decoder_radii']
        decoder_num_samples = decoder_layer_cfg['decoder_num_samples']
        decoder_query_mods = decoder_layer_cfg['decoder_query_mods']
        decoder_sa_channels = decoder_layer_cfg['decoder_sa_channels']
        decoder_aggregation_channels = decoder_layer_cfg['decoder_aggregation_channels']
        decoder_pooling = decoder_layer_cfg['decoder_pooling']
        assert len(self.src_idx) == len(decoder_radii) == len(decoder_num_samples) == len(decoder_query_mods) == \
               len(decoder_sa_channels) == len(decoder_aggregation_channels)
        num_decoder_layers = len(decoder_radii)
        self.decoder_layers = nn.ModuleList()
        decoder_layer_out_channels = 0
        for i in range(num_decoder_layers):
            decoder_layer = SetAbstractionMSG(
                in_channel=decoder_in_channels[i], radii=decoder_radii[i], num_samples=decoder_num_samples[i],
                mlp_channels=decoder_sa_channels[i], aggregation_channel=decoder_aggregation_channels[i],
                query_mods=decoder_query_mods[i], pooling=decoder_pooling[i],
                num_point=-1, fps_mod={'type': 'FPS-t3d'}, dilated_group=False, bias=False)
            self.decoder_layers.append(decoder_layer)

            if decoder_aggregation_channels[i] is not None:
                decoder_layer_out_channels += decoder_aggregation_channels[i]
            else:
                decoder_layer_out_channels += sum([c[-1] for c in decoder_sa_channels[i]])

        self.bbox_head = CBGHead(
            in_channels=decoder_layer_out_channels,
            tasks=tasks,
            pred_layer_cfg=pred_layer_cfg)

    def forward(self, feat_dict: dict) -> dict:
        """Forward pass.

        Note:
            The forward of VoteHead is divided into 2 steps:

                1. Predict bbox and score.
                2. Decode predictions.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            dict: Predictions of head.
        """
        seed_xyz, seed_fea, _ = self._extract_input(feat_dict)

        vote_xyz = self.proposal_head(seed_fea)
        vote_xyz = vote_xyz.transpose(1, 2)
        for axis_id in range(3):
            limitation = self.vote_limit[axis_id]
            vote_xyz[..., axis_id].clamp_(max=limitation, min=-limitation)
        center_proposals = vote_xyz + seed_xyz

        predictions = self.decoding(feat_dict, center_proposals)

        B, C, N = seed_fea.shape
        seed_length = make_arange_tensor(B + 1, seed_xyz.device) * N
        seed_xyz = seed_xyz.reshape(B * N, 3)
        vote_xyz = vote_xyz.reshape(B * N, -1)
        proposal = dict(vote_xyz=vote_xyz)
        results = dict(seed_xyz=seed_xyz, length=seed_length, proposal=proposal, predictions_list=predictions)
        return results

    def decoding(self, feat_dict: Dict[str, List[Tensor]], center_proposals: Tensor) -> List[Tensor]:
        len_fp_out = len(feat_dict['fp_features'])
        hierarchical_features = feat_dict['sa_features'][::-1]
        hierarchical_xyz = feat_dict['sa_xyz'][::-1]
        hierarchical_features[1: 1 + len_fp_out] = feat_dict['fp_features']
        hierarchical_xyz[1: 1 + len_fp_out] = feat_dict['fp_xyz']
        hierarchical_features.reverse()
        hierarchical_xyz.reverse()
        assert len(hierarchical_xyz) >= len(self.src_idx)

        seed_xyz = center_proposals  # (B, N, k)
        obj_fea_list = []
        for i, decoder_layer in enumerate(self.decoder_layers):
            src = self.src_idx[i]
            xyz, features = hierarchical_xyz[src], hierarchical_features[src]

            object_xyz, object_fea = \
                decoder_layer(points_xyz=xyz, points_fea=features, target_xyz=seed_xyz, target_fea=None)
            obj_fea_list.append(object_fea)

        obj_fea = torch.cat(obj_fea_list, dim=1)  # (B, C1+C2+...+Cn, N)
        B, C, N = obj_fea.shape
        obj_xyz = seed_xyz.reshape(B * N, 3)
        obj_fea = obj_fea.transpose(-1, -2).reshape(B * N, C)
        cls_predictions_tasks, reg_predictions_tasks = self.bbox_head(obj_fea)
        predictions_tasks = []
        for cls_predictions, reg_predictions in zip(cls_predictions_tasks, reg_predictions_tasks):
            predictions = self.bbox_coder.split_pred(cls_predictions, reg_predictions, seed_xyz=obj_xyz)
            predictions_tasks.append(predictions)
        return predictions_tasks

    def loss_by_feat(
            self,
            points: List[torch.Tensor],
            bbox_preds_dict: dict,
            batch_gt_instances_3d: List[InstanceData],
            batch_pts_semantic_mask: Optional[List[torch.Tensor]] = None,
            batch_pts_instance_mask: Optional[List[torch.Tensor]] = None,
            batch_input_metas: List[dict] = None,
            ret_target: bool = False,
            **kwargs) -> dict:
        """Compute loss.

        Args:
            points (list[torch.Tensor]): Input points.
            bbox_preds_dict (dict): Predictions from forward of vote head.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of gt_instances.
                It usually includes ``bboxes_3d`` and ``labels_3d`` attributes.
            batch_pts_semantic_mask (list[tensor]): Semantic mask of points cloud. Defaults to None. Defaults to None.
            batch_pts_semantic_mask (list[tensor]): Instance mask of points cloud. Defaults to None. Defaults to None.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            ret_target (bool): Return targets or not.  Defaults to False.

        Returns:
            dict: Losses of 3DSSD.
        """
        seed_xyz = bbox_preds_dict['seed_xyz']
        length = bbox_preds_dict['length']
        proposal = bbox_preds_dict['proposal']
        predictions_list = bbox_preds_dict['predictions_list']

        # proposal
        seed_list = []  # (B1+B2+..., 3)
        for l in range(len(length) - 1):
            s, e = length[l], length[l + 1]
            batch_seed = seed_xyz[s: e].detach()
            seed_list.append(batch_seed)
        proposal_targets = self.get_proposal_targets(seed_list, batch_gt_instances_3d)
        vote_loss = self.get_proposal_loss(proposal, proposal_targets)

        # bbox cls & reg
        seed_xyz = predictions_list[0]['seed_xyz']
        seed_list = []  # (B1+B2+..., 3)
        for l in range(len(length) - 1):
            s, e = length[l], length[l + 1]
            batch_seed = seed_xyz[s: e].detach()
            seed_list.append(batch_seed)
        is_proposal = [False] * len(seed_list)

        targets = self.get_targets(seed_list, is_proposal,
                                   batch_gt_instances_3d,
                                   batch_pts_semantic_mask,
                                   batch_pts_instance_mask)

        centerness_loss, center_loss, size_res_loss, dir_res_loss, velocity_loss = \
            self.get_loss(predictions_list, targets)

        losses = dict(
            vote_loss=vote_loss,
            centerness_loss=centerness_loss,
            center_loss=center_loss,
            size_res_loss=size_res_loss,
            dir_res_loss=dir_res_loss,
            velocity_loss=velocity_loss,
        )
        return losses

    def get_proposal_targets(self, seed_xyz: List[Tensor], batch_gt_instances_3d: List[InstanceData]) -> Tuple[Tensor]:
        objectness_targets, vote_targets = [], []
        vote_weights = []
        batch_gt_labels_3d = [gt_instances_3d.labels_3d for gt_instances_3d in batch_gt_instances_3d]
        batch_gt_bboxes_3d = [gt_instances_3d.bboxes_3d for gt_instances_3d in batch_gt_instances_3d]
        for seed, gt_bboxes, gt_labels in zip(seed_xyz, batch_gt_bboxes_3d, batch_gt_labels_3d):
            front_mask, assignment, _ = \
                self._assign_targets_by_points_inside(gt_bboxes, dict(seed_xyz=seed, is_proposal=True))
            objectness_target = gt_labels[assignment]
            objectness_target[~front_mask] = -1
            vote_target = gt_bboxes[assignment].gravity_center[:, :3] - seed
            vote_weight = front_mask.int()
            objectness_targets.append(objectness_target)
            vote_targets.append(vote_target)
            vote_weights.append(vote_weight)
        objectness_targets = torch.cat(objectness_targets, dim=0)
        vote_targets = torch.cat(vote_targets, dim=0)
        vote_weights = torch.cat(vote_weights, dim=0)
        vote_weights = vote_weights / (vote_weights.sum() + 1e-8)
        return objectness_targets, vote_targets, vote_weights

    def get_proposal_loss(self, proposal: Dict[str, Tensor], targets: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        objectness_targets, vote_targets, vote_weights = targets
        vote_xyz = proposal['vote_xyz']

        # calculate center loss, only for positive
        vote_loss = []
        for i in objectness_targets.unique():
            if i < 0:
                continue
            cls_mask = (objectness_targets == i)
            vote_loss.append(F.smooth_l1_loss(vote_xyz[cls_mask], vote_targets[cls_mask]))
        vote_loss = torch.stack(vote_loss).mean()
        return vote_loss

    def get_targets(
            self,
            seed: List[Tensor],
            is_proposal: List[bool],
            batch_gt_instances_3d: List[InstanceData] = None,
            batch_pts_semantic_mask: List[torch.Tensor] = None,
            batch_pts_instance_mask: List[torch.Tensor] = None,
            **kwargs
    ) -> Tuple[Tensor]:
        """Generate targets of detection head.

        Args:
            seed_xyz (list[torch.Tensor]): Seed points of each batch.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of gt_instances.
                It usually includes ``bboxes`` and ``labels`` attributes.  Defaults to None.
            batch_pts_semantic_mask (list[tensor]): Semantic gt mask for point clouds.  Defaults to None.
            batch_pts_instance_mask (list[tensor]): Instance gt mask for point clouds.  Defaults to None.

        Returns:
            tuple[torch.Tensor]: Targets of 3DSSD head.
        """
        # gt: [(K,), ...], [(K, 7(x,y,z,l,w,h,yaw)), ...]
        batch_gt_labels_3d = [gt_instances_3d.labels_3d for gt_instances_3d in batch_gt_instances_3d]
        batch_gt_bboxes_3d = [gt_instances_3d.bboxes_3d for gt_instances_3d in batch_gt_instances_3d]

        # find empty example
        for index in range(len(batch_gt_labels_3d)):
            if len(batch_gt_labels_3d[index]) == 0:
                fake_box = batch_gt_bboxes_3d[index].tensor.new_zeros(1, batch_gt_bboxes_3d[index].tensor.shape[-1])
                batch_gt_bboxes_3d[index] = batch_gt_bboxes_3d[index].new_box(fake_box)
                batch_gt_labels_3d[index] = batch_gt_labels_3d[index].new_zeros(1)

        if batch_pts_semantic_mask is None:
            batch_pts_semantic_mask = [None for _ in range(len(batch_gt_labels_3d))]
            batch_pts_instance_mask = [None for _ in range(len(batch_gt_labels_3d))]

        (cls_targets, heatmap_targets, center_targets, size_targets, dir_res_targets, velocity_targets,
         positive_mask, negative_mask, assignment) = multi_apply(
            self.get_targets_single,
            seed, is_proposal,
            batch_gt_bboxes_3d, batch_gt_labels_3d, batch_pts_semantic_mask, batch_pts_instance_mask,
            )

        cls_targets = torch.cat(cls_targets, dim=0)
        heatmap_targets = [torch.cat(i, dim=0) for i in zip(*heatmap_targets)]
        center_targets = torch.cat(center_targets, dim=0)
        size_targets = torch.cat(size_targets, dim=0)
        dir_res_targets = torch.cat(dir_res_targets, dim=0)
        velocity_targets = torch.cat(velocity_targets, dim=0)
        positive_mask = torch.cat(positive_mask, dim=0)
        negative_mask = torch.cat(negative_mask, dim=0)
        assignment = torch.cat(assignment, dim=0)
        objectness_targets = positive_mask.clone().long()
        box_weights = positive_mask.clone().float().unsqueeze(-1)
        return cls_targets, objectness_targets, heatmap_targets, center_targets, size_targets, dir_res_targets, \
               velocity_targets, box_weights, assignment

    def get_targets_single(self,
                           seed: Optional[Tensor],
                           is_proposal: bool,
                           gt_bboxes_3d: LiDARInstance3DBoxes,
                           gt_labels_3d: Tensor,
                           pts_semantic_mask: Optional[Tensor] = None,
                           pts_instance_mask: Optional[Tensor] = None,
                           **kwargs):
        """Generate targets of AS head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (torch.Tensor): Point-wise semantic label of each batch.
            pts_instance_mask (torch.Tensor): Point-wise instance label of each batch.
            seed_xyz (torch.Tensor): key points from backbone.
            is_proposal: (bool): whether the prediction is the proposal

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None
        seed_xyz = seed
        gt_bboxes_3d = gt_bboxes_3d.to(seed_xyz.device)
        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        # Generate fake GT for empty scene
        if valid_gt.sum() == 0:
            raise RuntimeError('Found a frame with no ground truth')

        bbox_preds = {'seed_xyz': seed_xyz, 'is_proposal': is_proposal}
        front_mask, assignment, bbox_assigned = self._assign_targets_by_points_inside(gt_bboxes_3d, bbox_preds)
        positive_mask = front_mask
        negative_mask = ~front_mask

        cls_targets, center_targets, size_targets, dir_res_targets, velocity_targets = \
            self.bbox_coder.encode(gt_bboxes_3d[assignment], gt_labels_3d[assignment], seed_xyz)
        heatmap_targets = self.draw_heatmap(seed, gt_bboxes_3d, gt_labels_3d, bbox_assigned)

        return (cls_targets, heatmap_targets, center_targets, size_targets, dir_res_targets, velocity_targets,
                positive_mask, negative_mask, assignment)

    def _assign_targets_by_points_inside(self, gt_bboxes_3d: LiDARInstance3DBoxes, bbox_preds: dict) -> Tuple:
        """Compute assignment by Hungarian matching"""
        gt_center = gt_bboxes_3d.gravity_center[:, :3]  # (K, 3)
        seed_xyz = bbox_preds['seed_xyz'][:, :3]  # (N, 3)
        box_pos_cost = torch.norm(gt_center.unsqueeze(0) - seed_xyz.unsqueeze(1), dim=-1)  # (N, K)
        match_cost = box_pos_cost.clone()

        device = match_cost.device
        assignment = torch.zeros(seed_xyz.shape[0], dtype=torch.int64, device=device)
        points_mask = torch.zeros_like(match_cost, dtype=torch.int64, device=device)
        front_mask = torch.zeros(seed_xyz.shape[0], dtype=torch.bool, device=device)
        match_cost_np = match_cost.cpu().clone().detach().numpy()

        point_index, box_index = linear_sum_assignment(match_cost_np)
        point_index, box_index = torch.from_numpy(point_index).to(device), torch.from_numpy(box_index).to(device)

        assign_cost = match_cost[point_index, box_index]
        cost_mask = assign_cost <= self.train_cfg['nearest_thr']
        point_index = point_index[cost_mask]
        box_index = box_index[cost_mask]

        points_mask[point_index, box_index] = 1
        assignment[point_index] = box_index
        front_mask[point_index] = True
        is_assigned, bbox_nearest_assigned = points_mask.max(0)  # (K,)
        bbox_nearest_assigned[~(is_assigned.bool())] = -1

        if self.train_cfg.get('inner_assign', False) or bbox_preds['is_proposal']:
            if bbox_preds['is_proposal']:
                enlarged_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(self.train_cfg.get('expand_dims_length', 0.5))
            else:
                enlarged_gt_bboxes_3d = gt_bboxes_3d
            if device.type == 'cuda':
                inner_mask = enlarged_gt_bboxes_3d.points_in_boxes_all(seed_xyz).bool()  # (N, K)
            else:
                inner_mask = judge_points_in_boxes(seed_xyz, enlarged_gt_bboxes_3d)  # (N, K)
            is_front, bbox_id = inner_mask.max(1)  # (N,)
            is_ambiguity = inner_mask.sum(1) > 1
            if is_ambiguity.any():
                unique_bbox_id = (box_pos_cost[is_ambiguity] / inner_mask[is_ambiguity]).abs().argmin(1)
                bbox_id[is_ambiguity] = unique_bbox_id

            point_index = torch.where(is_front)[0]
            box_index = bbox_id[is_front]
            points_mask[point_index, box_index] = 1
            assignment[point_index] = box_index
            front_mask[is_front] = True

        return front_mask, assignment, bbox_nearest_assigned

    def get_loss(self, predictions, targets):
        cls_targets, objectness_targets, heatmap_targets, center_targets, size_targets, dir_res_targets, \
        velocity_targets, box_weights, assignment = targets

        N = cls_targets.shape[0]
        num_task, num_cls = len(self.num_classes), sum(self.num_classes)

        # (N, num_task)
        task_mask = cls_targets.new_zeros((N, num_task), dtype=torch.bool)
        start, end = 0, 0
        for i, task_cls in enumerate(self.num_classes):
            end += task_cls
            for cls_idx in range(start, end):
                task_mask[..., i] |= (cls_targets == cls_idx)
            start = end

        # (N, num_task, 1)
        box_weights = box_weights.repeat(1, num_task) * task_mask  # (N, num_task)
        box_weights /= (box_weights.sum(0, keepdim=True) + 1e-8)
        box_weights = box_weights.unsqueeze(-1)

        # classification  (N, num_task, [1,2])
        objectness_loss = 0
        for task_idx in range(num_task):
            objectness_pred = predictions[task_idx]['obj_scores']
            objectness_targets = heatmap_targets[task_idx]
            objectness_pred = clip_sigmoid(objectness_pred)
            num_pos = objectness_targets.eq(1).float().sum().item()
            loss_heatmap = self.loss_objectness(objectness_pred, objectness_targets, avg_factor=max(num_pos, 1))
            objectness_loss += loss_heatmap

        # bbox center  (N, num_task, 3)
        center_pred = torch.stack([task_pred['center_offset'] for task_pred in predictions], dim=-2)
        center_targets = center_targets.unsqueeze(1).repeat(1, num_task, 1)
        center_loss = self.loss_center(center_pred, center_targets, weight=box_weights)

        # bbox size  (N, num_task, 3)
        size_pred = torch.stack([task_pred['size'] for task_pred in predictions], dim=-2)
        size_targets = size_targets.unsqueeze(1).repeat(1, num_task, 1)
        size_res_loss = self.loss_size_res(size_pred, size_targets, weight=box_weights)

        # bbox dir  (N, num_task, 2)
        dir_res_pred = torch.stack([task_pred['dir_res_norm'] for task_pred in predictions], dim=-2)
        dir_res_targets = dir_res_targets.unsqueeze(1).repeat(1, num_task, 1)
        dir_res_loss = self.loss_dir_res(dir_res_pred, dir_res_targets, weight=box_weights)

        # bbox velocity  (N, num_task, 2)
        velocity_pred = torch.stack([task_pred['velocity'] for task_pred in predictions], dim=-2)
        velocity_targets = velocity_targets.unsqueeze(1).repeat(1, num_task, 1)
        velocity_loss = self.loss_velocity(velocity_pred, velocity_targets, weight=box_weights)

        return objectness_loss, center_loss, size_res_loss, dir_res_loss, velocity_loss

    def draw_heatmap(self, p: Tensor, gt_bboxes_3d: LiDARInstance3DBoxes, gt_labels_3d: Tensor,
                     bbox_assigned: Tensor) -> List[Tensor]:
        gaussian_overlap = self.train_cfg.get('gaussian_overlap', 0.1)
        min_radius = self.train_cfg.get('min_radius', 2)
        gaussian_type = self.train_cfg.get('gaussian_type', ['nearst', 'gt_center'])
        gaussian_ratio = self.train_cfg.get('gaussian_ratio', 1)
        scale = 0.075 * 8
        p_C = p / scale

        heatmap_for_tasks = []
        cls_id = 0
        for num_task_cls in self.num_classes:
            task_class_id = [cls_id + i for i in range(num_task_cls)]
            task_mask = gt_labels_3d == task_class_id[0]
            for i in task_class_id[1:]:
                task_mask |= (gt_labels_3d == i)
            task_bboxes = gt_bboxes_3d[task_mask]
            task_labels = gt_labels_3d[task_mask] - cls_id
            task_box_assigned = bbox_assigned[task_mask]

            heatmap = p.new_zeros(num_task_cls, p.shape[0])

            center = task_bboxes.gravity_center[:, :3]  # (K, 3)
            center = center / scale
            seed2box_dist = ((p_C.unsqueeze(1) - center.unsqueeze(0)) ** 2).sum(-1)  # (N, K)

            dxy = task_bboxes.dims[:, :2] / scale
            dx, dy = dxy[:, 0], dxy[:, 1]
            radius = gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
            radius = torch.clamp_min(radius.int(), min=min_radius)

            # draw heatmap for each object
            for k in range(len(task_bboxes)):
                if dx[k] <= 0 or dy[k] <= 0:
                    continue

                cur_class_id = task_labels[k]
                dist = seed2box_dist[:, k]

                if 'gt_center' in gaussian_type:
                    draw_gaussian_to_heatmap_voxels(
                        heatmap[cur_class_id], dist, radius[k].item() * gaussian_ratio)

                assigned = task_box_assigned[k]
                if ('nearst' in gaussian_type) and (assigned != -1):
                    draw_gaussian_to_heatmap_voxels(
                        heatmap[cur_class_id], distance(p_C[:, :3], p_C[assigned, :3]),
                        radius[k].item() * gaussian_ratio)
            heatmap_for_tasks.append(heatmap.T)
            cls_id += num_task_cls
        return heatmap_for_tasks

    def predict(self,
                points: List[torch.Tensor],
                feats_dict: Dict[str, torch.Tensor],
                batch_data_samples: List[Det3DDataSample],
                use_nms: bool = True,
                **kwargs) -> List[InstanceData]:
        """
        Args:
            points (list[tensor]): Point clouds of multiple samples.
            feats_dict (dict): Features from FPN or backbone..
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes meta information of data.
            use_nms (bool): Whether do the nms for predictions.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData contains 3d Bounding boxes and corresponding
            scores and labels.
        """
        preds_dict = self(feats_dict)
        batch_size = len(batch_data_samples)
        batch_input_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            batch_input_metas.append(metainfo)
        results_list = self.predict_by_feat(points, preds_dict, batch_input_metas)
        return results_list

    def predict_by_feat(self, points: List[torch.Tensor], preds_dict: dict, batch_input_metas: List[dict],
                        **kwargs) -> List[InstanceData]:
        batch_size = len(batch_input_metas)
        seed_xyz = preds_dict['seed_xyz']
        seed_length = preds_dict['length']
        batch_ids = seed_length.new_zeros((seed_length[-1],))
        device = seed_xyz.device
        self.class_id_mapping_each_head = [i.to(device) for i in self.class_id_mapping_each_head]
        for i, l in enumerate(range(len(seed_length) - 1)):
            s, e = seed_length[l], seed_length[l + 1]
            batch_ids[s: e] = i
        predictions = preds_dict['predictions_list']
        ret_dicts = [{
            'pred_bboxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for _ in range(batch_size)]

        # process detections of each task, including topk / score filter / NMS
        for i, prediction in enumerate(predictions):
            final_pred_dicts = self.decode_bbox_nuscenes(batch_ids, prediction, batch_input_metas)

            for b, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[i][final_dict['pred_labels'].long()]
                ret_dicts[b]['pred_bboxes'].append(final_dict['pred_bboxes'])
                ret_dicts[b]['pred_scores'].append(final_dict['pred_scores'])
                ret_dicts[b]['pred_labels'].append(final_dict['pred_labels'])

        # merge detections of each task & validity check
        results_list = []
        for b, ret_dict in enumerate(ret_dicts):
            temp_results = InstanceData()
            bboxes = torch.cat(ret_dict['pred_bboxes'], dim=0)
            bboxes = batch_input_metas[0]['box_type_3d'](
                bboxes.clone(),
                box_dim=bboxes.shape[-1],
                with_yaw=self.bbox_coder.with_rot,
                origin=(0.5, 0.5, 0.5),
            )
            scores = torch.cat(ret_dict['pred_scores'], dim=0)
            labels = torch.cat(ret_dict['pred_labels'], dim=0)

            # validity check (remove empty bbox)
            if device.type == 'cuda':
                inlier_mask = bboxes.points_in_boxes_all(points[b])  # (N, K)
            else:
                inlier_mask = judge_points_in_boxes(points[b], bboxes)  # (N, K)
            nonempty_box_mask = inlier_mask.sum(0) > 0
            bboxes = bboxes[nonempty_box_mask]
            scores = scores[nonempty_box_mask]
            labels = labels[nonempty_box_mask]

            max_output_num = self.test_cfg.get('max_output_num', 500)
            if len(bboxes) > max_output_num:
                _, num_limit_ids = torch.topk(scores, k=max_output_num)
                bboxes = bboxes[num_limit_ids]
                scores = scores[num_limit_ids]
                labels = labels[num_limit_ids]

            temp_results.bboxes_3d = bboxes
            temp_results.scores_3d = scores
            temp_results.labels_3d = labels
            temp_results.set_metainfo({'key_points': seed_xyz[seed_length[b]: seed_length[b + 1]]})
            results_list.append(temp_results)
        return results_list

    def decode_bbox_nuscenes(self, batch_ids: Tensor, prediction: dict,
                             batch_input_metas: List[dict]) -> List[Dict[str, Tensor]]:
        K = self.test_cfg.get('max_obj_per_sample', 500)
        score_thresh = self.test_cfg.get('score_thr', 0.0)
        batch_size = len(batch_input_metas)
        heat_map = prediction['obj_scores'].sigmoid()
        scores, inds, class_ids = _topk_1d(None, batch_size, batch_ids, heat_map, K=K, nuscenes=True)
        bboxes = self.bbox_coder.decode(prediction)
        final_bboxes = gather_feat_idx(bboxes, inds, batch_size, batch_ids)  # (B, K, 9)
        final_scores = scores.reshape(batch_size, K)
        final_labels = class_ids.reshape(batch_size, K)
        mask = final_scores >= score_thresh

        ret_pred_dicts = []
        for k in range(batch_size):
            cur_mask = mask[k]
            cur_bboxes = final_bboxes[k, cur_mask]
            cur_scores = final_scores[k, cur_mask]
            cur_labels = final_labels[k, cur_mask]
            cur_input_metas = batch_input_metas[k]

            # nms
            if cur_bboxes.shape[0] > 0:
                nms_type = self.test_cfg['nms_cfg']['type']
                if nms_type == 'rotate':
                    from mmdet3d.structures import xywhr2xyxyr
                    from mmdet3d.models.layers import nms_bev
                    bev = cur_input_metas['box_type_3d'](
                        cur_bboxes.clone(),
                        box_dim=cur_bboxes.shape[-1],
                        with_yaw=self.bbox_coder.with_rot,
                        origin=(0.5, 0.5, 0.5)).bev
                    bboxes_for_nms = xywhr2xyxyr(bev)
                    scores_for_nms = cur_scores
                    classes_for_nms = cur_labels
                    nms_keep = []
                    for class_id in classes_for_nms.unique():
                        class_select_ids = torch.where(classes_for_nms == class_id)[0]
                        bboxes_nms_i = bboxes_for_nms[class_select_ids, :]
                        scores_i = scores_for_nms[class_select_ids]
                        if len(bboxes_nms_i) == 0:
                            continue
                        selected = nms_bev(bboxes_nms_i, scores_i, self.test_cfg.nms_cfg['iou_thr'])
                        nms_keep.append(class_select_ids[selected])
                    nms_keep = torch.cat(nms_keep, dim=0)
                else:
                    raise NotImplementedError(f'{nms_type} is not implemented')
                cur_bboxes = cur_bboxes[nms_keep]
                cur_scores = cur_scores[nms_keep]
                cur_labels = cur_labels[nms_keep]

            ret_pred_dicts.append({
                'pred_bboxes': cur_bboxes,
                'pred_scores': cur_scores,
                'pred_labels': cur_labels,
            })
        return ret_pred_dicts

    def _get_reg_out_channels(self) -> int:
        raise NotImplementedError


def distance(voxel_indices, center):
    distances = ((voxel_indices - center.unsqueeze(0)) ** 2).sum(-1)
    return distances


def gaussian_radius(height, width, min_overlap=0.5):
    """
    Args:
        height: (N)
        width: (N)
        min_overlap:
    Returns:
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2

    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def draw_gaussian_to_heatmap_voxels(heatmap, distances, radius, k=1):
    diameter = 2 * radius + 1
    sigma = diameter / 6
    masked_gaussian = torch.exp(- distances / (2 * sigma * sigma))

    torch.max(heatmap, masked_gaussian, out=heatmap)

    return heatmap

