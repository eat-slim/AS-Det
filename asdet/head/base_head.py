from typing import List, Optional, Tuple, Union, Dict
import io
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from pytorch3d.ops import knn_points, knn_gather
from scipy.optimize import linear_sum_assignment

from mmcv.ops.nms import batched_nms
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample
from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes, LiDARInstance3DBoxes
from asdet.loss.fkl_div import focal_kl_div


class BaseASHead(BaseModule):
    """
    base head for active sampling

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_res_loss (dict): Config of size residual regression loss.
        corner_loss (dict): Config of bbox corners regression loss.
    """

    def __init__(self,
                 num_classes: int,
                 bbox_coder: Union[ConfigDict, dict],
                 proposal_coder: Union[ConfigDict, dict],
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 objectness_loss: Optional[dict] = None,
                 center_loss: Optional[dict] = None,
                 dir_class_loss: Optional[dict] = None,
                 dir_res_loss: Optional[dict] = None,
                 size_res_loss: Optional[dict] = None,
                 corner_loss: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super(BaseASHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_objectness = MODELS.build(objectness_loss)
        self.loss_center = MODELS.build(center_loss)
        self.loss_dir_res = MODELS.build(dir_res_loss)
        self.loss_dir_class = MODELS.build(dir_class_loss)
        self.loss_size_res = MODELS.build(size_res_loss)
        self.loss_corner = MODELS.build(corner_loss)

        self.proposal_coder = TASK_UTILS.build(proposal_coder)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        self.figure_tobe_show_in_tensorboard = None

    def loss(self,
             points: List[Tensor],
             feats_dict: Dict[str, Tensor],
             batch_data_samples: List[Det3DDataSample],
             ret_target: bool = False,
             **kwargs) -> dict:
        """

        Args:
            points (list[tensor]): Points cloud of multiple samples.
            feats_dict (dict): Predictions from backbone or FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item contains the meta information of each sample
                and corresponding annotations.
            ret_target (bool): Whether return the assigned target. Defaults to False.

        Returns:
            dict:  A dictionary of loss components.
        """
        active_sampling_loss = self.active_sampling_loss(points, feats_dict, batch_data_samples)
        detection_loss = self.detection_loss(points, feats_dict, batch_data_samples, ret_target, **kwargs)
        losses = dict(detection_loss, **active_sampling_loss)
        return losses

    def active_sampling_loss(self,
                             points: List[torch.Tensor],
                             feats_dict: Dict[str, torch.Tensor],
                             batch_data_samples: List[Det3DDataSample]) -> dict:
        """
        Args:
            points (list[tensor]): Points cloud of multiple samples.
            feats_dict (dict): Predictions from backbone or FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item contains the meta information of each sample
                and corresponding annotations.

        Returns:
            dict:  A dictionary of loss components.
        """
        default_as_cfg = dict(
            density_radii=[0.8, 1.6, 4.8],
            density_K=16,
            gt_extra_width=0.05,
        )
        as_cfg = self.train_cfg.get('as_cfg', default_as_cfg)
        density_radii = as_cfg.get('density_radii', [0.8, 1.6, 4.8])
        density_K = as_cfg.get('density_K', 16)
        gt_extra_dim = as_cfg.get('gt_extra_dim', 0.05)

        layer_with_as = feats_dict.get('as_layers', [])  # List[int], layer index of which contains as layer
        if len(layer_with_as) == 0:
            return {}
        B = len(batch_data_samples)
        sa_xyz = feats_dict['sa_xyz']
        sa_logits_act = feats_dict['runtime'].pop('sa_logits_act')
        as_loss = [[] for _ in range(len(layer_with_as))]

        for batch_index in range(B):
            gt_instance = batch_data_samples[batch_index].gt_instances_3d
            gt_centers = gt_instance.bboxes_3d.gravity_center  # (K, 3)
            enlarged_gt_bboxes_3d = gt_instance.bboxes_3d.enlarged_box(gt_extra_dim)

            if self.figure_tobe_show_in_tensorboard is None:
                plt.figure(figsize=(3 * 10, len(layer_with_as) * 10), dpi=100)
                subplot_index = 1

            for as_index, layer_index in enumerate(layer_with_as):
                '''get predicted distribution'''
                current_sa_logits = sa_logits_act[as_index][batch_index].squeeze(0)  # (N,)
                current_sa_score = current_sa_logits / (current_sa_logits.sum() + 1e-8)
                sampling_distribution = torch.log(current_sa_score + 1e-8)
                current_sa_score = current_sa_score.unsqueeze(0)

                '''generate gt distribution (N,)'''
                sampling_input = sa_xyz[layer_index - 1][batch_index].detach()  # (N, 3)
                point2box_dist = torch.norm(sampling_input.unsqueeze(1) - gt_centers.unsqueeze(0), dim=-1)  # (N, K)

                # context sampling. apply Gaussian distribution function with sigma=1
                gt_logits = torch.exp(-(point2box_dist ** 2) / (2 * (1 ** 2)))
                context = gt_instance.bboxes_3d.enlarged_box(density_radii[layer_index - 2])
                if sampling_input.device.type == 'cuda':
                    valid_context_mask = context.points_in_boxes_all(sampling_input).bool()
                else:
                    # the implementation of MMDet3D doesn't support non CUDA devices
                    valid_context_mask = judge_points_in_boxes(sampling_input, context)
                gt_logits[~valid_context_mask] = 0

                # initial points' value
                if sampling_input.device.type == 'cuda':
                    inlier_mask = enlarged_gt_bboxes_3d.points_in_boxes_all(sampling_input).bool()
                else:
                    inlier_mask = judge_points_in_boxes(sampling_input, enlarged_gt_bboxes_3d)
                gt_logits[inlier_mask] = 1.0
                is_valid_loss = inlier_mask.any()

                # density scaling
                is_front, bbox_id = inlier_mask.detach().max(1)  # (N,)
                is_ambiguity = inlier_mask.sum(1) > 1
                if is_ambiguity.any():
                    unique_bbox_id = (point2box_dist[is_ambiguity] / inlier_mask[is_ambiguity]).abs().argmin(1)
                    bbox_id[is_ambiguity] = unique_bbox_id
                # get local density
                instance_id = bbox_id + 1
                instance_id[~is_front] = 0  # instance id of each point, 0 for background
                knn_results = knn_points(p1=sampling_input[None], p2=sampling_input[None], K=density_K)
                grouped_ids = knn_gather(instance_id[None, :, None], knn_results.idx).squeeze()
                same_ins_mask = grouped_ids == instance_id[:, None]
                dist_mask = (knn_results.dists.squeeze(0) <= (density_radii[layer_index - 2] ** 2))
                density = (same_ins_mask & dist_mask).sum(-1, keepdim=True)
                density = density.float()
                density[~is_front] = 1.0
                density[is_front] /= density_K  # density = (number of same instance in kernel) / (kernel size)
                gt_logits /= density

                # get max prob across all objects
                gt_logits = gt_logits.max(1)[0]  # (N, K) -> (N,)
                gt_distribution = torch.log((gt_logits / gt_logits.sum().clamp(min=1e-8)).clamp(min=1e-8))

                # FKL
                loss = focal_kl_div(sampling_distribution, gt_distribution, log_target=True, eps=1e-8)
                if is_valid_loss:
                    as_loss[as_index].append(loss)

                if self.figure_tobe_show_in_tensorboard is None:  # view sampling
                    plt.subplot(len(layer_with_as), 3, subplot_index)
                    subplot_index += 1

                    plt.title('gt distribution')
                    plt.axis('equal')
                    points = sampling_input.cpu().numpy()
                    centers = gt_centers.cpu().numpy()
                    color = (gt_logits / gt_logits.sum()).cpu().numpy() + 1e-10
                    plt.scatter(x=points[:, 0], y=points[:, 1], c=np.log(color), s=1)
                    plt.scatter(x=centers[:, 0], y=centers[:, 1], c='red', s=20)
                    plt.colorbar()

                    plt.subplot(len(layer_with_as), 3, subplot_index)
                    subplot_index += 1
                    plt.title(f'prediction distribution, diversity={loss.item():.3f}')
                    plt.axis('equal')
                    points = sampling_input.cpu().numpy()
                    centers = gt_centers.cpu().numpy()
                    color = current_sa_score.detach().mean(0).cpu().numpy()
                    plt.scatter(x=points[:, 0], y=points[:, 1], c=color, s=1)
                    plt.scatter(x=centers[:, 0], y=centers[:, 1], c='red', s=20)
                    plt.colorbar()

                    plt.subplot(len(layer_with_as), 3, subplot_index)
                    subplot_index += 1
                    plt.title('prediction sampling')
                    plt.axis('equal')
                    prediction_points = sa_xyz[layer_index][batch_index].detach().cpu()
                    plt.scatter(x=prediction_points[:, 0], y=prediction_points[:, 1], c='b', s=1)
                    plt.scatter(x=centers[:, 0], y=centers[:, 1], c='red', s=20)

            if self.figure_tobe_show_in_tensorboard is None:
                plot_buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(plot_buf, format='png')
                plot_buf.seek(0)
                image = Image.open(plot_buf)
                image = ToTensor()(image)
                self.figure_tobe_show_in_tensorboard = image.detach().cpu().clone()
                plt.close()

        loss_dict = {}
        as_loss = [sum(layer) / max(len(layer), 1) for layer in as_loss]
        as_loss = sum(as_loss)
        loss_dict['as_loss'] = as_loss
        return loss_dict

    def detection_loss(self,
                       points: List[torch.Tensor],
                       feats_dict: Dict[str, torch.Tensor],
                       batch_data_samples: List[Det3DDataSample],
                       ret_target: bool = False,
                       **kwargs) -> dict:
        """
        Args:
            points (list[tensor]): Points cloud of multiple samples.
            feats_dict (dict): Predictions from backbone or FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item contains the meta information of each sample
                and corresponding annotations.
            ret_target (bool): Whether return the assigned target. Defaults to False.

        Returns:
            dict:  A dictionary of loss components.
        """
        preds_dict = self(feats_dict)
        batch_gt_instance_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        batch_pts_semantic_mask = []
        batch_pts_instance_mask = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(data_sample.get('ignored_instances', None))
            batch_pts_semantic_mask.append(data_sample.gt_pts_seg.get('pts_semantic_mask', None))
            batch_pts_instance_mask.append(data_sample.gt_pts_seg.get('pts_instance_mask', None))

        loss_inputs = (points, preds_dict, batch_gt_instance_3d)
        losses = self.loss_by_feat(
            *loss_inputs,
            batch_pts_semantic_mask=batch_pts_semantic_mask,
            batch_pts_instance_mask=batch_pts_instance_mask,
            batch_input_metas=batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            ret_target=ret_target,
            runtime=feats_dict['runtime'],
            **kwargs)
        return losses

    def predict_by_feat(self, points: List[torch.Tensor], bbox_preds_dict: dict, batch_input_metas: List[dict],
                        **kwargs) -> List[InstanceData]:
        """Generate bboxes from key points

        Args:
            points (List[torch.Tensor]): Input points of multiple samples.
            bbox_preds_dict (dict): Predictions from vote head.
            batch_input_metas (list[dict]): Each item
                contains the meta information of each sample.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData cantains 3d Bounding boxes and corresponding
            scores and labels.
        """
        feats_dict = kwargs['feats_dict']
        # decode boxes
        sem_scores = F.sigmoid(bbox_preds_dict['obj_scores']).transpose(1, 2)
        obj_scores = sem_scores.max(-1)[0]
        bbox3d = self.bbox_coder.decode(bbox_preds_dict)
        batch_size = bbox3d.shape[0]
        points = torch.stack(points)
        results_list = []
        for b in range(batch_size):
            temp_results = InstanceData()
            bbox_selected, score_selected, labels = \
                self.multiclass_nms_single(
                    obj_scores[b], sem_scores[b], bbox3d[b], points[b, ..., :3], batch_input_metas[b])

            bbox = batch_input_metas[b]['box_type_3d'](
                bbox_selected.clone(),
                box_dim=bbox_selected.shape[-1],
                with_yaw=self.bbox_coder.with_rot)

            temp_results.bboxes_3d = bbox
            temp_results.scores_3d = score_selected
            temp_results.labels_3d = labels
            temp_results.set_metainfo({'key_points': feats_dict['seed_xyz'][b]})
            results_list.append(temp_results)

        return results_list

    def multiclass_nms_single(self, obj_scores: Tensor, sem_scores: Tensor,
                              bbox: Tensor, points: Tensor,
                              input_meta: dict) -> Tuple[Tensor, Tensor, Tensor]:
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): Semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))

        if isinstance(bbox, (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            box_indices = judge_points_in_boxes(points, bbox)
            nonempty_box_mask = box_indices.sum(0) > 0
        else:
            raise NotImplementedError('Unsupported bbox type!')

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        bbox_classes = torch.argmax(sem_scores, -1)

        if not nonempty_box_mask.any():
            import logging
            from mmengine.logging import print_log
            print_log('Found a prediction with empty output', logger='current', level=logging.WARNING)
            selected = nonempty_box_mask
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]
            return bbox_selected, score_selected, labels

        nms_type = self.test_cfg.nms_cfg['type']
        if nms_type == 'nms':
            nms_keep = batched_nms(
                minmax_box3d[nonempty_box_mask][:, [0, 1, 3, 4]],
                obj_scores[nonempty_box_mask], bbox_classes[nonempty_box_mask],
                self.test_cfg.nms_cfg)[1]
        elif nms_type == 'rotate':
            from mmdet3d.structures import xywhr2xyxyr
            from mmdet3d.models.layers import nms_bev
            bboxes_for_nms = xywhr2xyxyr(bbox[nonempty_box_mask].bev)
            scores_for_nms = obj_scores[nonempty_box_mask]
            classes_for_nms = bbox_classes[nonempty_box_mask]
            nms_keep = []
            for class_id in bbox_classes.unique():
                class_select_ids = torch.where(classes_for_nms == class_id)[0]
                bboxes_nms_i = bboxes_for_nms[class_select_ids, :]
                scores_i = scores_for_nms[class_select_ids]
                if len(bboxes_nms_i) == 0:
                    continue
                selected = nms_bev(bboxes_nms_i, scores_i, self.test_cfg.nms_cfg['iou_thr'])
                nms_keep.append(class_select_ids[selected])
            nms_keep = torch.cat(nms_keep, dim=0)
        else:
            raise ValueError

        if nms_keep.shape[0] > self.test_cfg.max_output_num:
            nms_keep = nms_keep[:self.test_cfg.max_output_num]

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores >= self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_keep], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

    def _assign_targets_by_points_inside(self, gt_bboxes_3d: BaseInstance3DBoxes, bbox_preds: dict) -> Tuple:
        """Compute assignment

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes[tensor(num_gt_box)]): Instance of gt bounding boxes.
            bbox_preds(dict): bbox predicted results:
                - seed_xyz: (N, 3) coordinates of key-points
                - center_offset: (N, 3) predicted bottom center offset of bboxes.
                - size: (N, 3) predicted bbox size.
                - dir_class: (N, 12) predicted bbox direction class.
                - dir_res_norm: (N, 12) predicted bbox direction residual.
                - obj_scores: (num_cls, N) predicted objectness classification

        Returns:
            points_mask: tensor[num_points,num_gt_box]
            assignment: tensor[num_points]
        """

        if not isinstance(gt_bboxes_3d, (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            raise NotImplementedError('Unsupported bbox type!')

        '''match cost'''
        gt_center = gt_bboxes_3d.gravity_center  # (K, 3)
        seed_xyz = bbox_preds['seed_xyz']  # (N, 3)
        box_pos_cost = torch.norm(gt_center.unsqueeze(0) - seed_xyz.unsqueeze(1), dim=-1)  # (N, K)
        match_cost = box_pos_cost.clone()

        device = match_cost.device
        assignment = torch.zeros(seed_xyz.shape[0], dtype=torch.int64, device=device)
        points_mask = torch.zeros_like(match_cost, dtype=torch.int64, device=device)
        category = torch.zeros(seed_xyz.shape[0], dtype=torch.int64, device=device)  # -1 = ignore，0 = background，1 = foreground
        match_cost_np = match_cost.cpu().clone().detach().numpy()

        # bipartite matching
        if self.train_cfg.get('use_bipartite_matching', False):
            point_index, box_index = linear_sum_assignment(match_cost_np)
            point_index, box_index = torch.from_numpy(point_index).to(device), torch.from_numpy(box_index).to(device)

            # too far, remove
            pos_distance_thr = self.train_cfg.get('pos_distance_thr', 5.0)
            assign_cost = match_cost[point_index, box_index]
            cost_mask = assign_cost <= pos_distance_thr
            point_index = point_index[cost_mask]
            box_index = box_index[cost_mask]

            # recording match results
            points_mask[point_index, box_index] = 1
            assignment[point_index] = box_index
            category[point_index] = 1
            is_assigned, bbox_nearest_assigned = points_mask.max(0)  # (K,)
            bbox_nearest_assigned[~(is_assigned.bool())] = -1
        else:
            bbox_nearest_assigned = None

        # each point are assigned to nearby bbox
        other_point_ids = torch.nonzero(category != 1).squeeze(1)
        if len(other_point_ids) > 0:
            other_point = seed_xyz[other_point_ids]
            if bbox_preds['is_proposal']:
                enlarged_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(self.train_cfg.expand_dims_length)
            else:
                enlarged_gt_bboxes_3d = gt_bboxes_3d
            if device.type == 'cuda':
                inlier_mask = enlarged_gt_bboxes_3d.points_in_boxes_all(other_point).bool()  # (N, K)
            else:
                inlier_mask = judge_points_in_boxes(other_point, enlarged_gt_bboxes_3d)  # (N, K)
            is_front, bbox_id = inlier_mask.detach().max(1)

            is_ambiguity = inlier_mask.sum(1) > 1
            if is_ambiguity.any():
                min_bbox_id = (box_pos_cost[other_point_ids][is_ambiguity] / inlier_mask[is_ambiguity]).abs().argmin(1)
                bbox_id[is_ambiguity] = min_bbox_id

            category[other_point_ids] = is_front.long()
            assignment[other_point_ids] = bbox_id
            points_mask[other_point_ids, bbox_id] = is_front.long()

            if not bbox_preds['is_proposal']:
                # ignore predictions near the boundary
                extend_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(self.train_cfg.ignore_dims_length)
                if device.type == 'cuda':
                    extend_inlier_mask = extend_gt_bboxes_3d.points_in_boxes_all(other_point).bool()  # (N, K)
                else:
                    extend_inlier_mask = judge_points_in_boxes(other_point, extend_gt_bboxes_3d)  # (N, K)
                in_extend_gt = extend_inlier_mask.any(1)
                is_ignore = is_front ^ in_extend_gt
                category[is_ignore] = -1
        return category, assignment, bbox_nearest_assigned

    def _get_cls_out_channels(self) -> int:
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (1)
        return self.num_classes

    def _get_reg_out_channels(self) -> int:
        """Return the channel number of regression outputs."""
        # Bbox classification and regression
        # (center residual (3), size regression (3)
        # heading class+residual (num_dir_bins*2)),
        return 3 + 3 + self.num_dir_bins * 2

    def _extract_input(self, feat_dict: dict) -> Tuple:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """
        seed_points = feat_dict['sa_xyz'][-1]
        seed_features = feat_dict['sa_features'][-1]
        seed_indices = feat_dict['sa_indices'][-1]

        return seed_points, seed_features, seed_indices


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
