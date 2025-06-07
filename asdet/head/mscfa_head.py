from typing import List, Optional, Tuple, Union, Dict
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.utils import multi_apply
from mmdet3d.models.dense_heads.base_conv_bbox_head import BaseConvBboxHead
from mmdet3d.registry import MODELS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample
from asdet.backbone.modules import SetAbstractionMSG
from asdet.head.base_head import BaseASHead


@MODELS.register_module()
class ASMSCFAHead(BaseASHead):
    """
    VoteHead with multi-scale center feature aggregation

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
                 decoder_layer_cfg: dict,
                 num_classes: int,
                 bbox_coder: Union[ConfigDict, dict],
                 proposal_coder: Union[ConfigDict, dict],
                 vote_limit: Optional[tuple] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 pred_layer_cfg: Optional[dict] = None,
                 objectness_loss: Optional[dict] = None,
                 center_loss: Optional[dict] = None,
                 dir_class_loss: Optional[dict] = None,
                 dir_res_loss: Optional[dict] = None,
                 size_res_loss: Optional[dict] = None,
                 corner_loss: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super(ASMSCFAHead, self).__init__(
            num_classes=num_classes,
            bbox_coder=bbox_coder,
            proposal_coder=proposal_coder,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            objectness_loss=objectness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_res_loss,
            corner_loss=corner_loss,
            init_cfg=init_cfg)

        if vote_limit is None:
            self.vote_limit = (math.inf, math.inf, math.inf)
        else:
            self.vote_limit = vote_limit
            assert len(self.vote_limit) == 3

        # proposals generator
        self.proposal_head = BaseConvBboxHead(
            **pred_layer_cfg,
            num_cls_out_channels=1,
            num_reg_out_channels=self._get_reg_out_channels())
        self.proposal_head.conv_cls.bias.data.fill_(-2.19)

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
        decoder_pred_layer_cfg = pred_layer_cfg.copy()
        decoder_pred_layer_cfg.pop('in_channels')
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

        self.prediction_head = BaseConvBboxHead(
                **decoder_pred_layer_cfg,
                in_channels=decoder_layer_out_channels,
                num_cls_out_channels=self._get_cls_out_channels(),
                num_reg_out_channels=self._get_reg_out_channels())
        self.prediction_head.conv_cls.bias.data.fill_(-2.19)

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

        cls_predictions, reg_predictions = self.proposal_head(seed_fea)
        proposals = self.proposal_coder.split_pred(cls_predictions, reg_predictions, seed_xyz)
        for axis_id in range(3):
            limitation = self.vote_limit[axis_id]
            proposals['center_offset'][..., axis_id].clamp_(max=limitation, min=-limitation)
        proposal_bboxes = self.proposal_coder.decode(proposals)

        predictions = self.decoding(feat_dict, proposal_bboxes)

        results = dict(seed_xyz=seed_xyz, predictions_list=[proposals, predictions])
        return results

    def decoding(self, feat_dict: Dict[str, List[Tensor]], proposals: Tensor) -> List[Tensor]:
        len_fp_out = len(feat_dict['fp_features'])
        hierarchical_features = feat_dict['sa_features'][::-1]
        hierarchical_xyz = feat_dict['sa_xyz'][::-1]
        hierarchical_features[1: 1 + len_fp_out] = feat_dict['fp_features']
        hierarchical_xyz[1: 1 + len_fp_out] = feat_dict['fp_xyz']
        hierarchical_features.reverse()
        hierarchical_xyz.reverse()
        assert len(hierarchical_xyz) >= len(self.src_idx)

        seed_xyz = self._get_object_query_pos(proposals)  # (B, N, k)
        init_fea = hierarchical_features[-1]
        obj_fea_list = []
        for i, decoder_layer in enumerate(self.decoder_layers):
            src = self.src_idx[i]
            xyz, features = hierarchical_xyz[src], hierarchical_features[src]

            object_xyz, object_fea = \
                decoder_layer(points_xyz=xyz, points_fea=features, target_xyz=seed_xyz, target_fea=init_fea)
            obj_fea_list.append(object_fea)

        obj_fea = torch.cat(obj_fea_list, dim=1)  # (B, C1+C2+...+Cn, N)
        cls_predictions, reg_predictions = self.prediction_head(obj_fea)
        predictions = self.bbox_coder.split_pred(cls_predictions, reg_predictions, seed_xyz=seed_xyz)
        return predictions

    def _get_object_query_pos(self, bboxes: Tensor) -> Tuple[Tensor, Tensor]:
        return bboxes[..., :3]

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
            dict: Detection Losses.
        """
        proposals, predictions = bbox_preds_dict['predictions_list']
        proposals['is_proposal'] = True
        predictions['is_proposal'] = False
        B, N, _ = proposals['seed_xyz'].shape
        device = proposals['seed_xyz'].device

        '''vote proposal'''
        targets = self.get_targets(points, proposals, batch_gt_instances_3d, None, None)
        center_targets, size_targets, dir_class_targets, dir_res_targets, corner3d_targets, \
        cls_targets, centerness_targets, box_loss_weights, heading_res_loss_weight, \
        objectness_targets, objectness_weights, assignment, bbox_targets = targets

        # calculate center loss, only for positive
        vote_loss = []
        for i in cls_targets.unique():
            cls_mask = (cls_targets == i) & (box_loss_weights > 0)
            vote_loss.append(F.smooth_l1_loss(proposals['center_offset'][cls_mask], center_targets[cls_mask]))
        vote_loss = torch.stack(vote_loss).mean()

        '''precise object prediction'''
        targets = self.get_targets(points, predictions, batch_gt_instances_3d, None, None)
        center_targets, size_targets, dir_class_targets, dir_res_targets, corner3d_targets, \
        cls_targets, centerness_targets, box_loss_weights, heading_res_loss_weight, \
        objectness_targets, objectness_weights, assignment, bbox_targets = targets
        pos_mask = objectness_targets.bool()

        # calculate centerness loss
        cls_one_hot = torch.zeros_like(predictions['obj_scores'].transpose(1, 2)).scatter(-1, cls_targets.unsqueeze(-1), 1)
        centerness_targets = cls_one_hot * centerness_targets.unsqueeze(-1)
        centerness_loss = self.loss_objectness(
            predictions['obj_scores'].transpose(2, 1),
            centerness_targets,
            weight=objectness_weights)

        # bbox regression, use gt direction cls
        one_hot_dir_class_targets = dir_class_targets.new_zeros(predictions['dir_class'].shape)
        one_hot_dir_class_targets.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        pred_bbox3d = self.bbox_coder.decode(
            dict(
                seed_xyz=predictions['seed_xyz'],
                center_offset=predictions['center_offset'],
                size=predictions['size'],
                dir_res_norm=predictions['dir_res_norm'],
                dir_class=one_hot_dir_class_targets,
                obj_scores=cls_one_hot.transpose(2, 1)))
        pred_bbox3d = pred_bbox3d.reshape(-1, pred_bbox3d.shape[-1])
        pred_bbox3d = batch_input_metas[0]['box_type_3d'](
            pred_bbox3d.clone(),
            box_dim=pred_bbox3d.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        bbox_preds = pred_bbox3d.tensor.clone().reshape(B, N, -1)
        bbox_res_preds = torch.cat([predictions['center_offset'], predictions['size'], bbox_preds[..., -1:]], dim=-1)
        bbox_res_targets = torch.cat([center_targets, size_targets, bbox_targets[..., -1:]], dim=-1)
        u, rdiou = self.get_rdiou(bbox_res_preds, bbox_res_targets)
        rdiou_loss_n = rdiou - u
        rdiou_loss_n = torch.clamp(rdiou_loss_n, min=-1.0, max=1.0)
        rdiou_loss_m = 1 - rdiou_loss_n
        rdiou_loss = rdiou_loss_m[pos_mask].mean()
        box_loss = rdiou_loss * 2.0  # loss weight following RDIOU

        # calculate direction class loss, only for positive
        dir_class_loss = self.loss_dir_class(
            predictions['dir_class'][pos_mask],
            dir_class_targets[pos_mask],  # bs, num_object
            weight=box_loss_weights[pos_mask])

        # calculate direction residual loss, only for positive
        dir_res_loss = self.loss_dir_res(
            predictions['dir_res_norm'][pos_mask],  # bs, num_object, fan_bin
            dir_res_targets.unsqueeze(-1).repeat(1, 1, self.num_dir_bins)[pos_mask],  # bs, num_object, fan_bin
            weight=heading_res_loss_weight[pos_mask])

        losses = dict(
            centerness_loss=centerness_loss,
            vote_loss=vote_loss,
            box_loss=box_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
        )
        return losses

    def get_rdiou(self, bboxes1, bboxes2):
        x1u, y1u, z1u = bboxes1[:, :, 0], bboxes1[:, :, 1], bboxes1[:, :, 2]
        l1, w1, h1 = torch.exp(bboxes1[:, :, 3]), torch.exp(bboxes1[:, :, 4]), torch.exp(bboxes1[:, :, 5])
        t1 = torch.sin(bboxes1[:, :, 6]) * torch.cos(bboxes2[:, :, 6])
        x2u, y2u, z2u = bboxes2[:, :, 0], bboxes2[:, :, 1], bboxes2[:, :, 2]
        l2, w2, h2 = torch.exp(bboxes2[:, :, 3]), torch.exp(bboxes2[:, :, 4]), torch.exp(bboxes2[:, :, 5])
        t2 = torch.sin(bboxes2[:, :, 6]) * torch.cos(bboxes1[:, :, 6])

        # we emperically scale the y/z to make their predictions more sensitive.
        x1 = x1u
        y1 = y1u * 2
        z1 = z1u * 2
        x2 = x2u
        y2 = y2u * 2
        z2 = z2u * 2

        # clamp is necessray to aviod inf.
        l1, w1, h1 = torch.clamp(l1, max=10, min=0.1), torch.clamp(w1, max=10, min=0.1), torch.clamp(h1, max=10, min=0.1)
        j1, j2 = torch.ones_like(h2), torch.ones_like(h2)

        volume_1 = l1 * w1 * h1 * j1
        volume_2 = l2 * w2 * h2 * j2

        inter_l = torch.max(x1 - l1 / 2, x2 - l2 / 2)
        inter_r = torch.min(x1 + l1 / 2, x2 + l2 / 2)
        inter_t = torch.max(y1 - w1 / 2, y2 - w2 / 2)
        inter_b = torch.min(y1 + w1 / 2, y2 + w2 / 2)
        inter_u = torch.max(z1 - h1 / 2, z2 - h2 / 2)
        inter_d = torch.min(z1 + h1 / 2, z2 + h2 / 2)
        inter_m = torch.max(t1 - j1 / 2, t2 - j2 / 2)
        inter_n = torch.min(t1 + j1 / 2, t2 + j2 / 2)

        inter_volume = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0) \
                       * torch.clamp((inter_d - inter_u), min=0) * torch.clamp((inter_n - inter_m), min=0)

        c_l = torch.min(x1 - l1 / 2, x2 - l2 / 2)
        c_r = torch.max(x1 + l1 / 2, x2 + l2 / 2)
        c_t = torch.min(y1 - w1 / 2, y2 - w2 / 2)
        c_b = torch.max(y1 + w1 / 2, y2 + w2 / 2)
        c_u = torch.min(z1 - h1 / 2, z2 - h2 / 2)
        c_d = torch.max(z1 + h1 / 2, z2 + h2 / 2)
        c_m = torch.min(t1 - j1 / 2, t2 - j2 / 2)
        c_n = torch.max(t1 + j1 / 2, t2 + j2 / 2)

        inter_diag = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2 + (t2 - t1) ** 2
        c_diag = (torch.clamp((c_r - c_l), min=0) ** 2
                  + torch.clamp((c_b - c_t), min=0) ** 2
                  + torch.clamp((c_d - c_u), min=0) ** 2
                  + torch.clamp((c_n - c_m), min=0) ** 2)

        union = volume_1 + volume_2 - inter_volume
        u = inter_diag / c_diag
        rdiou = inter_volume / union
        return u, rdiou

    def get_targets(
            self,
            points: List[Tensor],
            bbox_preds_dict: dict = None,
            batch_gt_instances_3d: List[InstanceData] = None,
            batch_pts_semantic_mask: List[torch.Tensor] = None,
            batch_pts_instance_mask: List[torch.Tensor] = None,
    ) -> Tuple[Tensor]:
        """Generate targets of detection head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            bbox_preds_dict (dict): Bounding box predictions of head.  Defaults to None.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of gt_instances.
                It usually includes ``bboxes`` and ``labels`` attributes.  Defaults to None.
            batch_pts_semantic_mask (list[tensor]): Semantic gt mask for point clouds.  Defaults to None.
            batch_pts_instance_mask (list[tensor]): Instance gt mask for point clouds.  Defaults to None.

        Returns:
            tuple[torch.Tensor]: Targets of detection head.
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

        seed_xyz, center_offset, size, dir_class, dir_res_norm, obj_scores = ([] for _ in range(6))
        for i in range(len(batch_gt_labels_3d)):
            seed_xyz.append(bbox_preds_dict['seed_xyz'][i])
            center_offset.append(bbox_preds_dict['center_offset'][i])
            size.append(bbox_preds_dict['size'][i])
            dir_class.append(bbox_preds_dict['dir_class'][i])
            dir_res_norm.append(bbox_preds_dict['dir_res_norm'][i])
            obj_scores.append(bbox_preds_dict['obj_scores'][i])
        is_proposal = bbox_preds_dict.get('is_proposal', False)

        (center_targets, size_targets, dir_class_targets, dir_res_targets, corner3d_targets,
         cls_targets, centerness_targets, positive_mask, negative_mask, assignment, bbox_targets) = multi_apply(
            self.get_targets_single,
            points, batch_gt_bboxes_3d, batch_gt_labels_3d, batch_pts_semantic_mask, batch_pts_instance_mask,
            seed_xyz, obj_scores, center_offset, size, dir_class, dir_res_norm, is_proposal=is_proposal)

        center_targets = torch.stack(center_targets)
        size_targets = torch.stack(size_targets)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        corner3d_targets = torch.stack(corner3d_targets)
        cls_targets = torch.stack(cls_targets)
        centerness_targets = torch.stack(centerness_targets)
        positive_mask = torch.stack(positive_mask)
        negative_mask = torch.stack(negative_mask)
        assignment = torch.stack(assignment)
        objectness_targets = positive_mask.clone()
        bbox_targets = torch.stack(bbox_targets)

        box_loss_weights = positive_mask / (positive_mask.sum() + 1e-6)

        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = dir_class_targets.new_zeros((batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        heading_res_loss_weight = heading_label_one_hot * box_loss_weights.unsqueeze(-1)

        objectness_weights = (positive_mask + negative_mask).float()
        pos_normalizer = objectness_weights.sum()
        objectness_weights /= torch.clamp(pos_normalizer, min=1.0)
        objectness_weights = objectness_weights.unsqueeze(-1)

        return center_targets, size_targets, dir_class_targets, dir_res_targets, corner3d_targets, \
               cls_targets, centerness_targets, box_loss_weights, heading_res_loss_weight, \
               objectness_targets, objectness_weights, assignment, bbox_targets

    def get_targets_single(self,
                           points: Tensor,
                           gt_bboxes_3d: BaseInstance3DBoxes,
                           gt_labels_3d: Tensor,
                           pts_semantic_mask: Optional[Tensor] = None,
                           pts_instance_mask: Optional[Tensor] = None,
                           seed_xyz: Optional[Tensor] = None,
                           obj_scores: Optional[Tensor] = None,
                           center_offset: Optional[Tensor] = None,
                           size: Optional[Tensor] = None,
                           dir_class: Optional[Tensor] = None,
                           dir_res_norm: Optional[Tensor] = None,
                           **kwargs):
        """Generate targets of AS head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (torch.Tensor): Point-wise semantic label of each batch.
            pts_instance_mask (torch.Tensor): Point-wise instance label of each batch.
            seed_xyz (torch.Tensor): key points from backbone.
            obj_scores (torch.Tensor): object scores of each batch.
            center_offset (torch.Tensor): box centers of each batch.
            size (torch.Tensor): box size of each batch.
            dir_class (torch.Tensor): box direction class of each batch.
            dir_res_norm (torch.Tensor): normed box direction residual of each batch.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        # Generate fake GT for empty scene
        if valid_gt.sum() == 0:
            raise RuntimeError('Found a frame with no ground truth')

        bbox_preds = {'seed_xyz': seed_xyz, 'obj_scores': obj_scores, 'center_offset': center_offset, 'size': size,
                      'dir_class': dir_class, 'dir_res_norm': dir_res_norm, 'is_proposal': kwargs['is_proposal']}
        category, assignment, bbox_assigned = self._assign_targets_by_points_inside(gt_bboxes_3d, bbox_preds)
        positive_mask = category > 0
        negative_mask = category == 0

        bbox_targets = gt_bboxes_3d[assignment].tensor.clone()
        if kwargs['is_proposal']:
            center_targets, size_targets, dir_class_targets, dir_res_targets, corner3d_targets, \
            cls_targets, centerness_targets = \
                self.proposal_coder.encode(gt_bboxes_3d[assignment], gt_labels_3d[assignment], seed_xyz)
            centerness_targets *= positive_mask
        else:
            center_targets, size_targets, dir_class_targets, dir_res_targets, corner3d_targets, \
            cls_targets, centerness_targets = \
                self.bbox_coder.encode(gt_bboxes_3d[assignment], gt_labels_3d[assignment], seed_xyz)
            centerness_targets *= positive_mask

        if self.train_cfg.get('use_bipartite_matching', False):
            best_pos = bbox_assigned[bbox_assigned != -1]
            centerness_targets[best_pos] = 1.0
        return (center_targets, size_targets, dir_class_targets, dir_res_targets, corner3d_targets,
                cls_targets, centerness_targets, positive_mask, negative_mask, assignment, bbox_targets)

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

        seed_xyz = preds_dict['seed_xyz']
        proposals, predictions = preds_dict['predictions_list']
        feats_dict['seed_xyz'] = seed_xyz

        batch_size = len(batch_data_samples)
        batch_input_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            batch_input_metas.append(metainfo)

        results_list = self.predict_by_feat(points, predictions, batch_input_metas,
                                            use_nms=use_nms, feats_dict=feats_dict, **kwargs)
        return results_list
