from typing import Dict, List
import torch
from torch import Tensor
import numpy as np

from mmdet3d.registry import TASK_UTILS
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.structures.bbox_3d import rotation_3d_in_axis


def get_centerness_target(seed_xyz, center_targets, size_targets, gt_bboxes_3d, obj_scores=None):
    # Centerness loss target
    center_offset = seed_xyz - center_targets
    canonical_xyz = rotation_3d_in_axis(center_offset.unsqueeze(1), -gt_bboxes_3d.yaw, axis=2).squeeze(1)

    distance_front = torch.clamp(size_targets[..., 0] - canonical_xyz[..., 0], min=0)
    distance_back = torch.clamp(size_targets[..., 0] + canonical_xyz[..., 0], min=0)
    distance_left = torch.clamp(size_targets[..., 1] - canonical_xyz[..., 1], min=0)
    distance_right = torch.clamp(size_targets[..., 1] + canonical_xyz[..., 1], min=0)
    distance_top = torch.clamp(size_targets[..., 2] - canonical_xyz[..., 2], min=0)
    distance_bottom = torch.clamp(size_targets[..., 2] + canonical_xyz[..., 2], min=0)

    centerness_l = torch.min(distance_front, distance_back) / torch.max(distance_front, distance_back).clamp(min=1e-8)
    centerness_w = torch.min(distance_left, distance_right) / torch.max(distance_left, distance_right).clamp(min=1e-8)
    centerness_h = torch.min(distance_bottom, distance_top) / torch.max(distance_bottom, distance_top).clamp(min=1e-8)
    centerness_targets = torch.clamp(centerness_l * centerness_w * centerness_h, min=0)
    centerness_targets = centerness_targets.pow(1 / 3.0)
    centerness_targets = torch.clamp(centerness_targets, min=0, max=1)

    return centerness_targets


@TASK_UTILS.register_module()
class AnchorFreeBaseBBoxCoder:
    """Anchor free BASE bbox coder for 3D boxes.

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        with_rot (bool): Whether the bbox is with rotation.
        with_vel (bool): Whether the bbox is with velocity.
    """

    def __init__(self, num_dir_bins: int, with_rot: bool = True, with_vel: bool = False) -> None:
        self.num_dir_bins = num_dir_bins
        self.with_rot = with_rot
        self.with_vel = with_vel

    def split_pred(self, cls_preds: Tensor, reg_preds: Tensor, seed_xyz: Tensor) -> Dict[str, Tensor]:
        """Split predicted features to specific parts.

        Args:
            cls_preds (Tensor): (B, num_cls, N) Class predicted features to split.
            reg_preds (Tensor): (B, 6 + num_bin * 2, N) Regression predicted features to split.
            seed_xyz (Tensor): (B, N, 3) Coordinates of points.

        Returns:
            dict[str, Tensor]: Split results.
        """
        start, end = 0, 0
        reg_preds_trans = reg_preds.transpose(1, 2)
        results = dict(seed_xyz=seed_xyz)

        # decode classification
        results['obj_scores'] = cls_preds

        # decode center_offset
        end += 3
        results['center_offset'] = reg_preds_trans[..., start:end]
        start = end

        # decode size
        end += 3
        results['size'] = reg_preds_trans[..., start:end]
        start = end

        # decode direction
        end += self.num_dir_bins
        results['dir_class'] = reg_preds_trans[..., start:end]
        start = end

        end += self.num_dir_bins
        results['dir_res_norm'] = reg_preds_trans[..., start:end]
        start = end

        # decode velocity
        if self.with_vel:
            end += 2
            results['velocity'] = reg_preds_trans[..., start:end]
            start = end

        return results

    def angle2class(self, angle: Tensor) -> tuple:
        """Convert continuous angle to a discrete class and a residual.

        Convert continuous angle to a discrete class and a small
        regression number from class center angle to current angle.

        Args:
            angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi),
                class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).

        Returns:
            tuple: Encoded discrete class and residual.
        """
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        angle_cls = shifted_angle // angle_per_class
        angle_res = shifted_angle - (
                angle_cls * angle_per_class + angle_per_class / 2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls: Tensor, angle_res: Tensor, limit_period: bool = True) -> Tensor:
        """Inverse function to angle2class.

        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].

        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle


@TASK_UTILS.register_module()
class AnchorFreeAbBBoxCoder(AnchorFreeBaseBBoxCoder):
    """Anchor free ABSOLUTE bbox coder for 3D boxes.

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        with_rot (bool): Whether the bbox is with rotation.
        with_vel (bool): Whether the bbox is with velocity.
    """

    def __init__(self, num_dir_bins: int, with_rot: bool = True, with_vel: bool = False) -> None:
        super().__init__(num_dir_bins, with_rot, with_vel)

    def encode(self, gt_bboxes_3d: BaseInstance3DBoxes, gt_labels_3d: Tensor, seed_xyz: Tensor) -> tuple:
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): (K, 7) Ground truth bboxes with shape.
            gt_labels_3d (Tensor): (K,) Ground truth classes.
            seed_xyz (Tensor): (N, 3) key-points for predicting objects

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d.gravity_center
        center_offset_target = center_target - seed_xyz

        # generate bbox size target
        size_target = gt_bboxes_3d.dims / 2

        # generate dir target
        box_num = gt_labels_3d.shape[0]
        if self.with_rot:
            dir_class_target, dir_res_target = self.angle2class(gt_bboxes_3d.yaw)
            dir_res_target /= (2 * np.pi / self.num_dir_bins)
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(box_num)

        # generate corner target
        corner3d_target = gt_bboxes_3d.corners

        # generate classification target @ centerness target
        cls_target = gt_labels_3d
        centerness_target = \
            get_centerness_target(seed_xyz.detach(), center_target, size_target, gt_bboxes_3d, obj_scores=None)

        if self.with_vel:
            velocity_target = gt_bboxes_3d.tensor[:, -2:]
            return center_offset_target, size_target, dir_class_target, dir_res_target, corner3d_target, \
                   cls_target, centerness_target, velocity_target
        else:
            return center_offset_target, size_target, dir_class_target, dir_res_target, corner3d_target, \
                   cls_target, centerness_target

    def decode(self, bbox_out: dict) -> Tensor:
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - seed_xyz: coordinates of key-points
                - center_offset: predicted bottom center offset of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res_norm: predicted bbox direction residual.
                - size: predicted bbox size.

        Returns:
            torch.Tensor: Decoded bbox3d with shape (B, N, 7).
        """
        seed_xyz = bbox_out['seed_xyz']
        B, N = seed_xyz.shape[:2]

        # decode bbox center
        center = bbox_out['seed_xyz'].detach() + bbox_out['center_offset']

        # decode bbox size
        bbox_size = torch.clamp(bbox_out['size'] * 2, min=0.1)

        # decode heading angle
        if self.with_rot:
            dir_res = bbox_out['dir_res_norm'] * (2 * np.pi / self.num_dir_bins)
            dir_class = torch.argmax(bbox_out['dir_class'], -1)
            dir_res = torch.gather(dir_res, 2, dir_class.unsqueeze(-1))
            dir_res.squeeze_(2)
            dir_angle = self.class2angle(dir_class, dir_res).reshape(B, N, 1)
        else:
            dir_angle = center.new_zeros(B, N, 1)

        if self.with_vel:
            velocity = bbox_out['velocity']
            bbox3d = torch.cat([center, bbox_size, dir_angle, velocity], dim=-1)
        else:
            bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d


@TASK_UTILS.register_module()
class AnchorFreeReBBoxCoder(AnchorFreeBaseBBoxCoder):
    """Anchor free RELATIVE bbox coder for 3D boxes.

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        with_rot (bool): Whether the bbox is with rotation.
        with_vel (bool): Whether the bbox is with velocity.
        mean_size (List[List[float]]): Mean size of each class
    """

    def __init__(self, num_dir_bins: int, with_rot: bool = True, with_vel: bool = False,
                 mean_size: List[List[float]] = None) -> None:
        super().__init__(num_dir_bins, with_rot, with_vel)
        self.mean_size: Tensor = torch.tensor(mean_size, dtype=torch.float)  # (num_cls, 3)
        assert self.mean_size.min() > 0
        diagonal = torch.sqrt(self.mean_size[:, 0] ** 2 + self.mean_size[:, 1] ** 2)  # (num_cls, )
        self.offset_regulation = torch.stack([diagonal, diagonal, self.mean_size[:, 2]], dim=1)  # (num_cls, 3)

    def encode(self, gt_bboxes_3d: BaseInstance3DBoxes, gt_labels_3d: Tensor, seed_xyz: Tensor) -> tuple:
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): (K, 7) Ground truth bboxes with shape.
            gt_labels_3d (Tensor): (K,) Ground truth classes.
            seed_xyz (Tensor): (N, 3) key-points for predicting objects

        Returns:
            tuple: Targets of center, size and direction.
        """
        device = seed_xyz.device
        # prepare relative value
        self.mean_size = self.mean_size.to(device)
        self.offset_regulation = self.offset_regulation.to(device)
        mean_size = self.mean_size[gt_labels_3d]
        offset_regulation = self.offset_regulation[gt_labels_3d]

        # generate center target
        center_target = gt_bboxes_3d.gravity_center
        center_offset_target = center_target - seed_xyz
        center_offset_norm_target = center_offset_target / offset_regulation

        # generate bbox size target
        size_target = gt_bboxes_3d.dims
        size_norm_target = torch.log(size_target / mean_size)

        # generate dir target
        box_num = gt_labels_3d.shape[0]
        if self.with_rot:
            dir_class_target, dir_res_target = self.angle2class(gt_bboxes_3d.yaw)
            dir_res_target /= (2 * np.pi / self.num_dir_bins)
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(box_num)

        # generate corner target
        corner3d_target = gt_bboxes_3d.corners

        # generate classification target @ centerness target
        cls_target = gt_labels_3d
        centerness_target = \
            get_centerness_target(seed_xyz.detach(), center_target, size_target, gt_bboxes_3d, obj_scores=None)

        if self.with_vel:
            velocity_target = gt_bboxes_3d.tensor[:, -2:]
            return center_offset_norm_target, size_norm_target, dir_class_target, dir_res_target, corner3d_target, \
                   cls_target, centerness_target, velocity_target
        else:
            return center_offset_norm_target, size_norm_target, dir_class_target, dir_res_target, corner3d_target, \
                   cls_target, centerness_target

    def decode(self, bbox_out: dict) -> Tensor:
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - seed_xyz: coordinates of key-points
                - center_offset: predicted bottom center offset of bboxes.
                - size: predicted bbox size.
                - dir_class: predicted bbox direction class.
                - dir_res_norm: predicted bbox direction residual.
                - obj_scores: predicted objectness classification

        Returns:
            torch.Tensor: Decoded bbox3d with shape (B, N, 7).
        """
        seed_xyz = bbox_out['seed_xyz'].detach()
        device = seed_xyz.device
        B, N = seed_xyz.shape[:2]

        # prepare relative value
        pred_labels_3d = bbox_out['obj_scores'].argmax(1)  # (B, N)
        batch_select_ids = torch.arange(B, device=device).unsqueeze(1).repeat(1, N)  # (B, N)
        self.mean_size = self.mean_size.to(device)
        self.offset_regulation = self.offset_regulation.to(device)
        mean_size = self.mean_size.unsqueeze(0).repeat(B, 1, 1)[batch_select_ids, pred_labels_3d]
        offset_regulation = self.offset_regulation.unsqueeze(0).repeat(B, 1, 1)[batch_select_ids, pred_labels_3d]

        # decode bbox center
        center = bbox_out['center_offset'] * offset_regulation + seed_xyz

        # decode bbox size
        bbox_size = torch.clamp(torch.exp(bbox_out['size']) * mean_size, min=0.1, max=100)

        # decode heading angle
        if self.with_rot:
            dir_res = bbox_out['dir_res_norm'] * (2 * np.pi / self.num_dir_bins)
            dir_class = torch.argmax(bbox_out['dir_class'], -1)
            dir_res = torch.gather(dir_res, 2, dir_class.unsqueeze(-1))
            dir_res.squeeze_(2)
            dir_angle = self.class2angle(dir_class, dir_res).reshape(B, N, 1)
        else:
            dir_angle = center.new_zeros(B, N, 1)

        if self.with_vel:
            velocity = bbox_out['velocity']
            bbox3d = torch.cat([center, bbox_size, dir_angle, velocity], dim=-1)
        else:
            bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d


@TASK_UTILS.register_module()
class AnchorFreeSimpleBBoxCoder:
    """Anchor free BASE bbox coder for 3D boxes.

    Args:
        with_rot (bool): Whether the bbox is with rotation.
        with_vel (bool): Whether the bbox is with velocity.
    """

    def __init__(self, with_rot: bool = True, with_vel: bool = False) -> None:
        self.with_rot = with_rot
        self.with_vel = with_vel

    def encode(self, gt_bboxes_3d: BaseInstance3DBoxes, gt_labels_3d: Tensor, seed_xyz: Tensor,
               return_centerness: bool = True) -> tuple:
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): (K, 7) Ground truth bboxes with shape.
            gt_labels_3d (Tensor): (K,) Ground truth classes.
            seed_xyz (Tensor): (N, 3) key-points for predicting objects

        Returns:
            tuple: Targets of center, size and direction.
        """
        ret_list = []

        # generate center target
        center_target = gt_bboxes_3d.gravity_center
        center_offset_target = center_target - seed_xyz

        # generate bbox size target
        size_target = gt_bboxes_3d.dims / 2

        # generate dir target
        dir_res_target = self.encode_angle(gt_bboxes_3d.yaw)

        # generate classification target @ centerness target
        cls_target = gt_labels_3d
        if return_centerness:
            centerness_target = \
                get_centerness_target(seed_xyz.detach(), center_target, size_target, gt_bboxes_3d, obj_scores=None)
            ret_list = [cls_target, centerness_target, center_offset_target, size_target, dir_res_target]
        else:
            ret_list = [cls_target, center_offset_target, size_target, dir_res_target]

        if self.with_vel:
            velocity_target = gt_bboxes_3d.tensor[:, -2:]
            ret_list.append(velocity_target)

        return tuple(ret_list)

    def decode(self, bbox_out: dict) -> Tensor:
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - seed_xyz: coordinates of key-points
                - center_offset: predicted bottom center offset of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res_norm: predicted bbox direction residual.
                - size: predicted bbox size.

        Returns:
            torch.Tensor: Decoded bbox3d with shape (B, N, 7).
        """
        seed_xyz = bbox_out['seed_xyz']

        # decode bbox center
        center = seed_xyz.detach() + bbox_out['center_offset']

        # decode bbox size
        bbox_size = torch.clamp(bbox_out['size'] * 2, min=0.1)

        # decode heading angle
        angle_norm = bbox_out['dir_res_norm']
        dir_angle = self.decode_angle(angle_norm)

        if self.with_vel:
            velocity = bbox_out['velocity']
            bbox3d = torch.cat([center, bbox_size, dir_angle, velocity], dim=-1)
        else:
            bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d

    def split_pred(self, cls_preds: Tensor, reg_preds: Tensor, seed_xyz: Tensor) -> Dict[str, Tensor]:
        """Split predicted features to specific parts.

        Args:
            cls_preds (Tensor): (N, num_cls) Class predicted features to split.
            reg_preds (Tensor): (N, 10) Regression predicted features to split.
            seed_xyz (Tensor): (N, 3) Coordinates of points.

        Returns:
            dict[str, Tensor]: Split results.
        """
        start, end = 0, 0
        reg_preds_trans = reg_preds
        results = dict(seed_xyz=seed_xyz)

        # decode classification
        results['obj_scores'] = cls_preds

        # decode center_offset
        end += 3
        results['center_offset'] = reg_preds_trans[..., start:end]
        start = end

        # decode size
        end += 3
        results['size'] = reg_preds_trans[..., start:end]
        start = end

        # decode direction
        end += 2
        results['dir_res_norm'] = reg_preds_trans[..., start:end]
        start = end

        # decode velocity
        if self.with_vel:
            end += 2
            results['velocity'] = reg_preds_trans[..., start:end]
            start = end

        return results

    def encode_angle(self, angle: Tensor) -> Tensor:
        """Convert angle to sin & cos.

        Args:
            angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi).

        Returns:
            tuple: Encoded sin & cos value.
        """
        sina = torch.sin(angle)
        cosa = torch.cos(angle)
        angle_norm = torch.stack([sina, cosa], dim=-1)  # (N, 2)
        return angle_norm

    def decode_angle(self, angle_norm: Tensor) -> Tensor:
        """Inverse function to angle2class.

        Args:
            angle_norm (torch.Tensor): sin & cos (N, 2)

        Returns:
            torch.Tensor: Decoded angle.
        """
        sina = angle_norm[:, :1]
        cosa = angle_norm[:, 1:2]
        angle = torch.atan2(sina, cosa)
        return angle


@TASK_UTILS.register_module()
class AnchorFreeNusBBoxCoder(AnchorFreeSimpleBBoxCoder):
    """Same bbox coder of VoxelNext for 3D boxes.

    Args:
        with_rot (bool): Whether the bbox is with rotation.
        with_vel (bool): Whether the bbox is with velocity.
    """

    def __init__(self, with_rot: bool = True, with_vel: bool = False, absolute_height: bool = False) -> None:
        super().__init__(with_rot, with_vel)
        self.absolute_height = absolute_height

    def encode(self, gt_bboxes_3d: BaseInstance3DBoxes, gt_labels_3d: Tensor, seed_xyz: Tensor,
               return_centerness: bool = True) -> tuple:
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): (K, 7) Ground truth bboxes with shape.
            gt_labels_3d (Tensor): (K,) Ground truth classes.
            seed_xyz (Tensor): (N, 3) key-points for predicting objects

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d.gravity_center
        center_offset_target = center_target - seed_xyz
        if self.absolute_height:
            center_offset_target[:, 2] = center_target[:, 2]

        # generate bbox size target
        size_target = gt_bboxes_3d.dims.log()

        # generate dir target
        dir_res_target = self.encode_angle(gt_bboxes_3d.yaw)

        # generate classification target @ centerness target
        cls_target = gt_labels_3d
        if return_centerness:
            centerness_target = \
                get_centerness_target(seed_xyz.detach(), center_target, size_target, gt_bboxes_3d, obj_scores=None)
            ret_list = [cls_target, centerness_target, center_offset_target, size_target, dir_res_target]
        else:
            ret_list = [cls_target, center_offset_target, size_target, dir_res_target]

        if self.with_vel:
            velocity_target = gt_bboxes_3d.tensor[:, -2:]
            ret_list.append(velocity_target)

        return tuple(ret_list)

    def decode(self, bbox_out: dict) -> Tensor:
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - seed_xyz: coordinates of key-points
                - center_offset: predicted bottom center offset of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res_norm: predicted bbox direction residual.
                - size: predicted bbox size.

        Returns:
            torch.Tensor: Decoded bbox3d with shape (B, N, 7).
        """
        seed_xyz = bbox_out['seed_xyz']

        # decode bbox center
        center = seed_xyz.detach() + bbox_out['center_offset']
        if self.absolute_height:
            center[:, 2] = bbox_out['center_offset'][:, 2]

        # decode bbox size
        bbox_size = bbox_out['size'].exp()

        # decode heading angle
        angle_norm = bbox_out['dir_res_norm']
        dir_angle = self.decode_angle(angle_norm)

        if self.with_vel:
            velocity = bbox_out['velocity']
            bbox3d = torch.cat([center, bbox_size, dir_angle, velocity], dim=-1)
        else:
            bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d
