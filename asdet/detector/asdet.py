from typing import Dict, List, Optional, Union, Tuple, Literal
import torch
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.structures.det3d_data_sample import ForwardResults, OptSampleList, SampleList


@MODELS.register_module()
class ASDet(SingleStage3DDetector):
    """ASDet Pipeline.

    Args:
        point_cloud_range (tuple[float]): Range of point cloud xyz.
        voxel_size (tuple[float]): Initial voxel size.
        backbone (dict): Config dict of detector's backbone.
        neck (dict): Config dict of detector's neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        init_cfg (dict, optional): the config to control the
           initialization. Default to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
    """
    VOXEL = 1
    POINT = 2
    MODE = dict(voxel=VOXEL, point=POINT)

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 mode: Literal['point', 'voxel'] = 'point',
                 voxel_size: Tuple[float] = None,
                 point_cloud_range: Tuple[float] = None,
                 **kwargs):
        self.mode = self.MODE[mode]
        if self.mode == self.VOXEL:
            assert voxel_size is not None and point_cloud_range is not None
            backbone['voxel_size'] = voxel_size
            bbox_head['voxel_size'] = voxel_size
            if neck is not None:
                neck['point_cloud_range'] = point_cloud_range
                neck['voxel_size'] = voxel_size
            self.point_cloud_range = point_cloud_range
            self.voxel_size = torch.tensor(voxel_size, dtype=torch.float).reshape(1, 3)  # (1, 3)
        super(ASDet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs)

    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: OptSampleList = None,
                *args,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            if isinstance(inputs, dict) and isinstance(data_samples, dict):
                true_inputs = [inputs['inputs'], data_samples['inputs']]
                true_data_samples = [inputs['data_samples'], data_samples['data_samples']]
                if len(args) > 0:
                    true_inputs += [data['inputs'] for data in args]
                    true_data_samples += [data['data_samples'] for data in args]
                return self.aug_test(true_inputs, true_data_samples, **kwargs)
            if isinstance(data_samples[0], list):
                # aug test
                assert len(data_samples[0]) == 1, 'Only support ' \
                                                  'batch_size 1 ' \
                                                  'in mmdet3d when ' \
                                                  'do the test' \
                                                  'time augmentation.'
                return self.aug_test(inputs, data_samples, **kwargs)
            else:
                return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def extract_feat(
        self, batch_inputs_dict: Dict[str, Tensor]
    ) -> Union[Tuple[Tensor], Dict[str, Tensor]]:
        """Directly extract features from the backbone+neck.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'img' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - imgs (torch.Tensor, optional): Image of each sample.

        Returns:
            tuple[Tensor] | dict:  For outside 3D object detection, we
                typically obtain a tuple of features from the backbone + neck,
                and for inside 3D object detection, usually a dict containing
                features will be obtained.
        """
        points = batch_inputs_dict['points']
        if self.mode == self.POINT:
            p = torch.stack(points)
        else:
            raise NotImplementedError
        x = self.backbone(p)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs_dict: Dict[str, Union[List, Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        feat_dict = self.extract_feat(batch_inputs_dict)
        points = batch_inputs_dict['points']
        losses = self.bbox_head.loss(points, feat_dict, batch_data_samples,
                                     **kwargs)
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instances, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                    contains a tensor with shape (num_instances, 7).
        """
        feats_dict = self.extract_feat(batch_inputs_dict)
        points = batch_inputs_dict['points']
        results_list = self.bbox_head.predict(points, feats_dict,
                                              batch_data_samples, **kwargs)
        data_3d_samples = self.add_pred_to_datasample(batch_data_samples,
                                                      results_list)
        return data_3d_samples

    def aug_test(self, aug_inputs_list: List[dict],
                 aug_data_samples: List[List[dict]], **kwargs):
        """Test with augmentation.

        Batch size always is 1 when do the augtest.

        Args:
            aug_inputs_list (List[dict]): The list indicate same data
                under differecnt augmentation.
            aug_data_samples (List[List[dict]]): The outer list
                indicate different augmentation, and the inter
                list indicate the batch size.
        """
        num_augs = len(aug_inputs_list)
        if num_augs == 1:
            return self.predict(aug_inputs_list[0], aug_data_samples[0])

        batch_size = len(aug_data_samples[0])
        assert batch_size == 1
        multi_aug_results = []
        for aug_id in range(num_augs):
            batch_inputs_dict = aug_inputs_list[aug_id]
            batch_data_samples = aug_data_samples[aug_id]
            feats_dict = self.extract_feat(batch_inputs_dict)
            points = batch_inputs_dict['points']
            results_list = self.bbox_head.predict(points, feats_dict, batch_data_samples, **kwargs)
            multi_aug_results.append(results_list[0])
        aug_input_metas_list = []
        for aug_id in range(num_augs):
            metainfo = aug_data_samples[aug_id][0].metainfo
            aug_input_metas_list.append(metainfo)

        aug_results_list = [item.to_dict() for item in multi_aug_results]
        # after merging, bboxes will be rescaled to the original image size
        merged_results_dict = merge_aug_bboxes_3d(aug_results_list, aug_input_metas_list, self.bbox_head.test_cfg)

        merged_results = InstanceData(**merged_results_dict)
        data_3d_samples = self.add_pred_to_datasample(aug_data_samples[0], [merged_results])
        return data_3d_samples

