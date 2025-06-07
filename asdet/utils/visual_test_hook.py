import os
import pickle
import torch
import matplotlib.pyplot as plt
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS
from typing import Optional, Sequence, Union
from .visualization import bbox2o3d, bbox2corner
from .point_sampling import voxel_sampling

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class VisualTestHook(Hook):
    priority = 'LOW'

    def __init__(self, root: str = 'auto', interval: int = 1, **kwargs):
        self.root = root
        self.image_dir = None
        self.source_dir = None
        self.interval = interval

    def before_test(self, runner) -> None:
        if self.root == 'auto':
            self.root = os.path.join(runner.work_dir, 'visualization')
        self.image_dir = os.path.join(self.root, 'image')
        self.source_dir = os.path.join(self.root, 'source')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.source_dir, exist_ok=True)
        runner.logger.info(f'Visual records will be saved to {self.root}.')

    def after_test_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: Optional[Sequence] = None):
        if (batch_idx + 1) % self.interval != 0:
            return

        # input_path = os.path.basename(data_batch['data_samples'][0].img_path[0])
        # input_name = input_path.split('.')[0]
        if isinstance(data_batch, list):
            data_batch = data_batch[0]
        input_path = os.path.basename(data_batch['data_samples'][0].lidar_path)
        input_name = input_path[:input_path.rfind('.')]
        if outputs[0].eval_ann_info is not None:
            gt_bbox = outputs[0].eval_ann_info['gt_bboxes_3d'].tensor  # object_num, 7
        elif len(outputs[0].gt_instances_3d) > 0:
            gt_bbox = outputs[0].gt_instances_3d.bboxes_3d.tensor
        else:
            gt_bbox = []
        test_recorder = {
            'lidar_path': data_batch['data_samples'][0].lidar_path,
            'input_points': data_batch['inputs']['points'][0].cpu(),
            'gt_bbox': gt_bbox.cpu(),
            'pred_labels': outputs[0].pred_instances_3d.labels_3d.detach().cpu(),
            'pred_bboxes': outputs[0].pred_instances_3d.bboxes_3d.detach().cpu().tensor,  # object_num, 7
            'pred_scores': outputs[0].pred_instances_3d.scores_3d.detach().cpu(),
            'key_points': getattr(outputs[0].pred_instances_3d.detach().cpu(), 'key_points', torch.tensor([])).detach().cpu(),
        }
        test_recorder.update({k: v.detach().cpu() for k, v in outputs[0].pred_instances_3d['metainfo'].items() if 'sampled_points' in k})
        self._save(test_recorder, input_name)
        self._plot(test_recorder, input_name)

    def _save(self, test_recorder, out_name):
        filename = os.path.join(self.source_dir, f'{out_name}.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(test_recorder, file)

    def _plot(self, test_recorder, out_name):
        filename = os.path.join(self.image_dir, f'{out_name}.jpg')
        input_points = test_recorder['input_points']  # input pointcloud, N, 4
        input_points = voxel_sampling([input_points], (0.1, 0.1, 0.1))[0]
        gt_bbox = test_recorder['gt_bbox']  # object_num, 7
        pred_labels = test_recorder['pred_labels']
        pred_bboxes = test_recorder['pred_bboxes']  # object_num, 7
        pred_scores = test_recorder['pred_scores']
        key_points = test_recorder['key_points']  # N, 3
        colors = ['#FF9900', '#9900FF', '#FF0000']

        plt.figure(figsize=(30, 15), dpi=100)
        plt.subplot(121)
        plt.title('X-Y (bev)')
        plt.axis('equal')
        plt.scatter(x=input_points[:, 0], y=input_points[:, 1], c=input_points[:, 2], s=1, label='input point cloud')
        if len(key_points) > 0:
            plt.scatter(key_points[:, 0], key_points[:, 1], s=20, marker='s', c='r', linewidths=1, edgecolors='k')
        for bbox in gt_bbox:
            conner = bbox2corner(bbox)  # 8, 3
            plt.plot([conner[0, 0], conner[1, 0]], [conner[0, 1], conner[1, 1]], c='blue')
            plt.plot([conner[1, 0], conner[5, 0]], [conner[1, 1], conner[5, 1]], c='blue')
            plt.plot([conner[5, 0], conner[4, 0]], [conner[5, 1], conner[4, 1]], c='blue')
            plt.plot([conner[4, 0], conner[0, 0]], [conner[4, 1], conner[0, 1]], c='blue')
            plt.scatter(x=conner[:, 0].mean(), y=conner[:, 1].mean(), c='blue', marker='x')

        for bbox, conf, cls in zip(pred_bboxes, pred_scores, pred_labels):
            conner = bbox2corner(bbox)  # 8, 3
            color = colors[cls % len(colors)]
            plt.plot([conner[0, 0], conner[1, 0]], [conner[0, 1], conner[1, 1]], c=color, alpha=conf.cpu().item())
            plt.plot([conner[1, 0], conner[5, 0]], [conner[1, 1], conner[5, 1]], c=color, alpha=conf.cpu().item())
            plt.plot([conner[5, 0], conner[4, 0]], [conner[5, 1], conner[4, 1]], c=color, alpha=conf.cpu().item())
            plt.plot([conner[4, 0], conner[0, 0]], [conner[4, 1], conner[0, 1]], c=color, alpha=conf.cpu().item())
            plt.scatter(x=conner[:, 0].mean(), y=conner[:, 1].mean(), c=color, marker='x')

        plt.subplot(122)
        plt.title('Y-Z (front)')
        plt.axis('equal')
        plt.scatter(x=input_points[:, 1], y=input_points[:, 2], c=input_points[:, 0], s=1, label='input point cloud')
        for bbox in gt_bbox:
            conner = bbox2corner(bbox)  # 8, 3
            plt.plot([conner[:, 1].min(), conner[:, 1].min()], [conner[:, 2].min(), conner[:, 2].max()],
                     c='blue')  # 3-0
            plt.plot([conner[:, 1].min(), conner[:, 1].max()], [conner[:, 2].max(), conner[:, 2].max()],
                     c='blue')  # 0-4
            plt.plot([conner[:, 1].max(), conner[:, 1].max()], [conner[:, 2].max(), conner[:, 2].min()],
                     c='blue')  # 4-7
            plt.plot([conner[:, 1].max(), conner[:, 1].min()], [conner[:, 2].min(), conner[:, 2].min()],
                     c='blue')  # 7-0
            plt.scatter(x=conner[:, 1].mean(), y=conner[:, 2].mean(), c='blue', marker='x')

        for bbox, conf, cls in zip(pred_bboxes, pred_scores, pred_labels):
            conner = bbox2corner(bbox)  # 8, 3
            color = colors[cls % len(colors)]
            plt.plot([conner[:, 1].min(), conner[:, 1].min()], [conner[:, 2].min(), conner[:, 2].max()], c=color,
                     alpha=conf.cpu().item())
            plt.plot([conner[:, 1].min(), conner[:, 1].max()], [conner[:, 2].max(), conner[:, 2].max()], c=color,
                     alpha=conf.cpu().item())
            plt.plot([conner[:, 1].max(), conner[:, 1].max()], [conner[:, 2].max(), conner[:, 2].min()], c=color,
                     alpha=conf.cpu().item())
            plt.plot([conner[:, 1].max(), conner[:, 1].min()], [conner[:, 2].min(), conner[:, 2].min()], c=color,
                     alpha=conf.cpu().item())
            plt.scatter(x=conner[:, 1].mean(), y=conner[:, 2].mean(), c=color, marker='x')

        plt.savefig(filename)
        plt.close()
