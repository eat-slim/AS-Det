from typing import Optional, Sequence, Union, Dict, Iterable, Literal
import io
import os
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from PIL import Image

from mmengine.hooks import Hook
from mmengine.model.wrappers import MMDistributedDataParallel
from mmdet3d.registry import HOOKS

from .visualization import bbox2o3d, bbox2corner
from .recoder import Recorder
from .point_sampling import voxel_sampling


DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class TensorboardHook(Hook):
    priority = 'LOW'

    def __init__(self, root: str = 'auto', interval: int = 10, iter_based_plot: int = -1,
                 metric_format: Literal['kitti', 'nuscenes'] = 'kitti', **kwargs):
        self.root = root
        self.interval = interval
        self.iter_cnt = 0
        self.writer = None
        self.eval_recorder = {}
        self.iter_based_plot = iter_based_plot
        self.metric_format = metric_format

    def before_train(self, runner) -> None:
        if runner.rank != 0:
            return

        if self.root == 'auto':
            name = os.path.split(runner.work_dir)[-1]
        else:
            name = self.root

        log_root = os.path.join('log_tb', name)
        comment = runner.cfg.comment
        log_dir = log_root + f" ({runner.cfg.comment})" if comment != '' else log_root
        self.writer = SummaryWriter(log_dir=log_dir)
        self.epoch_recoder = Recorder()
        self.iter_recoder = Recorder()
        runner.logger.info(f'Tensorboard records will be saved to {log_root}.')

    def before_train_epoch(self, runner) -> None:
        if runner.rank != 0:
            return

        lr_list = sorted(list(set([i['lr'] for i in runner.param_schedulers[0].optimizer.param_groups])), reverse=True)
        self.writer.add_scalar("runtime/learning_rate", lr_list[0], runner.epoch)
        for i, lr in enumerate(lr_list[1:]):
            self.writer.add_scalar(f"runtime/learning_rate{i+2}", lr, runner.epoch)

        if 'momentum' in runner.param_schedulers[0].optimizer.param_groups[0].keys():
            mom_list = sorted(list(set([i['momentum'] for i in runner.param_schedulers[0].optimizer.param_groups])), reverse=True)
            self.writer.add_scalar("runtime/momentum", mom_list[0], runner.epoch)
            for i, mom in enumerate(mom_list[1:]):
                self.writer.add_scalar(f"runtime/momentum{i+2}", mom, runner.epoch)

        model = runner.model
        if isinstance(model, MMDistributedDataParallel):
            model = model.module
        if hasattr(model, 'backbone'):
            for key, value in getattr(model.backbone, 'runtime_cfg', {}).items():
                if not isinstance(value, Iterable):
                    self.writer.add_scalar(f"runtime/{key}", value, runner.epoch)

    def after_train_epoch(self, runner) -> None:
        if runner.rank != 0:
            return

        epoch_metrics = self.epoch_recoder.mean()
        for label, metric in epoch_metrics.items():
            self.writer.add_scalar(f"train/epoch_{label}", metric, runner.epoch)
        self.epoch_recoder.clear()

        model = runner.model
        if isinstance(model, MMDistributedDataParallel):
            model = model.module
        if hasattr(model, 'bbox_head') and (getattr(model.bbox_head, 'figure_tobe_show_in_tensorboard', None) is not None):
            self.writer.add_image('train_active_sampling', model.bbox_head.figure_tobe_show_in_tensorboard, runner.epoch)
            model.bbox_head.figure_tobe_show_in_tensorboard = None

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs: Optional[dict] = None) -> None:
        if runner.rank != 0:
            return

        self.epoch_recoder.add_dict(outputs)
        self.iter_recoder.add_dict(outputs)
        if (runner.iter + 1) % self.interval == 0:
            iter_metrics = self.iter_recoder.mean()
            for label, metric in iter_metrics.items():
                self.writer.add_scalar(f"train/step_{label}", metric, runner.iter)
            self.iter_recoder.clear()
        if self.iter_based_plot > 0 and (runner.iter + 1) % self.iter_based_plot == 0:
            model = runner.model
            if isinstance(model, MMDistributedDataParallel):
                model = model.module
            if getattr(model.bbox_head, 'figure_tobe_show_in_tensorboard', None) is not None:
                self.writer.add_image('train_active_sampling', model.bbox_head.figure_tobe_show_in_tensorboard, runner.iter)
                model.bbox_head.figure_tobe_show_in_tensorboard = None

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        if runner.rank != 0:
            return

        if self.writer is not None:
            for label, metric in metrics.items():
                if self.metric_format == 'kitti':
                    if '3D_AP40' in label:
                        if isinstance(metric, Tensor):
                            metric = metric.cpu().detach().item()
                        self.writer.add_scalar(f"eval/{label}", metric, runner.epoch)
                elif self.metric_format == 'nuscenes':
                    label = label.split('/')[-1]
                    if label in ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE', 'NDS', 'mAP']:
                        if isinstance(metric, Tensor):
                            metric = metric.cpu().detach().item()
                        self.writer.add_scalar(f"eval/NuScenes metric/{label}", metric, runner.epoch)
            self._plot_gt_pred(runner)
            self.eval_recorder.clear()

            model = runner.model
            if isinstance(model, MMDistributedDataParallel):
                model = model.module
            if hasattr(model, 'bbox_head') and (getattr(model.bbox_head, 'figure_tobe_show_in_tensorboard', None) is not None):
                self.writer.add_image('val_active_sampling', model.bbox_head.figure_tobe_show_in_tensorboard, runner.epoch)
                model.bbox_head.figure_tobe_show_in_tensorboard = None

    def after_val_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: Optional[Sequence] = None):
        if runner.rank != 0:
            return

        for i, output in enumerate(outputs):
            if output.eval_ann_info is not None:
                gt_bbox = output.eval_ann_info['gt_bboxes_3d'].tensor  # object_num, 7
            else:
                gt_bbox = output.gt_instances_3d.bboxes_3d.tensor
            if gt_bbox.shape[0] > len(self.eval_recorder.get('gt_bbox', [])) or len(self.eval_recorder) == 0:
                self.eval_recorder = {
                    'input_points': data_batch['inputs']['points'][i],
                    'gt_bbox': gt_bbox,
                    'pred_labels': output.pred_instances_3d.labels_3d.detach().cpu(),
                    'pred_bboxes': output.pred_instances_3d.bboxes_3d.detach().cpu().tensor,  # object_num, 7
                    'pred_scores': output.pred_instances_3d.scores_3d.detach().cpu(),
                    'key_points': getattr(output.pred_instances_3d, 'key_points', torch.tensor([])).detach().cpu(),
                }

    def _plot_gt_pred(self, runner):
        import matplotlib.pyplot as plt
        input_points = self.eval_recorder['input_points']  # input pointcloud, N, 4
        input_points = voxel_sampling([input_points], (0.1, 0.1, 0.1))[0]
        gt_bbox = self.eval_recorder['gt_bbox']  # object_num, 7
        pred_labels = self.eval_recorder['pred_labels']
        pred_bboxes = self.eval_recorder['pred_bboxes']  # object_num, 7
        pred_scores = self.eval_recorder['pred_scores']
        key_points = self.eval_recorder['key_points']  # N, 3
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

        plot_buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        image = Image.open(plot_buf)
        image = ToTensor()(image)
        self.writer.add_image('val_predictions', image, runner.epoch)
