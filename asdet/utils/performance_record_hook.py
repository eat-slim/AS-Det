from typing import Optional, Union, Dict

from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS


DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class PerformanceRecordHook(Hook):
    priority = 'VERY_LOW'

    def __init__(self, key='Overall_3D_AP40_moderate'):
        self.key = key
        self.best_performance = None
        self.best_performance_all = None

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        if runner.rank != 0:
            return

        is_found = False
        for label, metric in metrics.items():
            if label.split("/")[-1] == self.key:
                this_performance = metric
                is_found = True

        if not is_found:
            runner.logger.warning(f'The performance corresponding to the given key \'{self.key}\' was not found.')
            return

        if self.best_performance is None or this_performance >= self.best_performance:
            self.best_performance = this_performance
            self.best_performance_all = (runner.epoch, metrics)
            runner.logger.info(f'Found a new best performance with {this_performance:.4f} {self.key}')

    def after_train(self, runner):
        if self.best_performance is None:
            return

        performance_str = ''
        for label, metric in self.best_performance_all[1].items():
            performance_str += f'{label.split("/")[-1]}: {metric:.4f}\n'

        runner.logger.info(f'The Best Performance appeared in epoch {self.best_performance_all[0]}:\n\n'
                           f'{performance_str}\n')



