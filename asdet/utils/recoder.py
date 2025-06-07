from torch import Tensor
from typing import Dict, List


class Recorder:
    def __init__(self):
        self.record_dict: Dict[str, List] = {}
        self.reduction_func = {
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'best': self.best,
            'none': lambda x: x
        }

    def add_dict(self, metric_dict: dict):
        for key, value in metric_dict.items():
            if key not in self.record_dict.keys():
                self.record_dict[key] = []
            if isinstance(value, Tensor):
                value = value.cpu().detach().item()
            self.record_dict.get(key).append(value)

    def add_item(self, key: str, value):
        if key not in self.record_dict.keys():
            self.record_dict[key] = []
        if isinstance(value, Tensor):
            value = value.cpu().detach().item()
        self.record_dict.get(key).append(value)

    def mean(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                return_dict[key] = sum(value) / len(value)
        return return_dict

    def max(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                return_dict[key] = max(value)
        return return_dict

    def min(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                return_dict[key] = min(value)
        return return_dict

    def best(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                if value[0] > value[-1]:
                    return_dict[key] = min(value)
                else:
                    return_dict[key] = max(value)
        return return_dict

    def tostring(self, reduction='best') -> str:
        assert reduction in ['min', 'max', 'mean', 'best', 'none']
        reduction_dic = self.reduction_func.get(reduction)()
        string = ''
        if len(reduction_dic) > 0:
            for key, value in reduction_dic.items():
                if isinstance(value, list):
                    value_str = value
                else:
                    value_str = f'{value:4.5f}'
                string += f'\t{key:<20s}: ({value_str})\n'
            string = '\n' + string
        return string

    def clear(self):
        self.record_dict.clear()
