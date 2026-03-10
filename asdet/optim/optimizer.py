try:
    from muon import SingleDeviceMuonWithAuxAdam
except ImportError as e:
    print(e)
    print(r'If Muon Optimizer is not yet installed, run the command: pip install git+https://github.com/KellerJordan/Muon')
    exit()
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from mmengine.registry import OPTIM_WRAPPERS, OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.optim import OptimWrapper, DefaultOptimWrapperConstructor


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MuonOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """OptimWrapper constructor for MuonWithAuxAdam with layer-wise grouping."""

    def __call__(self, model: nn.Module) -> OptimWrapper:
        if hasattr(model, 'module'):
            model = model.module

        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        optim_wrapper_cfg.setdefault('type', 'OptimWrapper')

        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_type = optimizer_cfg.pop('type')
        assert optimizer_type == 'MuonWithAuxAdam'

        lr1 = optimizer_cfg.get('lr1')
        lr2 = optimizer_cfg.get('lr2')
        weight_decay = optimizer_cfg.get('weight_decay', 0.0)
        betas = optimizer_cfg.get('betas', (0.9, 0.95))
        min_hidden_dims = optimizer_cfg.get('min_hidden_dims', 32)
        include_params = optimizer_cfg.get('include_params', [])
        exclude_params = optimizer_cfg.get('exclude_params', [])

        hidden_layers = dict()
        nonhidden_layers = dict()
        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                full_name = name + '.' + param_name
                if param.ndim < 2:
                    nonhidden_layers[full_name] = param
                else:
                    weight_shape = param.shape if 'kernel' not in full_name else param.shape[1:]
                    if min(weight_shape) < min_hidden_dims:
                        if isinstance(module, nn.Linear):
                            nonhidden_layers[full_name] = param
                        elif isinstance(module, _ConvNd) and min(weight_shape[:2]) < min_hidden_dims:
                            nonhidden_layers[full_name] = param
                        else:
                            hidden_layers[full_name] = param
                    else:
                        hidden_layers[full_name] = param

        hidden_params = []
        nonhidden_params = []
        for k, v in hidden_layers.items():
            for extra in exclude_params:
                if extra in k:
                    nonhidden_params.append(v)
                    break
            else:
                hidden_params.append(v)
        for k, v in nonhidden_layers.items():
            for extra in include_params:
                if extra in k:
                    hidden_params.append(v)
                    break
            else:
                nonhidden_params.append(v)

        param_groups = [
            dict(params=hidden_params, use_muon=True, lr=lr1, weight_decay=weight_decay),
            dict(params=nonhidden_params, use_muon=False, lr=lr2, weight_decay=weight_decay, betas=betas),
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        for key, value in optimizer.param_groups[0].items():
            if key in ['lr', 'weight_decay']:
                optimizer.defaults[key] = value
        optimizer.defaults['betas'] = (optimizer.param_groups[0]['momentum'], None)
        optimizer.param_groups[0]['betas'] = (optimizer.param_groups[0]['momentum'], None)
        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg,
            default_args=dict(optimizer=optimizer)
        )
        return optim_wrapper

