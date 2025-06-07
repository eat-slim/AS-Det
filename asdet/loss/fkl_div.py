import torch
from torch import Tensor


def focal_kl_div(
    input: Tensor,
    target: Tensor,
    reduction: str = "sum",
    log_target: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    assert reduction in ['mean', 'sum', 'none']
    assert input.shape == target.shape

    # weight = log(ReLU((y - x) / eps) + e)

    if not log_target:
        target = target.log()

    weight = (torch.relu((target.exp() - input.exp()) / eps) + torch.e).log()
    weight = weight.detach()
    loss = torch.kl_div(input, target, log_target=True) * weight

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        pass
    return loss
