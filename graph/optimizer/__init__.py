from typing import Iterable
import torch
from torch.optim import Optimizer

__all__ = ["get_optimizer"]


def get_optimizer(parameters: [Iterable[torch.Tensor] | Iterable[dict]], **kwargs) -> Optimizer:
    """
    Get optimizer.

    :param parameters: The parameters of the optimizer.
    :param kwargs: The parameters of the optimizer.
    :return: The optimizer function
    """
    name = kwargs.pop("name")
    if name == "ranger20":
        from .ranger20 import Ranger20
        return Ranger20(parameters, **kwargs)
    else:
        raise ValueError(f"Invalid optimizer name, but got {name}.")
