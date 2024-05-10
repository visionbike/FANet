from torch import nn
from torch.optim import Optimizer


def get_lr_scheduler(optimizer: Optimizer, **kwargs) -> None | nn.Module:
    """
    Get learning rate scheduler.

    :param optimizer: the optimizer.
    :param kwargs: the optional configuration.
    :return: the LR scheduler.
    """
    name = kwargs.pop("name")
    if name == "none":
        scheduler = None
    elif name == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, **kwargs)
    elif name == "multi_step":
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, **kwargs)
    elif name == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, **kwargs)
    elif name == "cosine_onecycle":
        from .cosine_onecycle_start import CosineAnnealingOneCycleStartLR
        scheduler = CosineAnnealingOneCycleStartLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Invalid LR scheduler name, but got {name}.")
    return scheduler
