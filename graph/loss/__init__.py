import torch.nn as nn

__all__ = ["get_loss"]


def get_loss(**kwargs) -> nn.Module:
    """
    Get loss function.

    :param kwargs: the criterion configuration.
    :return: the loss function.
    """
    name = kwargs.pop("name")
    if name == "ce":
        loss = nn.CrossEntropyLoss(**kwargs)
    elif name == 'focal':
        from .focal import FocalLoss
        loss = FocalLoss(**kwargs)
    else:
        raise ValueError(f"Invalid loss function name, but got {name}.")
    return loss
