import torch
from torch import nn
from .init_zero import *

__all__ = ["init_weights"]


@torch.no_grad()
def init_weights(m: nn.Module) -> None:
    """
    Init layer's weights.

    :param m: Weight initialization function for the network.
    """
    if isinstance(m, nn.Linear):
        init_zero_linear(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv1d):
        init_zero_conv1d(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        init_zero_conv2d(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, tuple([nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d])):
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
