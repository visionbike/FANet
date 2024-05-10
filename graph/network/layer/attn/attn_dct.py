import torch
from torch import nn as nn
from tsignal import dct

__all__ = ["DCTFilter"]


class DCTFilter(nn.Module):
    """
    DCT-based Global Filter that replaces self-attention.
    """

    def __init__(self, in_dims: int):
        """
        :param in_dims: the number of channels in input tensor.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dims, in_dims, bias=False),
            nn.Mish()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, C, L).
        :return: the output tensor in shape of (N, C, L).
        """
        # compute weight based on the frequency feature
        w = self.fc(dct(x, 'ortho'))
        x = x * w
        return x
