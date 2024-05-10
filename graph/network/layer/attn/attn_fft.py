import torch
from torch import nn
from complextorch import CVTensor
from complextorch.nn.modules.attention.eca import CVEfficientChannelAttention1d

__all__ = ["FFTFilter", "FFTFilter2"]


class FFTFilter(nn.Module):
    """
    FFT attention module using global filter.
    """

    def __init__(self, in_channels: int, in_dims: int):
        """
        :param in_channels: the number of channels in input tensor.
        :param in_dims: the length of the input tensor.
        """
        super().__init__()
        out_dims = int(in_dims // 2) + 1
        self.w = nn.Parameter(torch.ones((in_channels, out_dims, 2,)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, C, L).
        :return: the output tensor in shape of (N, C, L).
        """
        shape = x.size()
        # shape: (N, C, L) -> (N, C, L // 2 + 1)
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        w = torch.view_as_complex(self.w)
        x = x * w
        # shape: (N, C, L // 2 + 1) -> (N, C, L)
        x = torch.fft.irfft(x, dim=-1, n=shape[-1], norm='ortho')
        return x


class FFTFilter2(nn.Module):
    """
    FFT attention module using Efficient Channel Attention.
    """

    def __init__(self, in_channels: int, in_dims: int):
        """
        :param in_channels: the number of channels in input tensor.
        :param in_dims: the length of the input tensor.
        """
        super().__init__()
        out_dims = int(in_dims // 2) + 1
        self.eca = CVEfficientChannelAttention1d(in_channels, b=18, gamma=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, C, L).
        :return: the output tensor in shape of (N, C, L).
        """
        shape = x.size()
        # shape: (N, C, L) -> (N, C, L // 2 + 1)
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        x = torch.view_as_real(x)
        x = CVTensor(x[..., 0], x[..., 1])
        x = self.eca(x).complex
        # shape: (N, C, L // 2 + 1) -> (N, C, L)
        x = torch.fft.irfft(x, dim=-1, n=shape[-1], norm='ortho')
        return x
