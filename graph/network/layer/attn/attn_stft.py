import torch
import torch.nn as nn
from einops import rearrange
from tsignal.window import get_window_function
from complextorch import CVTensor
from complextorch.nn.modules.attention.eca import CVEfficientChannelAttention1d

__all__ = ["STFTFilter", "STFTFilter2"]


class STFTFilter(nn.Module):
    """
    Short-Time Frequency Transform based attention module
    """

    def __init__(self, in_channels: int, in_dims: int, n_fft: int, hop_length: int, window_name: str = "tukey"):
        """
        :param in_dims: the length of the input tensor.
        :param n_fft: the number of frequencies in FFT.
        :param hop_length: the distance between neighboring sliding window frames.
        :param window_name: the window function name. Default: "tukey".
        """
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        # get window
        self.window = get_window_function(window_name)(n_fft)

        self.frame_length = int(n_fft // 2) + 1
        self.num_frames = 1 + int((in_dims - n_fft) // hop_length)
        self.w = nn.Parameter(torch.ones((in_channels, self.frame_length, self.num_frames, 2)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, C, L).
        :return: the output tensor in shape of (N, C, L).
        """
        shape = x.size()
        # shape: (N, C, L) -> (N * C, L)
        x = rearrange(x, "b c l -> (b c) l").contiguous()
        # shape: (N * C, L) -> (N*C, frame_length, num_frames)
        x = torch.stft(x, self.n_fft, self.hop_length, window=self.window.to(device=x.device), center=False, return_complex=True)
        w = torch.view_as_complex(self.w)
        # shape: (N * C, frame_length, num_frames) -> (N, C, frame_length, num_frames)
        x = rearrange(x, "(b c) f t -> b c f t", c=shape[-2])
        x = x * w
        # shape: (N, C, frame_length, num_frames) -> (N*C, frame_length, num_frames)
        x = rearrange(x, 'b c f t -> (b c) f t')
        # shape: (N * C, F, T) -> (N * C, L)
        x = torch.istft(x, self.n_fft, self.hop_length, window=self.window.to(x.device), length=shape[-1])
        # shape: (N * C, L) -> (N, C, L)
        x = rearrange(x, '(b c) l -> b c l', c=shape[-2])
        return x


class STFTFilter2(nn.Module):
    """
    Short-Time Frequency Transform based attention module using Efficient Channel Attention.
    """

    def __init__(self, in_channels: int, n_fft: int, hop_length: int, window_name: str = "tukey"):
        """
        :param in_channels: the number of channels in input tensor.
        :param n_fft: the number of frequencies in FFT.
        :param hop_length: the distance between neighboring sliding window frames.
        :param window_name: the window function name. Default: "tukey".
        """
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        # get window
        self.window = get_window_function(window_name)(n_fft)
        self.eca = CVEfficientChannelAttention1d(in_channels * self.frame_length, b=18, gamma=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, C, L).
        :return: the output tensor in shape of (N, C, L).
        """
        shape = x.size()
        # shape: (N, C, L) -> (N * C, L)
        x = rearrange(x, "b c l -> (b c) l").contiguous()
        # shape: (N * C, L) -> (N*C, frame_length, num_frames)
        x = torch.stft(x, self.n_fft, self.hop_length, window=self.window.to(device=x.device), center=False, return_complex=True)
        x = rearrange(x, "(b c) f t -> b (c f) t", c=shape[-2])
        x = torch.view_as_real(x)
        x = CVTensor(x[..., 0], x[..., 1])
        x = self.eca(x).complex
        # mapping
        # shape: (N, C, frame_length, num_frames) -> (N*C, frame_length, num_frames)
        x = rearrange(x, "b (c f) t -> (b c) f t", c=shape[-2])
        # shape: (N * C, F, T) -> (N * C, L)
        x = torch.istft(x, self.n_fft, self.hop_length, window=self.window.to(x.device), length=shape[-1])
        # shape: (N * C, L) -> (N, C, L)
        x = rearrange(x, "(b c) l -> b c l", c=shape[-2])
        return x
