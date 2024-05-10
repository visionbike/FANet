import torch
from torch import nn as nn
from einops import rearrange
from einops.layers.torch import Reduce
from tsignal.window import get_window_function
from tsignal import dct, sdct

__all__ = ["SDCTFilter"]


class SDCTFilter(nn.Module):
    """
    Short-Time Discrete Cosine Transform based attention module.
    """

    def __init__(self, in_dims: int, n_fft: int, hop_length: int, window_name: str = "tukey"):
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
        self.fc = nn.Sequential(
            Reduce("bc f t -> bc f", reduction="mean"),
            nn.Linear(n_fft, in_dims, False),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, C, L).
        :return: the output tensor in shape of (N, C, L).
        """
        shape = x.size()
        # shape: (N, C, L) -> (N * C, L)
        x = rearrange(x, "b c l -> (b c) l")
        # shape: (N * C, L) -> (N * C, frame_length, num_frames)
        w = sdct(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x.device))
        # (N * C, frame_length, num_frames) -> (N * C, L)
        w = self.fc(w)
        w = rearrange(w, "(b c) l -> b c l", c=shape[-2])
        x = rearrange(x, "(b c) l -> b c l", c=shape[-2])
        x = x * w
        return x
