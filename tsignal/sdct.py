import torch
from .dct import *

__all__ = ["sdct", "isdct"]


def sdct(x: torch.Tensor, n_fft: int, hop_length: int = None, window: torch.Tensor = None) -> torch.Tensor:
    """
    Short-Time Discrete Cosine Transform (SDCT). No padding is applied to the signals.

    References:
    * https://github.com/jonashaag/pydct
    * https://github.com/seungwonpark/istft-pytorch/blob/master/istft_irfft.py

    :param x: the input signal in shape of (..., L) with L is number of samples.
    :param n_fft: the number of frequencies in FFT.
    :param hop_length: the distance between neighboring sliding window frames. Default is None (treated as floor(n_fft // 4)).
    :param window: the window to use for DCT.
    :return: the short-Time DCT matrix in shape of (..., frame_length, num_frames).
    """
    framed = x.unfold(-1, n_fft, (n_fft - hop_length))
    if window is not None:
        window = window.to(framed)
        framed = framed * window
    return dct(framed, norm="ortho").transpose(-1, -2)


def isdct(x: torch.Tensor, *, n_fft: int, hop_length: int = None, window: torch.Tensor = None, length: int = None) -> torch.Tensor:
    """
    Inverse Short-Time Discrete Cosine Transform (iSDCT).

    References:
    * https://github.com/jonashaag/pydct
    * https://github.com/seungwonpark/istft-pytorch/blob/master/istft_irfft.py

    :param x: the Short-Time DCT matrix in shape of (..., frame_length, num_frames).
    :param n_fft: the number of frequencies in FFT.
    :param hop_length: the distance between neighboring sliding window frames. Default is None (treated as `floor(n_fft // 4)`).
    :param window: the window to use for DCT.
    :param length: the number of samples of the output signal.
    :return: the reconstructed output signals in shape of (..., L) with L is the number of samples.
    """
    # shape: (..., frame_length, num_frames)
    shape = x.size()
    if n_fft is None:
        n_fft = 2 * (shape[-2] - 1)
    if hop_length is None:
        hop_length = n_fft // 4
    if window is not None:
        window = window.to(x.device).view(1, -1)

    frame_step = (n_fft - hop_length)
    length_expected = n_fft + frame_step * (shape[-1] - 1)
    z = torch.zeros(shape[0], length_expected, device=x.device)

    # overlap and add
    for i in range(shape[-1]):
        idx = i * frame_step
        idcted = idct(x[..., i], norm="ortho")
        z[..., idx: (idx + n_fft)] += idcted * window
    z = z[..., (n_fft // 2):]

    if length is not None:
        if z.size(-1) > length:
            z = z[..., :length]
        elif z.size(-1) < length:
            z = torch.cat([z[..., :length], torch.zeros(shape[0], length - z.size(-1), device=z.device)], dim=1)
    coeff = n_fft / float(hop_length) / 2.0
    z = z / coeff
    return z
