import math
import torch
from .valid import *

__all__ = ["tukey_window"]


def tukey_window(win_length: int,
                 periodic: bool = True,
                 *,
                 dtype: torch.dtype = None,
                 layout: torch.layout = torch.strided,
                 device: torch.device = None,
                 requires_grad: bool = False) -> torch.Tensor:
    """
    The PyTorch-based Tukey window function.

    :param win_length: the size of returned window.
    :param periodic: whether to return a window to be used as periodic function, otherwise, to return a symmetric window.
    :param dtype: the desired data type of returned tensor.
    :param layout: the desired layout of returned tensor.
    :param device: the desired device of returned tensor.
    :param requires_grad: whether "autograd" should record operations on the returned tensor.
    :return: the output tensor if size (win_length,) containing the window.
    """
    return _tukey_function(win_length, sym=(not periodic), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)


def _tukey_function(M: int,
                    *,
                    alpha: float = 0.5,
                    sym: bool = True,
                    dtype: torch.dtype = None,
                    layout: torch.layout = torch.strided,
                    device: torch.device = None,
                    requires_grad: bool = False) -> torch.Tensor:
    """
    The tukey function inspired from Scipy.

    Reference:
    * https://github.com/scipy/scipy/blob/main/scipy/signal/windows/_windows.py

    :param M: the number of points in the output window. If zero, an empty array is returned.
    :param alpha: the cosine fraction, specified as a real scalar. Default: 0.5
    :param sym: the shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window. Default: 0.5.
    :param dtype: the desired data type of returned tensor.
    :param layout: the desired layout of returned tensor.
    :param device: the desired device of returned tensor.
    :param requires_grad: whether to autograd should record operations on the returned tensor.
    :return: the output tensor if size (win_length,) containing the window.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()

    check_window_function("tukey", M, dtype, layout)

    if M == 0:
        return torch.empty((0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    elif M == 1:
        return torch.ones((1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    if alpha <= 0.0:
        return torch.ones((M,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    elif alpha >= 1.0:
        return torch.hann_window(M, periodic=(not sym), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)

    M, need_trunc = (M + 1, True) if not sym else (M, False)

    n = torch.arange(0, M, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    width = int(math.floor(alpha * (M - 1) / 2.0))
    w = torch.concatenate([
        0.5 * (1 + torch.cos(torch.pi * (-1 + 2.0 * n[0: width + 1] / alpha / (M - 1)))),
        torch.ones(n[width + 1: M - width - 1].shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad),
        0.5 * (1 + torch.cos(torch.pi * (-2.0 / alpha + 1 + 2.0 * n[M - width - 1:] / alpha / (M - 1))))
    ])

    return w[: -1] if need_trunc else w
