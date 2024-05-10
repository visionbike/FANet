from typing import Callable
import torch


def get_window_function(name: str = "tukey") -> Callable[..., torch.Tensor]:
    """
    Get the signal window function.

    :param name: the window function name. Supporting "hann", "hamming", "tukey" window functions. Default: "tukey".
    :return: the signal window function.
    """
    if name == "tukey":
        from .tukey import tukey_window
        return tukey_window
    elif name == "hann":
        return torch.hann_window
    elif name == "hamming":
        return torch.hamming_window
    else:
        raise ValueError(f"Invalid window function, but got {name}.")
