import torch

__all__ = ["check_window_function"]


def check_window_function(func_name: str, M: int, dtype: torch.dtype, layout: torch.layout) -> None:
    """
    Performs common checks for all the defined windows. This function should be called before computing any window.

    :param func_name: the name of the window function.
    :param M: the length of the window.
    :param dtype: the desired data type of returned tensor.
    :param layout: the desired layout of returned tensor.
    """
    if M < 0:
        raise ValueError(f"func_name = '{func_name}' requires non-negative window length, but got {M}.")
    if layout is not torch.strided:
        raise ValueError(f"func_name = '{func_name}' is implemented for strided tensors only, but got {layout}.")
    if not (dtype in [torch.float32, torch.float64]):
        raise ValueError(f"func_name = '{func_name}' expects float32 or float64 dtypes, but got {dtype}.")
