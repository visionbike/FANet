import multiprocessing as multproc
from functools import partial
import numpy as np

__all__ = ["roll_window"]


def roll_window(x: np.ndarray | list[np.ndarray],
                step_size: int = 5,
                window_size: int = 52,
                multiproc: bool = True) -> np.ndarray | list[np.ndarray]:
    """

    :param x: the signal in shape of (L, C) or a list of signal windows.
    :param step_size: the step size for rolling window. Default: 5.
    :param window_size: the kernel size. Default: 52.
    :param multiproc: whether to use multiprocessing for processing data, otherwise use multithreading instead.
    :return: the list of windows in shape of (N, window_size, C).
    """
    def _roll_window(x_: np.ndarray, s_, k_):
        return np.dstack([x_[i: i + x_.shape[0] - k_ + 1: s_] for i in range(k_)])

    if isinstance(x, np.ndarray):
        return _roll_window(x, step_size, window_size)
    elif isinstance(x, list):
        if multiproc:
            num_workers = multproc.cpu_count() - 1
            num_samples = len(x)
            chunk_size = num_samples // num_workers if num_samples > num_workers else num_samples
            with multproc.Pool(processes=num_workers) as p:
                z = list(
                    p.imap(partial(roll_window, step_size=step_size, window_size=window_size), x, chunksize=chunk_size))
        else:
            z = list(map(partial(roll_window, step_size=step_size, window_size=window_size), x))
        return z
    else:
        TypeError(f"Invalid data type of input, got {type(x)}.")
