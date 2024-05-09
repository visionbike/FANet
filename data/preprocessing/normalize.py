import multiprocessing as multproc
from functools import partial
import numpy as np

__all__ = ["norm_minmax"]


def norm_minmax(x: np.ndarray | list[np.ndarray],
                minimum: np.ndarray = None,
                maximum: np.ndarray = None,
                multiproc: bool = True) -> list[np.ndarray]:
    """
    Min-max normalization function.

    :param x: the signal in shape of (L, C) or a list of signal windows.
    :param minimum: the minimum value. Default: None.
    :param maximum: the maximum value. Default: None.
    :param multiproc: whether to use multiprocessing to process list of windows, otherwise, to use multithreading. Default: True.
    :return: the first appearance value or the first appearance array.
    """
    def _norm_minmax(x_: np.ndarray, min_: np.ndarray = None, max_: np.ndarray = None):
        x_min = np.min(x, axis=0) if min_ is None else min_
        x_max = np.max(x, axis=0) if max_ is None else max_
        return (x_ - x_min) / (x_max - x_min)

    if isinstance(x, np.ndarray):
        return _norm_minmax(x, minimum, maximum)
    elif isinstance(x, list):
        if multiproc:
            num_workers = multproc.cpu_count() - 1
            num_samples = len(x)
            chunk_size = num_samples // num_workers if num_samples > num_workers else num_samples
            with multproc.Pool(processes=num_workers) as p:
                z = list(p.imap(partial(norm_minmax, minimum=minimum, maximum=maximum), x, chunksize=chunk_size))
        else:
            z = list(map(partial(norm_minmax, minimum=minimum, maximum=maximum), x))
        return z
    else:
        raise TypeError(f"Invalid data type of input, got {type(x)}.")
