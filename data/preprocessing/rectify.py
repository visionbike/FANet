import multiprocessing as multproc
import numpy as np

__all__ = ["rectify"]


def rectify(x: np.ndarray | list[np.ndarray], multiproc: bool = True) -> np.ndarray | list[np.ndarray]:
    """
    Rectifying function.

    :param x: the signal in shape of (L, C) or a list of signal windows.
    :param multiproc: whether to use multiprocessing to process list of windows, otherwise, to use multithreading. Default: True.
    :return: the first appearance value or the first appearance array.
    """
    def _rectify(x_: np.ndarray):
        return np.abs(x_)

    if isinstance(x, np.ndarray):
        return _rectify(x)
    elif isinstance(x, list):
        if multiproc:
            num_workers = multproc.cpu_count() - 1
            num_samples = len(x)
            chunk_size = num_samples // num_workers if num_samples > num_workers else num_samples
            with multproc.Pool(processes=num_workers) as p:
                z = list(p.imap(rectify, x, chunksize=chunk_size))
        else:
            z = list(map(rectify, x))
        return z
    else:
        raise TypeError(f"Invalid data type of input, got {type(x)}.")
