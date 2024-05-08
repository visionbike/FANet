import multiprocessing as multproc
import numpy as np
import polars as pl

__all__ = [
    "get_first_appearance",
    "get_major_appearance"
]


def get_first_appearance(x: np.ndarray | list[np.ndarray], multiproc: bool = True) -> int | np.ndarray:
    """
    Get the first appearance of value in the window array.

    :param x: the window array or a list of windows.
    :param multiproc: whether to use multiprocessing to process list of windows, otherwise, to use multithreading. Default: True.
    :return: the first appearance value or the first appearance array.
    """
    def _get_first_appearance(x_: np.ndarray) -> int:
        return pl.LazyFrame(x_).unique(maintain_order=True).collect().item(0, 0)

    if isinstance(x, np.ndarray):
        return _get_first_appearance(x)
    elif isinstance(x, list):
        if multiproc:
            num_workers = multproc.cpu_count() - 1
            num_samples = len(x)
            chunk_size = num_samples // num_workers if num_samples > num_workers else num_samples
            with multproc.Pool(processes=num_workers) as p:
                z = list(p.imap(_get_first_appearance, x, chunksize=chunk_size))
        else:
            z = list(map(_get_first_appearance, x))
        return np.asarray(z)
    else:
        raise TypeError(f"Invalid input type data, got {type(x)}.")


def get_major_appearance(x: np.ndarray | list[np.ndarray], multiproc: bool = True) -> int | np.ndarray:
    """
    Get the major appearance of value in the window array.

    :param x: the window array or a list of windows.
    :param multiproc: whether to use multiprocessing to process list of windows, otherwise, to use multithreading. Default: True.
    :return: the major appearance value or the first appearance array.
    """

    def _get_major_appearance(x_: np.ndarray) -> int:
        return pl.LazyFrame(x_).select(pl.col("colum_0").value_counts(sort=True)).unnest("column_0").collect().item(0, 0)

    if isinstance(x, np.ndarray):
        return _get_major_appearance(x)
    elif isinstance(x, list):
        if multiproc:
            num_workers = multproc.cpu_count() - 1
            num_samples = len(x)
            chunk_size = num_samples // num_workers if num_samples > num_workers else num_samples
            with multproc.Pool(processes=num_workers) as p:
                z = list(p.imap(get_major_appearance, x, chunksize=chunk_size))
        else:
            z = list(map(get_major_appearance, x))
        return np.asarray(z)
    else:
        raise TypeError(f"Invalid input type data, got {type(x)}.")
