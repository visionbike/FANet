import multiprocessing as multproc
from functools import partial
import numpy as np
from scipy import signal as scig

__all__ = [
    "filter_DC",
    "filter_highpass_butterworth",
    "filter_lowpass_butterworth",
    "filter_moving_average"
]


def filter_lowpass_butterworth(x: np.ndarray | list[np.ndarray],
                               cutoff: float = 2.0,
                               fs: float = 200.0,
                               order: int = 4,
                               zero_phase: bool = True,
                               multiproc: bool = True) -> np.ndarray | list[np.ndarray]:
    """
    Butterworth IIR low-pass filter function.

    :param x: the signal in shape of (L, C) or a list of signal windows.
    :param cutoff: the cutoff frequency. Default: 2.0.
    :param fs: the sampling rate. Default: 200.0.
    :param order: the filter's order. Default: 4.
    :param zero_phase: whether to use zero phase filter for offline data processing. Default: True.
    :param multiproc: whether to use multiprocessing to process data, otherwise, to use multithreading. Default: True.
    :return: the processed signal.
    """
    def _filter_lowpass_butterworth(x_: np.ndarray, cutoff_: float, order_: int, zero_phase_: bool):
        nyq = 0.5 * fs  # nyquist frequency
        cutoff_ = cutoff_ / nyq
        sos = scig.butter(N=order_, Wn=cutoff_, btype="lowpass", analog=False, output="sos")
        return scig.sosfiltfilt(sos, x_, axis=0) if zero_phase_ else scig.sosfilt(sos, x_, axis=0)

    if isinstance(x, np.ndarray):
        return _filter_lowpass_butterworth(x, cutoff, order, zero_phase)
    elif isinstance(x, list):
        if multiproc:
            num_workers = multproc.cpu_count() - 1
            num_samples = len(x)
            chunk_size = num_samples // num_workers if num_samples > num_workers else num_samples
            with multproc.Pool(processes=num_workers) as p:
                z = list(p.imap(partial(_filter_lowpass_butterworth, cutoff=cutoff, fs=fs, order=order, zero_phase=zero_phase), x, chunksize=chunk_size))
        else:
            z = list(map(partial(_filter_lowpass_butterworth, cutoff=cutoff, fs=fs, order=order, zero_phase=zero_phase), x))
        return z
    else:
        raise TypeError(f"Invalid input type data, got {type(x)}.")


def filter_highpass_butterworth(x: np.ndarray | list[np.ndarray],
                                cutoff: float = 2.0,
                                fs: float = 200.0,
                                order: int = 4,
                                zero_phase: bool = True,
                                multiproc: bool = True) -> np.ndarray | list[np.ndarray]:
    """
    Butterworth IIR high-pass filter function.

    :param x: the signal in shape of (L, C) or a list of signal windows.
    :param cutoff: the cutoff frequency. Default: 2.0.
    :param fs: the sampling rate. Default: 200.0.
    :param order: the filter's order. Default: 4.
    :param zero_phase: whether to use zero phase filter for offline data processing. Default: True.
    :param multiproc: whether to use multiprocessing to process data, otherwise, to use multithreading. Default: True.
    :return: the processed signal.
    """
    def _filter_highpass_butterworth(x_: np.ndarray, cutoff_: float, order_: int, zero_phase_: bool):
        nyq = 0.5 * fs  # nyquist frequency
        cutoff_ = cutoff_ / nyq
        sos = scig.butter(N=order_, Wn=cutoff_, btype="highpass", analog=False, output="sos")
        return scig.sosfiltfilt(sos, x_, axis=0) if zero_phase_ else scig.sosfilt(sos, x_, axis=0)

    if isinstance(x, np.ndarray):
        return _filter_highpass_butterworth(x, cutoff, order, zero_phase)
    elif isinstance(x, list):
        if multiproc:
            num_workers = multproc.cpu_count() - 1
            num_samples = len(x)
            chunk_size = num_samples // num_workers if num_samples > num_workers else num_samples
            with multproc.Pool(processes=num_workers) as p:
                z = list(p.imap(partial(_filter_highpass_butterworth, cutoff=cutoff, fs=fs, order=order, zero_phase=zero_phase), x, chunksize=chunk_size))
        else:
            z = list(map(partial(_filter_highpass_butterworth, cutoff=cutoff, fs=fs, order=order, zero_phase=zero_phase), x))
        return z
    else:
        raise TypeError(f"Invalid input type data, got {type(x)}.")


def filter_DC(x: np.ndarray | list[np.ndarray], multiproc: bool = True) -> np.ndarray | list[np.ndarray]:
    """
    DC (zero-mean shifting) filter function.

    :param x: the signal in shape of (L, C) or a list of signal windows.
    :param multiproc: whether to use multiprocessing to process data, otherwise, to use multithreading. Default: True.
    :return: the processed signal.
    """
    def _filter_DC(x_: np.ndarray):
        return x_ - np.mean(x_, axis=0)

    if isinstance(x, np.ndarray):
        return _filter_DC(x)
    elif isinstance(x, list):
        if multiproc:
            num_workers = multproc.cpu_count() - 1
            num_samples = len(x)
            chunk_size = num_samples // num_workers if num_samples > num_workers else num_samples
            with multproc.Pool(processes=num_workers) as p:
                z = list(p.imap(filter_DC, x, chunksize=chunk_size))
        else:
            z = list(map(filter_DC, x))
        return z
    else:
        raise TypeError(f"Invalid input type data, got {type(x)}.")


def filter_moving_average(x: np.ndarray | list[np.ndarray], window_size: int = 3, multiproc: bool = True) -> np.ndarray | list[np.ndarray]:
    """
    Moving average filter function.

    :param x: the signal in shape of (L, C) or a list of signal windows.
    :param window_size: the kernel window size. Default: 3.
    :param multiproc: whether to use multiprocessing to process data, otherwise, to use multithreading. Default: True.
    :return: the processed signal.
    """
    def _filter_moving_average(x_: np.ndarray):
        k = np.ones(window_size) / window_size
        return np.vstack([np.convolve(x_[:, i], k, mode='valid') for i in range(x.shape[-1])]).T

    if isinstance(x, np.ndarray):
        return _filter_moving_average(x)
    elif isinstance(x, list):
        if multiproc:
            num_workers = multproc.cpu_count() - 1
            num_samples = len(x)
            chunk_size = num_samples // num_workers if num_samples > num_workers else num_samples
            with multproc.Pool(processes=num_workers) as p:
                z = list(p.imap(filter_DC, x, chunksize=chunk_size))
        else:
            z = list(map(filter_DC, x))
        return z
    else:
        raise TypeError(f"Invalid input type data, got {type(x)}.")
