import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import lightning as l

__all__ = [
    "generate_seed_through_os",
    "get_num_devices",
    "setup_seed",
]


def generate_seed_through_os() -> int:
    """
    Usually the best random sample you could get in any programming language is generated through the operating system.
    In Python, you can use the os module.

    Reference:
    * https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967

    :return: the random seed value generated from os module.
    """
    RANDOM_SIZE = 4
    # return a string of size random bytes suitable for cryptographic use
    data_random = os.urandom(RANDOM_SIZE)
    seed_random = int.from_bytes(data_random, byteorder="big")
    return seed_random


def setup_seed(seed: int, use_cuda: bool = True, use_cudnn: bool = True, cudnn_benchmark: bool = False, cudnn_deterministic: bool = False) -> None:
    """
    Set up seed and benchmark.

    :param seed: the input random seed value.
    :param use_cuda: whether to use CUDA. Default: True.
    :param use_cudnn: whether to use CUDNN. Default: True.
    :param cudnn_benchmark: whether to use CUDNN benchmark. Default: False.
    :param cudnn_deterministic: whether to use CUDNN deterministic. Default: False
    """
    seed = generate_seed_through_os() if seed is None else seed
    # if any of the libraries or code rely on NumPy seed the global NumPy RNG.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    l.seed_everything(seed, workers=True)
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        if use_cudnn:
            # Causes cuDNN to deterministically select an algorithm,
            # possibly at the cost of reduced performance
            # (the algorithm itself may be nondeterministic).
            cudnn.benchmark = cudnn_benchmark
            # Causes cuDNN to use a deterministic convolution algorithm, but may slow down performance.
            # It will not guarantee that your training process is deterministic
            # if you are using other libraries that may use nondeterministic algorithms
            cudnn.deterministic = cudnn_deterministic
        else:
            # Controls whether cuDNN is enabled or not.
            # If you want to enable cuDNN, set it to True.
            cudnn.enabled = use_cudnn


def get_num_devices(use_cuda: bool = True) -> int:
    """
    Get the number of computing devices.

    :param use_cuda: whether to use CUDA. Default: True.
    :return: the number of computing devices.
    """
    if use_cuda:
        return torch.cuda.device_count()
    return os.cpu_count()
