"""
Reference:
 - ZerO Initialization: Initializing Neural Networks with only Zeros and Ones (https://arxiv.org/abs/2110.12661)
 - https://github.com/jiaweizzhao/ZerO-initialization/blob/main/example_mnist.ipynb
 - https://github.com/jiaweizzhao/ZerO-initialization/issues/1
"""
import math
from scipy.linalg import hadamard
import torch
from torch import nn

__all__ = ["init_zero_linear",
           "init_zero_conv1d",
           "init_zero_conv2d"]


@torch.no_grad()
def init_zero_linear(weight: nn.Parameter | torch.Tensor) -> None:
    """
    ZerO initialization for fully connected layer (algorithm I in the paper).

    :param weight: the linear layer's weights.
    """
    m = weight.data.size(0)
    n = weight.data.size(1)

    if m <= n:
        weight.data = nn.init.eye_(torch.empty(m, n))
    else:
        m_clog = math.ceil(math.log2(m))
        p = 2 ** m_clog
        weight.data = torch.matmul(
            torch.torch.matmul(nn.init.eye_(torch.empty(m, p)), torch.tensor(hadamard(p), dtype=torch.float) / torch.sqrt(torch.tensor(p))),
            nn.init.eye_(torch.empty(p, n))
        )


@torch.no_grad()
def init_zero_conv1d(weight: nn.Parameter | torch.Tensor) -> None:
    """
    ZerO initialization for 1D convolutional layer (Algorithm II in the paper).

    :param weight: the 1D convolution layer's weights.
    """
    m = weight.data.size(0)
    n = weight.data.size(1)
    k = weight.data.size(2)
    k_index = int(math.floor(k / 2))

    if m <= n:
        weight.data[:, :, k_index] = nn.init.eye_(torch.empty(m, n))
    else:
        m_clog = math.ceil(math.log2(m))
        p = 2 ** m_clog
        weight.data[:, :, k_index] = torch.matmul(
            torch.matmul(nn.init.eye_(torch.empty(m, p)), torch.tensor(hadamard(p, dtype=float)) / torch.sqrt(torch.tensor(p))),
            nn.init.eye_(torch.empty(p, n))
        )


@torch.no_grad()
def init_zero_conv2d(weight: nn.Parameter | torch.Tensor) -> None:
    """
    ZerO initialization for 2D convolutional layer (Algorithm II in the paper).

    :param weight: the 2D convolution layer's weights.
    """
    m = weight.data.size(0)
    n = weight.data.size(1)
    k = weight.data.size(2)
    t = weight.data.size(3)

    k_index, l_index = int(math.floor(k / 2)), int(math.floor(t / 2))
    if m <= n:
        weight.data[:, :, k_index, l_index] = nn.init.eye_(torch.empty(m, n))
    else:
        m_clog = math.ceil(math.log2(m))
        p = 2 ** m_clog
        weight.data[:, :, k_index, l_index] = torch.matmul(
            torch.matmul(nn.init.eye_(torch.empty(m, p)), torch.tensor(hadamard(p, dtype=float)) / torch.sqrt(torch.tensor(p))),
            nn.init.eye_(torch.empty(p, n))
        )
