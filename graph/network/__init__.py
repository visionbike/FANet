from typing import Type
from copy import deepcopy
import torch.nn as nn
from torch.nn import Identity, LayerNorm, BatchNorm1d, BatchNorm2d, ReLU, GELU, Mish
from .arch import *
from .layer import *
from .init import *

__all__ = ["get_network"]


def get_norm_layer(name: str = "none") -> Type[Identity | LayerNorm | BatchNorm1d | BatchNorm2d] | nn.Module:
    """
    The function to return normalization layer.

    :param name: the normalization layer name. Default: "none".
    :return: the norm layer.
    """
    if name == "none":
        return Identity
    elif name == "ln":
        return LayerNorm
    elif name == "bn1d":
        return BatchNorm1d
    elif name == "bn2d":
        return BatchNorm2d
    else:
        raise ValueError(f"Invalid normalization layer name, but got {name}.")


def get_act_layer(name: str = "none") -> Type[Identity | ReLU | GELU | Mish] | nn.Module:
    """
    The function to return activation layer.

    :param name: the activation layer name. Default: "none".
    :return: the activation layer.
    """
    if name == "none":
        act_layer = Identity
    elif name == "relu":
        act_layer = ReLU
    elif name == "gelu":
        act_layer = GELU
    elif name == "mish":
        act_layer = Mish
    else:
        raise ValueError(f"Invalid activation layer name, but got {name}.")
    return act_layer


def get_attn_layer(name: str) -> Type[Identity | FFTFilter | FFTFilter2 | STFTFilter | STFTFilter2| DCTFilter | SDCTFilter] | nn.Module:
    """
    The function to return attention layer.

    :param name: the attention layer name. Default: "none".
    :return: the attention layer.
    """
    if name == "none":
        att_layer = nn.Identity
    elif name == "attn_fft":
        att_layer = FFTFilter
    elif name == "attn_fft2":
        att_layer = FFTFilter2
    elif name == "attn_stft":
        att_layer = STFTFilter
    elif name == "attn_stft2":
        att_layer = STFTFilter2
    elif name == "attn_dct":
        att_layer = DCTFilter
    elif name == "attn_sdct":
        att_layer = SDCTFilter
    else:
        raise ValueError(f"Invalid attention layer name, but got {name}.")
    return att_layer


def get_arch(name: str = "mlp_baseline") -> Type[MLPBaseline | MLPTransformer] | nn.Module:
    """
    The function to return the network architecture.

    :param name: the architecture name. Default: "mlp_baseline".
    :return: the network architecture.
    """
    if name == "mlp_baseline":
        net = MLPBaseline
    elif name == "mlp_transformer":
        net = MLPTransformer
    else:
        raise ValueError(f"Invalid network module name, but got {name}.")
    return net


def get_network(**kwargs) -> nn.Module:
    """
    Get network module.

    :param kwargs: the network configuration.
    :return: the network module.
    """
    if "attn_kwargs" not in kwargs.keys():
        raise ValueError(f"Key not found: 'attn_kwargs'.")
    att_kwargs = deepcopy(kwargs.pop("attn_kwargs"))
    att_layer = get_attn_layer(att_kwargs.pop["name"])
    norm_layer = get_norm_layer(kwargs["norm"])
    act_layer = get_act_layer(kwargs["act_layer"])
    net = get_arch(kwargs["name"])
    model = net(norm_layer=norm_layer, act_layer=act_layer, att_layer=att_layer, att_kwargs=att_kwargs, **kwargs)
    model.apply(init_weights)
    return model
