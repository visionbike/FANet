import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

__all__ = ["MLPTransformer"]


class MLPBlock(nn.Module):
    """
    Residual MLP block implementation.
    """

    def __init__(self,
                 in_channels: int,
                 in_dims: int,
                 mid_channels: int,
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.Mish,
                 att_layer: nn.Module = nn.Identity,
                 att_kwargs: dict = None,
                 drop_rate: float = 0.0,
                 init_values: float = 1e-4):
        """
        :param in_channels: the number of channels in input tensor.
        :param in_dims: the length of the input tensor.
        :param mid_channels: the number of channels in the inner tensor.
        :param norm_layer: the normalization layer. Default: nn.LayerNorm.
        :param act_layer: the activation function. Default: nn.Mish.
        :param att_layer: the attention layer. Default: nn.Identity.
        :param att_kwargs: the attention layer arguments. Default: None.
        :param drop_rate: the dropout ratio. Default: 0.0.
        :param init_values: the initialized value for the layer scale normalization. Default: 1e-4.
        """
        if init_values < 0.0 or init_values > 1.0:
            raise ValueError(f"Invalid initial value, but got {init_values}.")
        #
        super().__init__()
        #
        self.pre_norm = norm_layer(in_channels)
        # attention layer
        self.att = nn.Sequential(
            Rearrange("b l c -> b c l"),
            nn.Linear(in_dims, in_dims) if isinstance(att_layer, nn.Identity) else att_layer(**att_kwargs),
            Rearrange("b c l -> b l c"),
        )
        self.post_norm = norm_layer(in_channels)
        # head MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            act_layer(),
            nn.Dropout(drop_rate),
            nn.Linear(mid_channels, in_channels),
            nn.Dropout(drop_rate)
        )
        #
        self.gamma = 1.0 if init_values == 0 else nn.Parameter(init_values * torch.ones((in_channels,)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, L, C).
        :return: the output tensor in shape of (N, L, C).
        """
        return x + self.gamma * self.mlp(self.post_norm(self.att(self.pre_norm(x))))


class MLPTransformer(nn.Module):
    """
    The MLPTransformer implementation.

    Reference:
    * Global Filter Networks for Image Classification (https://arxiv.org/abs/2107.00645)
    """

    def __init__(self,
                 in_channels: int,
                 in_dims: int,
                 mid_channels: int,
                 mlp_channels: int,
                 num_classes: int,
                 num_blocks: int = 4,
                 drop_rate: float = 0.0,
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.Mish,
                 att_layer: nn.Module = nn.Identity,
                 att_kwargs: dict = None,
                 init_values: float = 1e-4,
                 norm_last: bool = True,
                 avg_pooling: bool = True):
        """
        :param in_channels: the number of channels in input tensor.
        :param in_dims: the length of the input tensor.
        :param mid_channels: the number of channels in the inner tensor.
        :param mlp_channels: the number of channels in the mlp layer.
        :param num_classes: the number of classes.
        :param num_blocks: the number of residual MLPBlock. Default: 4.
        :param drop_rate: the dropout ratio. Default: 0.0.
        :param norm_layer: the normalization layer. Default: nn.LayerNorm.
        :param act_layer: the activation function. Default: nn.Mish.
        :param att_layer: the attention layer. Default: nn.Identity.
        :param att_kwargs: the attention layer arguments. Default: None.
        :param init_values: the initialized value for the layer scale normalization. Default: 1e-4.
        :param norm_last:  whether to apply normalization before pooling. Default: True.
        :param avg_pooling: whether to apply average pooling. Default: True.
        """
        super().__init__()
        # project layer
        self.proj = nn.Linear(in_channels, mid_channels)
        # residual blocks
        self.res_blocks = nn.ModuleList([
            MLPBlock(mid_channels, in_dims, mlp_channels, norm_layer, act_layer, att_layer, att_kwargs, drop_rate, init_values)
            for _ in range(num_blocks)
        ])
        # pooling layer
        self.norm_last = norm_layer(mid_channels) if norm_last else nn.Identity()
        self.pool = Reduce("b l c -> b c", "mean") if avg_pooling else Reduce("b l c -> b c", "sum")
        # classifier layer
        self.head = nn.Sequential(
            nn.Linear(mid_channels, num_classes, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, L, C).
        :return: the output tensor in shape of (N, num_classes).
        """
        # projection block
        z = self.proj(x)
        # residual blocks
        for block in self.res_blocks:
            z = block(z)
        # temporal pooling
        z = self.pool(self.norm_last(z))
        # classifier head
        z = self.head(z)
        return z
