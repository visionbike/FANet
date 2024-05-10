import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

__all__ = ["MLPBaseline"]


class MLPBaseline(nn.Module):
    """
    The implementation of the baseline neural network.

    Reference:
    * sEMG Gesture Recognition with a Simple Model of Attention (https://arxiv.org/abs/2006.03645)
    """

    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 num_classes: int,
                 norm_layer: nn.Module = nn.LayerNorm,
                 act_layer: nn.Module = nn.Mish,
                 att_layer: nn.Module = nn.Identity,
                 att_kwargs: dict = None,
                 drop_rate: float = 0.36):
        """
        :param in_channels: the number of channels in input tensor.
        :param mid_channels: the number of channels in the inner tensor.
        :param num_classes: the number of classes.
        :param norm_layer: the normalization layer. Default: nn.LayerNorm.
        :param act_layer: the activation layer. Default: nn.Mish.
        :param att_layer: the activation layer. Default: nn.Identity.
        :param att_kwargs: the attention layer arguments. Default: None.
        :param drop_rate: the dropout ratio. Default: 0.36.
        """
        super().__init__()
        # projection block
        self.proj = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            norm_layer(mid_channels),
            act_layer()
        )
        # attention layer
        self.att = att_layer(**att_kwargs) if isinstance(att_layer, nn.Identity) else \
            nn.Sequential(
                Rearrange("b l c -> b c l"),
                att_layer(**att_kwargs),
                Rearrange("b c l -> b l c")
            )
        self.norm = norm_layer(mid_channels)
        self.pool = Reduce("b l c -> b c", "mean")
        # classifier head
        fc_in_channels = mid_channels
        list_fc_mid_channels = [500, 500, 2000]
        fc_modules = []
        for fc_mid_channels in list_fc_mid_channels:
            fc_modules += [
                nn.Linear(fc_in_channels, fc_mid_channels),
                norm_layer(fc_mid_channels),
                act_layer(),
                nn.Dropout(drop_rate),
            ]
            fc_in_channels = fc_mid_channels
        fc_modules += [nn.Linear(fc_in_channels, num_classes, False)]
        self.head = nn.Sequential(*fc_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, L, C).
        :return: the output tensor in shape of (N, num_classes).
        """
        # expansion block
        z = self.proj(x)
        # attention block
        z = self.att(z)
        # temporal pooling
        z = self.pool(self.norm(z))
        # classifier head
        z = self.head(z)
        return z
