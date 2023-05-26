import torch
import torch.nn as nn
from typing import Union

class EnsembleConv2d(nn.Module):
    """
       Convolutional layer for ensemble models
    """
    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            ensemble_size: int = 1,
            stride: int = 1,
            padding: Union[str, int] = 0,
            dilation: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None) -> None:
        super(EnsembleConv2d, self).__init__()
        assert isinstance(ensemble_size, int) and ensemble_size > 0, "ensemble_size should be a positive integer"

        self.ensemble_size = ensemble_size
        self.in_channels = in_channels * self.ensemble_size
        self.out_channels = out_channels * self.ensemble_size
        self.layer = nn.Conv2d(self.in_channels, self.out_channels,
                               kernel_size, stride, padding, dilation, ensemble_size,
                               bias, padding_mode, device, dtype)

    def forward(self, input: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if input.shape[channel_dim] != self.in_channels:
            factor = self.in_channels // input.shape[channel_dim]
            input = input.repeat_interleave(factor, dim=channel_dim)
        return self.layer(input)

    def extra_repr(self) -> str:
        return f'ensemble_size={self.ensemble_size}, in_channels={self.in_channels}, out_channels={self.out_channels}'
