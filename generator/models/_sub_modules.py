import torch
from torch import nn


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if layer_name == 'Conv2d':
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif layer_name == 'Linear':
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)
    elif 'Norm' in layer_name:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


class ReshapeBlock(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        return x.view(*self.shape)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_output=False):
        super().__init__()
        if is_output:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.block(x)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, output_size: tuple[int, int] = None, is_output=False):
        super().__init__()
        if is_output:
            self.block = nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
                nn.LayerNorm([out_channels, *output_size]),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.block(x)


class LinearBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, is_output=False):
        super().__init__()
        if is_output:
            self.block = nn.Linear(input_size, output_size)
        else:
            self.block = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.block(x)


'''
The following block was tested but made the result worse. Maybe it will be helpful in the future work.
'''

#
# class ResBlock(nn.Module):
#     def __init__(self, n_channels, norm: nn.Module):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
#             norm,
#             nn.ReLU(inplace=True),
#             nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
#             norm,
#         )
#         self.apply(init_weights)
#
#     def forward(self, x: torch.Tensor):
#         return x, self.conv_block(x), self.conv_block(x) + x
