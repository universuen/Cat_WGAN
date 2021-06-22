import torch
from torch import nn

from .._utils import cal_conv2d_output_size
from .. import config


class Discriminator(nn.Module):
    def __init__(
            self,
            input_size=config.data.image_size,
    ):
        super().__init__()

        current_size = cal_conv2d_output_size(
            input_size=input_size,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1),
        )

        modules = [
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.LayerNorm([64, *current_size]),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            )
        ]

        current_channels_num = 64

        while current_size != (4, 4):
            current_size = cal_conv2d_output_size(
                input_size=current_size,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            )
            modules.extend(
                [
                    nn.Conv2d(
                        in_channels=current_channels_num,
                        out_channels=current_channels_num * 2,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                        bias=False,
                    ),
                    nn.LayerNorm([current_channels_num * 2, *current_size]),
                    nn.LeakyReLU(
                        negative_slope=0.2,
                        inplace=True,
                    )
                ]
            )
            current_channels_num *= 2

        self.conv = nn.Sequential(
            *modules,
            nn.Conv2d(
                in_channels=current_channels_num,
                out_channels=current_channels_num * 2,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(current_channels_num * 2, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
