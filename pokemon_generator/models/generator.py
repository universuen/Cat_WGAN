from functools import partial

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Linear(256, 512),
        )

        self.reshape = lambda x: x.view(-1, 512, 1, 1)

        current_size = [4, 4]

        modules = [
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=output_size,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            # nn.LayerNorm([output_size, *current_size]),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(0.2, True),
        ]

        num_channels = output_size // 2
        while num_channels != 4:
            current_size = [i * 2 for i in current_size]

            modules.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels=num_channels * 2,
                        out_channels=num_channels,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_channels),
                    nn.LeakyReLU(0.2, True),
                ]
            )
            num_channels //= 2

        self.transpose_conv_layers = nn.Sequential(
            *modules,
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=3,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.reshape(x)
        x = self.transpose_conv_layers(x)
        return x
