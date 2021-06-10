import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int
    ):
        super().__init__()
        assert output_size % 2 == 0, "output_size should be an even integer."

        current_size = [4, 4]

        modules = [
            nn.ConvTranspose2d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.LayerNorm([output_size, *current_size]),
            nn.ReLU(True),
        ]

        num_channels = output_size // 2
        while num_channels != 4:

            current_size = [i * 2 for i in current_size]

            modules.append(
                nn.ConvTranspose2d(
                    in_channels=num_channels * 2,
                    out_channels=num_channels,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                ),
            )
            modules.append(
                nn.LayerNorm([num_channels, *current_size]),
            )
            modules.append(
                nn.ReLU(True),
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
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        x = self.transpose_conv_layers(x)
        return x
