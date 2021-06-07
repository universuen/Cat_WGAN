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
        self._output_size = output_size
        modules = [
            nn.ConvTranspose2d(
                in_channels=input_size,
                out_channels=output_size * 2,
                kernel_size=(2, 2),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.BatchNorm2d(output_size * 2),
            nn.ReLU(True),
        ]
        num_channels = output_size
        while num_channels != 4:
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
                nn.BatchNorm2d(num_channels),
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
