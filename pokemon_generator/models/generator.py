import torch
from torch import nn

from .. import config


class Generator(nn.Module):
    def __init__(
            self,
            input_size: int = config.data.latent_vector_size,
            output_size: int = config.data.image_size,
    ):
        super().__init__()

        self.reshape = lambda x: x.view(-1, config.data.latent_vector_size, 1, 1)

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

        self.transpose_conv = nn.Sequential(
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
        x = self.reshape(x)
        x = self.transpose_conv(x)
        return x
