import torch
from torch import nn

from .._utils import cal_output_size, cal_conv2d_output_size


class Discriminator(nn.Module):
    def __init__(
            self,
            input_size: int
    ):
        super().__init__()

        self.conv_pool_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.LayerNorm(
                [
                    32,
                    *cal_conv2d_output_size(
                        input_size=input_size,
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                ]
            ),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(
                in_channels=32,
                out_channels=128,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.LayerNorm(
                [
                    128,
                    *cal_conv2d_output_size(
                        input_size=cal_conv2d_output_size(
                            input_size=input_size,
                            kernel_size=(4, 4),
                            stride=(2, 2),
                            padding=(1, 1),
                        ),
                        kernel_size=(4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                ]
            ),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.LeakyReLU(0.2, True),
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(
                in_features=cal_output_size(
                    (1, 3, input_size, input_size),
                    self.conv_pool_layers
                ),
                out_features=512
            ),
            nn.Linear(512, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_pool_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x
