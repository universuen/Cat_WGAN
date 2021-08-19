import torch
from torch import nn

from ._sub_modules import DownSampleBlock, ReshapeBlock, LinearBlock


class Discriminator(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

        self.model = nn.Sequential(
            DownSampleBlock(3, 16, (64, 64)),
            DownSampleBlock(16, 32, (32, 32)),
            DownSampleBlock(32, 64, (16, 16)),
            DownSampleBlock(64, 128, (8, 8)),
            DownSampleBlock(128, 256, (4, 4)),
            DownSampleBlock(256, 512, (2, 2)),
            DownSampleBlock(512, 1024, is_output=True),
            ReshapeBlock(-1, 1024),
            LinearBlock(1024, 1, is_output=True),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
