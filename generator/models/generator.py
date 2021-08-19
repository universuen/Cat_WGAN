import torch
from torch import nn

from ..config.data import latent_vector_size
from ._sub_modules import ReshapeBlock, LinearBlock, UpSampleBlock


class Generator(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()

        self.model = nn.Sequential(
            LinearBlock(latent_vector_size, 1024, is_output=True),
            ReshapeBlock(-1, 1024, 1, 1),
            UpSampleBlock(1024, 512),
            UpSampleBlock(512, 256),
            UpSampleBlock(256, 128),
            UpSampleBlock(128, 64),
            UpSampleBlock(64, 32),
            UpSampleBlock(32, 16),
            UpSampleBlock(16, 3, is_output=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x
