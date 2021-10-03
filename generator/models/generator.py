import torch
from torch import nn

from ..config.data import latent_vector_size


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Conv' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif layer_name == 'Linear':
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif 'Norm' in layer_name:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.expand_latent_vector = nn.Sequential(
            nn.Linear(latent_vector_size, 256, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1024, bias=False),
        )
        self.reshape = lambda x: x.view(-1, 1024, 1, 1)
        self.up_sample = nn.Sequential(
            # 1024 * 1 * 1
            nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 * 4 * 4
            nn.Upsample(scale_factor=4),
            nn.Conv2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 * 8 * 8
            nn.Upsample(scale_factor=4),
            nn.Conv2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 * 16 * 16
            nn.Upsample(scale_factor=4),
            nn.Conv2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 * 32 * 32
            nn.Upsample(scale_factor=4),
            nn.Conv2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 * 64 * 64
            nn.Upsample(scale_factor=4),
            nn.Conv2d(32, 3, 4, 2, 1, bias=False),
            # 3 * 128 * 128
            nn.Tanh()
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x = self.expand_latent_vector(x)
        x = self.reshape(x)
        x = self.up_sample(x)
        return x
