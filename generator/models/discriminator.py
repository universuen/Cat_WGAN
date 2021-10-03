import torch
from torch import nn


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Conv' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif layer_name == 'Linear':
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif 'Norm' in layer_name:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # 3 * 128 * 128
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.GroupNorm(2, 32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 * 64 * 64
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 * 32 * 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 * 16 * 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 * 8 * 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 * 4 * 4
            nn.Conv2d(512, 1024, 4, 1, 0, bias=False),
            # 1024 * 1 * 1
            nn.Flatten(),
            nn.Linear(1024, 32, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1, bias=False),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        return self.model(x)
