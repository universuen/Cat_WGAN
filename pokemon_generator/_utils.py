from typing import Iterable

import torch
from torch import nn


def get_output_size(
        input_size: Iterable[int],
        module: nn.Module
) -> int:
    x = torch.randn(*input_size)
    with torch.no_grad():
        y = module(x)
    return len(y.flatten())
