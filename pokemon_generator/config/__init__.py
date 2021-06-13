"""
This package contains the configuration for each module in this project.
"""
import torch

from . import (
    data,
    path,
    logger,
    training,
)

device = 'auto'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
