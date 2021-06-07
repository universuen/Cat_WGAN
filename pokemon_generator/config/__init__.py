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

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
