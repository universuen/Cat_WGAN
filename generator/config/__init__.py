"""
This package contains the configuration for each module in this project.
"""
import random

import torch.backends.cudnn
import torch
import numpy as np

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

seed = 123

if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
