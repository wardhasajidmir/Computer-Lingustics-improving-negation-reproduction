"""Helper functions written by the group."""
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Fix all random seeds so that runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
