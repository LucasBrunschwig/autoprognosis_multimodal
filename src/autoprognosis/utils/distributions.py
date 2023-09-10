# stdlib
import random

# third party
import numpy as np

# autoprognosis absolute
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        import torch

        break
    except ImportError:
        depends = ["torch"]
        install(depends)


def enable_reproducible_results(seed: int = 0) -> None:
    """Set fixed seed for all the libraries"""
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
