#%%
import numpy as np

x = np.ones((3, 1, 5))
y = np.ones((1, 4, 5))
print(f"{x=}")
print(f"{y=}")

z = x + y

#%%

x = np.ones((8, 2, 6))
y = np.ones((8, 2))
# print(f"{x=}")

# z = x + y

print(len(x.shape))

#%%
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

Arr = np.ndarray
grad_tracking_enabled = True

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part4_backprop"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part4_backprop.tests as tests
from part4_backprop.utils import get_mnist, visualize
from plotly_utils import line


def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by
        broadcasting rules.
    """
    # Step 1: sum and remove prepended dims, so both arrays have same number of dims
    n_dims_to_sum = len(broadcasted.shape) - len(original.shape)
    broadcasted = broadcasted.sum(axis=tuple(range(n_dims_to_sum)))

    # Step 2: sum over dims which were originally 1 (but don't remove them)
    dims_to_sum = tuple([i for i, (o, b) in enumerate(zip(original.shape, broadcasted.shape)) if o == 1 and b > 1])
    broadcasted = broadcasted.sum(axis=dims_to_sum, keepdims=True)

    assert broadcasted.shape == original.shape
    return broadcasted


#%% 