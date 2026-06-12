# %%

import torch as t
from jaxtyping import Float, Int
from torch import Tensor

a = t.zeros((2, 3))
print(a.shape)
b = a[None, :]
print(b.shape)
