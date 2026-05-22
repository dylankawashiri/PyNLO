from typing import Any

import numpy as np
from torch import tensor, Tensor

def to_numpy(arr: Any) -> np.ndarray:
    getter = getattr(arr, "get", None)
    host_arr = getter() if callable(getter) else arr
    return np.asarray(host_arr)

def checker(func: Any) -> Any:
    """Prevents function call if input variables are the same as defined variables."""
    def wrapper(self: Any, *args: Any, **kwargs: Any):
        if all(kwargs[kwarg] == getattr(self, kwarg, None) for kwarg in kwargs):
            return None
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])
        return func(self, *args, **kwargs)
    return wrapper

def to_tensor(arr: Any) -> Tensor:
    return tensor(arr)