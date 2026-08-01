import numpy as np
import torch

from typing import Union


TensorLike = Union[np.ndarray, torch.Tensor]


def _get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _as_tensor(
    x: TensorLike, 
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x
        t = t.to(device=device, dtype=dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)


def _check_tensor(obj):
    if obj == None:
        raise ValueError(
            f"expected `torch.Tensor`, got `None`"
        )
    if not isinstance(obj, torch.Tensor):
        raise ValueError(
            f"expected `torch.Tensor`, got `{type(obj)}`"
        )