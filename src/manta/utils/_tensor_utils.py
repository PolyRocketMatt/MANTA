import numpy as np
import torch

from typing import Union


TensorLike = Union[np.ndarray, torch.Tensor]

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def _get_device():
    return _DEVICE


def _set_device(device: str):
    _DEVICE = device


def _as_tensor(
    x: TensorLike, 
    dtype: torch.dtype,
    device: str = "cpu"
) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x
        t = t.to(device=device, dtype=dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)


def _from_tensor(
    x: TensorLike
) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _off_diag(x: torch.Tensor) -> torch.Tensor:
    N, M = x.shape
    if N != M:
        raise ValueError("expected square matrix")
    return x.flatten()[:-1].view(N-1, N+1)[:, 1:].flatten()