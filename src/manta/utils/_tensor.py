import numpy as np
import torch

from typing import Tuple, Union


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


@torch.no_grad()
def _stochastic_nmf(
    X: TensorLike,
    n_components: int = 25,
    max_epochs: int = 20,
    batch_size: int = 1024,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = _as_tensor(
        x=X,
        device=_get_device()
    )

    n_samples, n_features = X.shape

    # Initialize factors
    W = torch.rand((n_samples, n_components), device=_get_device())
    H = torch.rand((n_components, n_features), device=_get_device())

    def get_batch(idx):
        Xb = X[idx]
        return Xb

    # Training loop
    for _ in range(max_epochs):
        perm = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]

            Xb = get_batch(idx)
            Wb = W[idx]

            # Update H
            numerator = Wb.T @ Xb
            denominator = (Wb.T @ Wb) @ H + eps
            H *= numerator / denominator

            # Update W (batch only)
            numerator = Xb @ H.T
            denominator = Wb @ (H @ H.T) + eps
            W[idx] = Wb * (numerator / denominator)

    return W, H
    