import itertools
import torch

from typing import Tuple


def _cubic_bspline(u: torch.Tensor) -> torch.Tensor:
    a = u.abs()
    out = torch.zeros_like(u)

    m1 = a < 1.0
    m2 = (a >= 1.0) & (a < 2.0)

    out = torch.where(m1, (2.0 / 3.0) - a.pow(2) + 0.5 * a.pow(3), out)
    out = torch.where(m2, (2.0 - a).pow(3) / 6.0, out)

    return out


def _cubic_bspline_derivative(u):
    a = u.abs()
    out = torch.zeros_like(u)

    m1 = a < 1.0
    m2 = (a >= 1.0) & (a < 2.0)
    
    out = torch.where(m1, -2.0 * u + 1.5 * u * a, out)
    out = torch.where(m2, -(2.0 - a).pow(2) * 0.5 * u.sign(), out)
    
    return out


@torch.no_grad()
def _eval_stencil(
    x: torch.Tensor,
    origin: torch.Tensor,
    h: float,
    grid_shape: Tuple[int,...]
) -> torch.Tensor:
    """
    Sparse cubic B-spline stencil

    Returns:
        flat_idx:   [N, 4^D] flat control-point indices
        weights:    [N, 4^D] B-spline products
    """
    N, D = x.shape
    device = x.device

    u = (x - origin.unsqueeze(0)) / h
    u_floor = u.floor().long()

    offsets = torch.arange(-1, 3, device=device)
    u_diff = u.unsqueeze(3) - offsets.view(1, 1, 4)
    w = _cubic_bspline(u_diff)

    idx_per_dim = u_floor.unsqueeze(2) + offsets.view(1, 1, 4)
    grid_lims = torch.tensor(grid_shape, dtype=torch.long, device=device)
    idx_per_dim = idx_per_dim.clamp(min=0, max=grid_lims.view(1, D, 1) - 1)

    strides = torch.ones(D, dtype=torch.long, device=device)
    for d in range(1, D):
        strides[d] = strides[d - 1 ] * grid_shape[d - 1]
    flat_per_dim = idx_per_dim * strides.view(1, D, 1)

    combinations = torch.tensor(
        list(itertools.product(range(4), repeat=D)),
        dtype=torch.long,
        device=device
    )
    combinations = combinations.t().unsqueeze(0).expand(N, D, -1)

    flat_idx = flat_per_dim.gather(2, combinations).sum(dim=1)
    weights = w.gather(2, combinations).prod(dim=1)

    return flat_idx, weights


@torch.no_grad()
def _eval_stencil_derivative(
    x: torch.Tensor,
    origin: torch.Tensor,
    h: float,
    grid_shape: Tuple[int,...],
    axis: int = 0
) -> torch.Tensor:
    """
    Sparse cubic B-spline stencil

    Returns:
        flat_idx:   [N, 4^D] flat control-point indices
        weights:    [N, 4^D] B-spline products
    """
    N, D = x.shape
    device = x.device

    u = (x - origin.unsqueeze(0)) / h
    u_floor = u.floor().long()

    offsets = torch.arange(-1, 3, device=device)
    u_diff = u.unsqueeze(3) - offsets.view(1, 1, 4)
    
    w = _cubic_bspline(u_diff)
    dw = _cubic_bspline_derivative(u_diff) / h

    w = w.clone()
    w[:, axis, :] = dw[:, axis, :]

    idx_per_dim = u_floor.unsqueeze(2) + offsets.view(1, 1, 4)
    grid_lims = torch.tensor(grid_shape, dtype=torch.long, device=device)
    idx_per_dim = idx_per_dim.clamp(min=0, max=grid_lims.view(1, D, 1) - 1)

    strides = torch.ones(D, dtype=torch.long, device=device)
    for d in range(1, D):
        strides[d] = strides[d - 1 ] * grid_shape[d - 1]
    flat_per_dim = idx_per_dim * strides.view(1, D, 1)

    combinations = torch.tensor(
        list(itertools.product(range(4), repeat=D)),
        dtype=torch.long,
        device=device
    )
    combinations = combinations.t().unsqueeze(0).expand(N, D, -1)

    flat_idx = flat_per_dim.gather(2, combinations).sum(dim=1)
    weights = w.gather(2, combinations).prod(dim=1)

    return flat_idx, weights