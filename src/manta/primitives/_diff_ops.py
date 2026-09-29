import torch

from typing import Literal, Tuple


def _diff1(L: int, device: str = "cpu"):
    """
    First-order difference matrix [L-1, L] (dense)
    """
    D =  torch.zeros(max(L - 1, 0), L, device=device)
    if L > 1:
        idx = torch.arange(L - 1, device=device)
        D[idx, idx]     = -1.0
        D[idx, idx + 1] =  1.0
    return D


def _diff2(L: int, device: str = "cpu"):
    """
    Second-order difference matrix [L - 2, L] (dense)
    """
    D = torch.zeros(max(L - 2, 0), L, device=device)
    if L > 2:
        idx = torch.arange(L - 2, device=device)
        D[idx, idx]     =  1.0
        D[idx, idx + 1] = -2.0
        D[idx, idx + 2] = 1.0
    return D


def _build_gmrf(
    grid_shape: Tuple[int,...],
    shape: Literal["bending", "membrane", "combined"] = "bending",
    device: str = "cpu"
) -> torch.Tensor:
    """
    Gaussian-Markov random field precision L0 (dense) on an
    N-D regular grid via Kronecker products
    """
    D = len(grid_shape)
    P = 1
    for L in grid_shape:
        P *= L

    L0 = torch.zeros(P, P)
    for axis in range(D):
        if shape in ("membrane", "combined"):
            D1 = _diff1(grid_shape[axis])
            mats = [D1.T @ D1 if d == axis else torch.eye(grid_shape[D]) for d in range(D)]
            op = mats[0]

            for d in range(1, D):
                op = torch.kron(op, mats[d])

            L0 = L0 + op

        if shape in ("bending", "combined"):
            D2 = _diff2(grid_shape[axis])
            mats = [D2.T @ D2 if d == axis else torch.eye(grid_shape[d]) for d in range(D)]
            op = mats[0]

            for d in range(1, D):
                op = torch.kron(op, mats[d])

            L0 = L0 + op

    L0 = L0 + 1e-5 * torch.eye(P)
    return L0


def _apply_diff_axis(
    mu: torch.Tensor,
    grid_shape: Tuple[int,...],
    axis: int = 0,
    order: int = 1
) -> torch.Tensor:
    """
    Apply the first- or second-order finite difference operator along 
    `axis` on a grid-shaped vector field `mu` (flattened, [P])
    
    Returns a flat vector whose length is the number of edges
    """
    shape = list(grid_shape)
    mu = mu.view(*shape)

    # Move target axis to the end
    perm = list(range(len(shape)))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    mu = mu.permute(perm).contiguous()

    if order == 1:
        d = mu[..., 1:] - mu[..., :-1]
    else:
        d = mu[..., 2:] - 2.0 * mu[..., 1:-1] + mu[..., :-2]
    return d.reshape(-1)


def _apply_diff_axis_sq_variance(
    v: torch.Tensor,
    grid_shape: Tuple[int,...],
    axis: int = 0,
    order: int = 1
) -> torch.Tensor:
    """
    Apply the squared difference operator (sum of squared finite
    difference coefficients) along `axis` to a per-control-point
    variance vector `v`
    """
    shape = list(grid_shape)
    v = v.view(*shape)

    perm = list(range(len(shape)))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    v = v.permute(perm).contiguous()

    if order == 1:
        d2 = v[..., 1:] + v[..., :-1]
    else:
        d2 = v[..., 2:] + 4.0 * v[..., 1:-1] + v[..., :-2]
    return d2.reshape(-1)