import torch

from typing import Dict, List, Literal, Tuple


def _strides(grid_shape: Tuple[int,...]) -> List[int]:
    D = len(grid_shape)
    s = [1] * D
    for i in range(D - 2, -1, 1):
        s[i] = s[i + 1] * grid_shape[i + 1]
    return s


def _diff1_edge_pairs(
    grid_shape: Tuple[int,...], 
    axis: int = 0, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (a, b) index pairs for first-order edges along `axis`.
    """
    D = len(grid_shape)
    strides = _strides(grid_shape)
    grids = torch.meshgrid(
        *[torch.arange(L, device=device) for L in grid_shape], indexing="ij"
    )
    mask = grids[axis] < grid_shape[axis] - 1

    flat = torch.zeros(grid_shape, dtype=torch.long, device=device)
    for i in range(D):
        flat = flat + grids[i] * strides[i]
    a = flat[mask]

    grids_plus = [g.clone() for g in grids]
    grids_plus[axis] = grids_plus[axis] + 1
    flat_plus = torch.zeros(grid_shape, dtype=torch.long, device=device)
    
    for i in range(D):
        flat_plus = flat_plus + grids_plus[i] * strides[i]
    b = flat_plus[mask]
    return a, b


def _diff2_edge_triples(
    grid_shape: Tuple[int,...], 
    axis: int = 0, 
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (a, b, c) triples for second-order edges along `axis`.
    """
    D = len(grid_shape)
    strides = _strides(grid_shape)
    grids = torch.meshgrid(
        *[torch.arange(L, device=device) for L in grid_shape], indexing="ij"
    )
    mask = grids[axis] < grid_shape[axis] - 2

    flat = torch.zeros(grid_shape, dtype=torch.long, device=device)
    for i in range(D):
        flat = flat + grids[i] * strides[i]
    a = flat[mask]

    def _shift(k):
        gp = [g.clone() for g in grids]
        gp[axis] = gp[axis] + k
        f = torch.zeros(grid_shape, dtype=torch.long, device=device)
        for i in range(D):
            f = f + gp[i] * strides[i]
        return f[mask]

    b = _shift(1)
    c = _shift(2)

    return a, b, c


def _build_weighted_gmrf(
    D: int, 
    grid_shape: Tuple[int,...], 
    w_dict: dict, 
    reg_shape: Literal["bending", "membrane", "combined"] = "bending", 
    device: str = "cpu"
) -> torch.Tensor:
    P = 1
    for L in grid_shape:
        P *= L
    L0 = torch.zeros(P, P, device=device)

    for axis in range(D):
        if reg_shape in ("membrane", "combined"):
            w = w_dict[f"w1{axis}"].to(device)
            a, b = _diff1_edge_pairs(grid_shape, axis, device)
            
            # D = (-1 at a, +1 at b). D^T diag(w) D:
            L0.index_put_((a, a), w, accumulate=True)
            L0.index_put_((b, b), w, accumulate=True)
            L0.index_put_((a, b), -w, accumulate=True)
            L0.index_put_((b, a), -w, accumulate=True)

        if reg_shape in ("bending", "combined"):
            w = w_dict[f"w2{axis}"].to(device)
            a, b, c = _diff2_edge_triples(grid_shape, axis, device)
            
            # D = (1 at a, -2 at b, +1 at c)
            L0.index_put_((a, a), w, accumulate=True)
            L0.index_put_((a, b), -2.0 * w, accumulate=True)
            L0.index_put_((a, c), w, accumulate=True)
            L0.index_put_((b, a), -2.0 * w, accumulate=True)
            L0.index_put_((b, b), 4.0 * w, accumulate=True)
            L0.index_put_((b, c), -2.0 * w, accumulate=True)
            L0.index_put_((c, a), w, accumulate=True)
            L0.index_put_((c, b), -2.0 * w, accumulate=True)
            L0.index_put_((c, c), w, accumulate=True)

    L0 = L0 + 1e-5 * torch.eye(P, device=device)
    return L0