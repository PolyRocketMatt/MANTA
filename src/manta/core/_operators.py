import anndata as ad
import torch
import torch.nn.functional as F

from typing import Tuple

from ..utils._tensor_utils import (
    _get_device,
    _check_tensor
)


def _voxelize(
    adata: ad.AnnData,
    spatial_key: str = "spatial_manta",
    resolution: int = 128,
    eps: float = 1e-8
) -> None:
    pts = adata.obsm.get(spatial_key)
    _check_tensor(pts)
    
    # Grid bounds
    bounds_min = pts.min(0).values
    bounds_max = pts.max(0).values
    span = bounds_max - bounds_min + eps
    
    # Voxelize
    voxel_size = span / resolution
    voxel_indices = ((pts - bounds_min) / voxel_size).long().clamp(0, resolution-1)

    adata.uns[f"voxel_{resolution}"] = voxel_size
    adata.obsm[f"voxel_{resolution}"] = voxel_indices


def _density(
    adata: ad.AnnData,
    spatial_key: str = "spatial_manta",
    resolution: int = 128,
    sigma: float = 1.0
) -> None:
    pts = adata.obsm.get(spatial_key)
    device = _get_device()
    dtype = torch.float32
    _check_tensor(pts)

    N, D = pts.shape
    voxel_size, voxel_indices = _voxelize(pts, resolution)

    # Density grid
    grid_shape = [resolution] * D
    density_grid = torch.zeros(*grid_shape, device=device, dtype=dtype)

    # Compute flat indices
    strides = torch.tensor(
        [resolution ** (D - i - 1) for i in range(D)],
        device=device,
        dtype=torch.long
    )
    flat_idx = (voxel_indices * strides).sum(dim=1)

    density_grid_flat = density_grid.flatten()
    density_grid_flat.index_add_(0, flat_idx, torch.ones(N, device=device, dtype=dtype))
    density_grid = density_grid_flat.view(*grid_shape)

    # Smoothing (separable Gaussians)
    if sigma > 0:
        radius = int(3 * sigma)
        coords = torch.arange(-radius, radius+1, device=device, dtype=dtype)

        gauss = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss /= gauss.sum()

        # Add batch + channel dimensions
        grid = density_grid.unsqueeze(0).unsqueeze(0)

        # Obtain conv method for current dimension
        conv = {
            1: F.conv1d,
            2: F.conv2d,
            3: F.conv3d
        }.get(D, None)
        if conv is None:
            raise NotImplementedError(f"conv{D}d not supported in PyTorch")

        for dim in range(D):
            shape = [1, 1] + [1] * D
            shape[2 + dim] = -1
            kernel = gauss.view(shape)

            padding = [0] * D
            padding[dim] = radius

            padding = tuple(padding)

            grid = conv(grid, kernel, padding=tuple(padding))

        density_grid = grid[0, 0]
    
    # Interpolate density
    idx = voxel_indices.clamp(0, resolution - 1)
    rho = density_grid[tuple(idx[:, i] for i in range(D))]

    # Gradient (central differences)
    grad = torch.zeros_like(pts)

    for i in range(D):
        plus = idx.clone()
        minus = idx.clone()

        plus[:, i] = (plus[:, i] + 1).clamp(0, resolution - 1)
        minus[:, i] = (minus[:, i] - 1).clamp(0, resolution - 1)

        plus_vals = density_grid[tuple(plus[:, j] for j in range(D))]
        minus_vals = density_grid[tuple(minus[:, j] for j in range(D))]

        grad[:, i] = (plus_vals - minus_vals) / (2 * voxel_size[i])

    adata.obsm[f"rho_{resolution}"] = rho
    adata.obsm[f"rho_grad_{resolution}"] = grad