import anndata as ad
import torch

from ..core._operators import _density_nd
from ..utils._tensor_utils import (
    TensorLike,
    _get_device,
    _as_tensor,
    _check_tensor
)


@torch.no_grad()
def _sample_importance(
    adata: ad.AnnData,
    fraction: float,
    bin_size: int,
    spatial_key: str = "spatial_manta",
    sample_key: str = "sample_importance",
    gamma: float = 1.0
) -> None:    
    device = _get_device()
    pts = _as_tensor(adata.obsm.get(spatial_key), device=device)
    _check_tensor(pts)

    # Compute density here  
    _density_nd(
        adata=adata,
        spatial_key=spatial_key,
        resolution=bin_size,
        sigma=1.0,
        normalize=True
    )


    rho = adata.uns.get(f'rho_{bin_size}')['rho']
    _check_tensor(rho)
    
    if pts.numel() == 0:
        raise ValueError("no elements to sample from")

    N, _ = pts.shape
    N_rho = rho.shape[0]
    if N != N_rho:
        raise ValueError(
            f"all points must have associated distribution value (expected {N}, got {N_rho})"
        )

    n = int(fraction * N)
    k = min(max(1, n), N)

    # Make sure to normalize the provided distribution
    p = rho / rho.sum()

    # Apply gamma
    p = p ** gamma

    # Sample (without replacement)
    sampled_indices = torch.multinomial(
        input=p,
        num_samples=k,
        replacement=False
    )

    adata.uns[f"{sample_key}"] = {
        "n": n,
        "distribution": rho / rho.sum(),
        "gamma": gamma,
        "indices": sampled_indices,
        "pts": pts[sampled_indices]
    }


@torch.no_grad()
def _sample_stratified(
    adata: ad.AnnData,
    bin_size: float |  TensorLike,
    spatial_key: str = "spatial_manta",
    sample_key: str = "sample_stratified",
    shuffle: bool = True,
) -> None:
    device = _get_device()
    pts = _as_tensor(adata.obsm.get(spatial_key), device=device)
    _check_tensor(pts)

    if pts.numel() == 0:
        raise ValueError("no elements to sample from")

    N, D = pts.shape
    device = _get_device()

    # Normalize bin size
    if not torch.is_tensor(bin_size):
        bin_size = torch.full(
            (D,),
            float(bin_size),
            device=device,
            dtype=pts.dtype
        )
    else:
        bin_size = _as_tensor(
            x=bin_size,
            dtype=pts.dtype,
            device=device
        )

    if bin_size.numel() != D:
        raise ValueError(
            f"bin_size must have length {D}"
        )
    if torch.any(bin_size < 0):
        raise ValueError(
            "bin_size must be positive"
        )

    # Shuffle to make representatives of each bin random
    if shuffle:
        perm = torch.randperm(N, device=device)
        pts_s = pts[perm]
    else:
        perm = torch.arange(N, device=device)
        pts_s = pts

    # Compute integer bin coordinates
    mins = pts_s.min(dim=0).values
    bins = torch.floor(
        (pts_s - mins) / bin_size
    ).long()

    # Find unique occupied bins
    _, inverse = torch.unique(
        bins,
        dim=0,
        return_inverse=True
    )

    # Since the first occurence represents the bin,
    # sorting by inverse gives one representative per bin
    sorted_inverse, order = torch.sort(inverse)
    keep = torch.ones_like(sorted_inverse, dtype=torch.bool)
    keep[1:] = (
        sorted_inverse[1:] != sorted_inverse[:-1]
    )
    selected = order[keep]

    # Restore indexing
    original_indices = perm[selected]

    adata.uns[f"{sample_key}"] = {
        "n": selected.shape[0],
        "indices": original_indices,
        "pts": pts_s[selected]
    }


@torch.no_grad()
def _sample_approximate_fps(
    adata: ad.AnnData,
    fraction: float,
    bin_size: float,
    spatial_key: str = "spatial_manta",
    sample_key: str = "sample_afps",
    shuffle: bool = True,
) -> None:
    device = _get_device()
    pts = _as_tensor(adata.obsm.get(spatial_key), device=device)
    _check_tensor(pts)

    if pts.numel() == 0:
        raise ValueError("no elements to sample from")

    N, _ = pts.shape

    # Shuffle to make representatives of each bin random
    if shuffle:
        perm = torch.randperm(N, device=device)
        pts_s = pts[perm]
    else:
        perm = torch.arange(N, device=device)
        pts_s = pts

    # Voxel representation
    mins = pts_s.min(dim=0).values
    voxels = torch.floor(
        (pts - mins) / bin_size
    ).long()

    _, inverse = torch.unique(
        voxels,
        dim=0,
        return_inverse=True
    )
    M = inverse.max() + 1
    first = torch.full(
        (M,),
        N,
        device=device,
        dtype=torch.long
    )
    ids = torch.arange(N, device=device)

    first.scatter_reduce_(
        0,
        inverse,
        ids,
        reduce="amin"
    )

    coarse_pts = pts[first]

    # (Approximate) FPS routine
    M = coarse_pts.shape[0]
    n = int(N * fraction)
    k = min(n, M)
    selected = torch.empty(
        k,
        dtype=torch.long,
        device=device
    )

    # Better initialization
    center = coarse_pts.mean(dim=0)
    current = torch.argmax(
        ((coarse_pts - center) ** 2).sum(dim=1)
    )

    selected[0] = current

    min_dist2 = (
        (coarse_pts - coarse_pts[current]) ** 2
    ).sum(dim=1)
    min_dist2[current] = -1

    for i in range(1, k):
        current = torch.argmax(min_dist2)
        selected[i] = current

        dist2 = (
            (coarse_pts - coarse_pts[current]) ** 2
        ).sum(dim=1)

        min_dist2 = torch.minimum(
            min_dist2,
            dist2
        )
        min_dist2[current] = -1

    original_selected = first[selected]

    adata.uns[f"{sample_key}"] = {
        "n": n,
        "indices": perm[original_selected],
        "pts": pts_s[original_selected]
    }
    