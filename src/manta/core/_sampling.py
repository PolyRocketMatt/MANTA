import anndata as ad
import torch

from ..utils._tensor_utils import (
    _check_tensor
)


@torch.no_grad()
def _sample_importance(
    adata: ad.AnnData,
    n: int,
    spatial_key: str = "spatial_manta",
    distribution_key: str | None = None,
    sample_key: str = "samples_importance",
    gamma: float = 1.0
) -> None:
    if distribution_key == None:
        raise ValueError(
            f"expected distribution key to be `str`, got `None`"
        )
    
    pts = adata.obsm.get(spatial_key)
    rho = adata.obsm.get(distribution_key)
    _check_tensor(pts)
    _check_tensor(rho)

    N, _ = pts.shape
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