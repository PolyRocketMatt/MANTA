import anndata as ad
import torch

from ..utils._transform import Transform


def _register_coordinates(
    adata: ad.AnnData,
    pts: torch.Tensor,
    spatial_key: str = 'spatial',
) -> None:
    adata.obsm[spatial_key] = pts


def _register_transform(
    adata: ad.AnnData,
    ndim: int,
    rotation: torch.Tensor | None = None,
    translation: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> None:    
    transform = Transform(
        rotation=rotation,
        translation=translation,
        scale=scale,
        ndim=ndim
    )

    adata.uns.setdefault("transforms", [])
    adata.uns["transforms"].append(transform)