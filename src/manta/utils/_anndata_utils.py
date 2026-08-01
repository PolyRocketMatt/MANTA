import anndata as ad
import numpy as np
import torch

from typing import List

from ._transform import Transform


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


def _concat(
    adatas: List[ad.AnnData],
    batch_key: str = "batch"
) -> ad.AnnData:
    # Collect all obsm keys
    all_keys = set()
    for a in adatas:
        all_keys.update(a.obsm.keys())

    # Determine dimensionality for each key
    dims = {}

    for key in all_keys:
        for a in adatas:
            if key in a.obsm:
                dims[key] = a.obsm[key].shape[1]
                break

    # Ensure every adata has every obsm key
    normalized = []

    for a in adatas:
        a = a.copy()

        for key in all_keys:
            if key not in a.obsm:
                d = dims[key]

                a.obsm[key] = np.full(
                    (a.n_obs, d),
                    np.nan,
                    dtype=np.float32
                )

        normalized.append(a)

    # Finally, proper concatenation
    return ad.concat(
        normalized,
        join="outer",
        merge="unique",
        label=batch_key
    )


def _split(
    adata: ad.AnnData,
    batch_key: str = "batch"
) -> List[ad.AnnData]:
    return [adata[adata.obs[batch_key] == sample].copy() for sample in adata.obs[batch_key].unique()]