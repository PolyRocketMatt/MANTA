import anndata as ad
import torch
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from typing import List

from ..utils.anndata_utils import (
    _register_coordinates, 
    _register_transform
)
from ..utils.tensor_utils import _check_tensor


def _intersect_genes(
    adatas: List[ad.AnnData],
    gene_key: str,
) -> List[ad.AnnData]:
    if not adatas:
        return []

    upper_names = [
        adata.var[gene_key].str.upper()
        for adata in adatas
    ]

    # Compute intersection without modifying AnnData objects
    common_genes = set(upper_names[0])
    for names in upper_names[1:]:
        common_genes.intersection_update(names)


    result = []
    for adata, names in zip(adatas, upper_names):
        mask = names.isin(common_genes)

        filtered = adata[:, mask].copy()
        filtered.var_names = names[mask]

        result.append(filtered)

    return result


def _to_tensor(
    adata: ad.AnnData,
    spatial_key: str = 'spatial',
    key_added: str = 'spatial'
) -> ad.AnnData:
    pts = adata.obsm[spatial_key]

    if isinstance(pts, np.ndarray):
        tensor_pts = torch.from_numpy(pts)

    adata.obsm[key_added] = tensor_pts


def _center(
    adata: ad.AnnData,
    spatial_key: str = 'spatial',
    key_added: str = 'spatial'
) -> ad.AnnData:
    pts = adata.obsm[spatial_key]

    _check_tensor(pts)

    ndim = pts.shape[-1]
    center = (pts.min(dim=0) + pts.max(dim=0)) / 2
    pts_centered = pts - center

    _register_coordinates(
        adata=adata,
        pts=pts_centered,
        spatial_key=key_added
    )

    _register_transform(
        adata=adata,
        ndim=ndim,
        translation=-center
    )


def _preprocess_adata(
    adata: ad.AnnData,
    spatial_key: str = 'spatial',
    key_added: str = 'spatial_manta'
):
    _to_tensor(
        adata=adata,
        spatial_key=spatial_key,
        key_added=key_added
    )

    spatial_key = key_added

    _center(
        adata=adata,
        spatial_key=spatial_key,
        key_added=key_added
    )


def _preprocess(
    source: ad.AnnData,
    target: ad.AnnData,
    gene_key: str = 'gene',
    spatial_key: str = 'spatial',
    key_added: str = 'spatial_manta'
):
    # Intersecting can't happen parallelized :(
    _intersect_genes(
        adatas=[source, target],
        gene_key=gene_key
    )

    # Centering can happen parallelized :)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _preprocess_adata,
                adata=source,
                spatial_key=spatial_key,
                key_added=key_added
            ),
            executor.submit(
                _preprocess_adata,
                adata=target,
                spatial_key=spatial_key,
                key_added=key_added
            )
        ]

        for future in futures:
            future.result()