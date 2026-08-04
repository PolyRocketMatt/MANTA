import anndata as ad
import concord as ccd
import torch
import numpy as np
import scipy.sparse as sp
import rapids_singlecell as rsc

from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import NMF, PCA
from typing import List

from ..utils._anndata_utils import (
    _register_coordinates, 
    _register_transform,
)
from ..utils._gpu import _stochastic_nmf
from ..utils._progress import (
    ProgressFn,
    _get_progress,
    _update_progress
)
from ..utils._tensor_utils import (
    _as_tensor,
    _from_tensor,
    _check_tensor
)


def _pca(
    adata: ad.AnnData,
    n_components: int = 25,
    basis_key: str = "X_pca",
    progress: ProgressFn = None
) -> None:  
    if progress:  
        _update_progress(
            progress=progress, 
            message="Running PCA"
        )

    try:
        from cuml.decomposition import PCA as cumlPCA

        pca = cumlPCA(n_components=n_components, svd_solver="auto")
        X_pca = pca.fit_transform(adata.X)

        adata.obsm[basis_key] = X_pca
    except ImportError:
        pca = PCA(n_components=n_components, svd_solver="auto")
        X_pca = pca.fit_transform(adata.X)

        adata.obsm[basis_key] = X_pca


def _nmf(
    adata: ad.AnnData,
    n_components: int = 25,
    basis_key: str = "X_nmf",
    progress: ProgressFn = None
) -> None:
    if progress:  
        _update_progress(
            progress=progress, 
            message="Running NMF"
        )
    
    try:
        X = adata.X
        X = X.toarray() if hasattr(X, "toarray") else X
        W, _ = _stochastic_nmf(
            x=X,
            n_components=n_components,
            max_epochs=100,
            batch_size=1024
        )

        adata.obsm[basis_key] = _from_tensor(W)
        X_nmf = W
    except Exception as e:
        nmf = NMF(n_components=n_components)
        X_nmf = nmf.fit_transform(adata.X)

        adata.obsm[basis_key] = X_nmf

    adata.uns[basis_key] = {
        "values": X_nmf
    }


def _integration(
    adata: ad.AnnData,
    batch_key: str = "batch",
    basis: str = "X_pca",
    progress: ProgressFn = None
) -> None:
    if progress:  
        _update_progress(
            progress=progress, 
            message="Running Batch Correction"
        )
    
    rsc.pp.harmony_integrate(
        adata=adata,
        key=batch_key,
        basis=basis
    )
    rsc.pp.neighbors(adata, use_rep=f'{basis}_harmony')
    rsc.tl.umap(adata)


def _intersect_genes(
    adatas: List[ad.AnnData],
    gene_key: str,
    progress: ProgressFn = None
) -> List[ad.AnnData]:
    if not adatas:
        return []

    if progress:
        _update_progress(
            progress=progress, 
            message="Intersecting genes"
        )

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


def _center(
    adata: ad.AnnData,
    spatial_key: str = 'spatial',
    key_added: str = 'spatial',
    progress: ProgressFn = None
) -> ad.AnnData:
    if progress:
        _update_progress(
            progress=progress, 
            message="Centering"
        )

    pts = _as_tensor(adata.obsm[spatial_key])
    _check_tensor(pts)

    ndim = pts.shape[-1]
    center = (pts.min(dim=0).values + pts.max(dim=0).values) / 2
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