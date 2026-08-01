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
    _concat,
    _split
)
from ..utils._tensor_utils import _check_tensor
from ..utils._gpu import _stochastic_nmf


def _pca(
    adata: ad.AnnData,
    n_components: int = 25,
    basis_key: str = "X_pca"
) -> None:    
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
    basis_key: str = "X_nmf"
) -> None:
    try:
        X = adata.X
        X = X.toarray() if hasattr(X, "toarray") else X
        W, _ = _stochastic_nmf(
            x=X,
            n_components=n_components,
            max_epochs=100,
            batch_size=1024
        )

        adata.obsm[basis_key] = W
    except Exception:
        nmf = NMF(n_components=n_components)
        X_nmf = nmf.fit_transform(adata.X)

        adata.obsm[basis_key] = X_nmf


def _integration(
    adata: ad.AnnData,
    batch_key: str = "batch",
    basis: str = "X_pca",
) -> None:
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
    key_added: str = 'spatial_manta',
):  
    # Make sure we are working with Tensors
    # IMPORTANT - From here, we only work with tensor objects
    _to_tensor(
        adata=adata,
        spatial_key=spatial_key,
        key_added=key_added
    )

    spatial_key = key_added

    # Centering
    _center(
        adata=adata,
        spatial_key=spatial_key,
        key_added=key_added
    )


def _preprocess(
    source: ad.AnnData,
    target: ad.AnnData,
    batch_key: str = "batch",
    n_components: int = 25,
    pca_basis_key: str = "X_pca",
    nmf_basis_key: str = "X_nmf",
    gene_key: str = "gene",
    spatial_key: str = "spatial",
    key_added: str = "spatial_manta",
):
    # Batch correction/gene intersection can't be parallelized :(
    adatas = [source, target]

    adata = _concat(
        adatas=adatas,
        batch_key=batch_key
    )

    _pca(
        adata=adata,
        n_components=n_components,
        basis_key=pca_basis_key
    )

    _nmf(
        adata=adata,
        n_components=n_components,
        basis_key=nmf_basis_key
    )

    _integration(
        adatas=adatas,
        batch_key=batch_key,
        basis=pca_basis_key
    )

    source = adatas[0]
    target = adatas[1]
    
    adatas: List[ad.AnnData] = _intersect_genes(
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