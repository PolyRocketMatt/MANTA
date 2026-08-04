import anndata as ad

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from ..core._preprocessing import (
    _pca,
    _nmf,
    _integration,
    _intersect_genes,
    _center
)
from ..utils._anndata_utils import (
    _concat,
    _split
)
from ..utils._progress import (
    _get_progress,
    _update_progress,
)
from ..utils._tensor_utils import _as_tensor


def preprocess(
    source: ad.AnnData,
    target: ad.AnnData,
    batch_key: str = "batch",
    n_components: int = 25,
    pca_basis_key: str = "X_pca",
    nmf_basis_key: str = "X_nmf",
    gene_key: str = "gene",
    spatial_key: str = "spatial",
    key_added: str = "spatial_manta",
    centering: bool = False,
) -> Tuple[ad.AnnData, ad.AnnData]:
    progress, _ = _get_progress(
        steps=6 if centering else 5,
        desc="Preprocessing"
    )

    # Initial, independent PCA
    with ThreadPoolExecutor(max_workers=2) as executor:
        _update_progress(
            progress=progress, 
            message="Initial PCA"
        )

        futures = [
            executor.submit(
                _pca,
                adata=source,
                n_components=n_components,
                basis_key="X_pca",
            ),
            executor.submit(
                _pca,
                adata=target,
                n_components=n_components,
                basis_key="X_pca",
            ),
        ]

        for future in futures:
            future.result()

    # Batch correction/gene intersection can't be parallelized :(
    adatas = [source, target]
    adata = _concat(
        adatas=adatas,
        batch_key=batch_key
    )

    _integration(
        adata=adata,
        batch_key=batch_key,
        basis=pca_basis_key,
        progress=progress
    )

    _pca(
        adata=adata,
        n_components=n_components,
        basis_key=pca_basis_key,
        progress=progress
    )

    _nmf(
        adata=adata,
        n_components=n_components,
        basis_key=nmf_basis_key,
        progress=progress
    )

    adatas = _split(adata)
    adatas: List[ad.AnnData] = _intersect_genes(
        adatas=adatas,
        gene_key=gene_key,
        progress=progress
    )
    
    for adata in adatas:
        #N = adata.obsm[spatial_key].shape[0]

        # Update positions
        adata.obsm[key_added] = adata.obsm[spatial_key]

        # Insert multichannel types
        adata.uns[pca_basis_key] = { "values": _as_tensor(adata.obsm[pca_basis_key]) }
        adata.uns[nmf_basis_key] = { "values": _as_tensor(adata.obsm[nmf_basis_key]) }

    spatial_key = key_added

    # Centering can happen parallelized :)
    if centering:
        with ThreadPoolExecutor(max_workers=2) as executor:
            _update_progress(
                progress=progress, 
                message="Centering"
            )

            futures = [
                executor.submit(
                    _center,
                    adata=adata,
                    spatial_key=spatial_key,
                    key_added=key_added,
                )
                for adata in adatas
            ]

            for future in futures:
                future.result()

    return adatas