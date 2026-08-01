import anndata as ad
import torch

from ..utils._gpu import (
    _standardize,
    _build_knn_graph
)


def _compute_graph(
    adata: ad.AnnData,
    sampling_key: str |  None = None,
    graph_key: str = "graph",
    k: int = 6,
    alpha: float = 2.0
) -> None:
    if sampling_key == None:
        raise ValueError("expected a sampling key")

    sampling = adata.uns.get(sampling_key)
    if sampling == None:
        raise ValueError(
            f"expected sampling for key `{sampling_key}`, got None"
        )
    
    pts = sampling['pts']
    indices = sampling['indices']

    A, D, P, edge_index = _build_knn_graph(
        x=pts,
        k=k,
        alpha=alpha,
        batch=None,
        mutual=True,
        loop=False
    )

    adata.uns[graph_key] = {
        "A": A,
        "D": D,
        "P": P,
        "edge_iindex": edge_index,
        "k": k,
        "alpha": alpha,
        "indices": indices
    } 


def _compute_base_features(
    adata: ad.AnnData,
    sampling_key: str |  None = None,
    graph_key: str | None = None,
    pca_basis_key: str | None = None,
    nmf_basis_key: str | None = None,
    feature_key: str = "feature",
) -> None:
    if sampling_key == None:
        raise ValueError("expected valid sampling_key, got None")
    if graph_key == None:
        raise ValueError("expected valid graph_key, got None")
    if pca_basis_key == None:
        raise ValueError("expected valid pca_basis_key, got None")
    if nmf_basis_key == None:
        raise ValueError("expected valid nmf_basis_key, got None")

    sampling = adata.uns.get(sampling_key)
    if sampling == None:
        raise ValueError(
            f"expected sampling for key `{sampling_key}`, got None"
        )

    indices = sampling['indices']

    pca_X = adata.obsm.get(pca_basis_key)
    nmf_X = adata.obsm.get(nmf_basis_key)

    if pca_X == None:
        raise ValueError(
            f"expected pca embedding for key `{pca_basis_key}`, got None"
        )
    if nmf_X == None:
        raise ValueError(
            f"expected nmf embedding for key `{nmf_basis_key}`, got None"
        )

    pca_X = _standardize(x=pca_X[indices])
    nmf_X = _standardize(x=nmf_X[indices])

    graph = adata.uns.get(graph_key)
    if graph == None:
        raise ValueError(
            f"expected graph representation for key `{graph_key}`, got None"
        )

    P = graph['P']

    PX_PCA = torch.sparse.mm(P, pca_X)
    PPX_PCA = torch.sparse.mm(P, PX_PCA)

    PX_NMF = torch.sparse.mm(P, nmf_X)
    PPX_NMF = torch.sparse.mm(P, PX_NMF)

    feature = torch.cat(
        [
            pca_X,
            PX_PCA,
            PPX_PCA,
            nmf_X,
            PX_NMF,
            PPX_NMF
        ],
        dim=1
    )

    adata.uns[feature_key] = {
        "feature": feature,
        "pca_basis_key": pca_basis_key,
        "nmf_basis_key": nmf_basis_key,
        "graph_key":  graph_key,
        "indices": indices
    }