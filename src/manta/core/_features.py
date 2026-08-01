import anndata as ad
import torch

from torch_cluster import knn_graph

from ..utils._gpu import (
    _standardize,
    _build_knn_graph
)
from ..utils._tensor_utils import (
    _get_device,
    _as_tensor,
    _check_tensor
)


def _compute_graph(
    adata: ad.AnnData,
    sampling_key: str |  None = None,
    graph_key: str = "graph",
    k: int = 6,
    alpha: float = 2.0
) -> None:
    if sampling_key == None:
        raise ValueError("expected valid sampling_key, got None")

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


def _compute_gene_features(
    adata: ad.AnnData,
    sampling_key: str |  None = None,
    graph_key: str | None = None,
    pca_basis_key: str | None = None,
    nmf_basis_key: str | None = None,
    feature_key: str = "gene_features",
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
    _check_tensor(indices)

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

    pca_X = _standardize(x=pca_X)
    nmf_X = _standardize(x=nmf_X)

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
    )[indices]

    adata.uns[feature_key] = {
        "feature": feature,
        "sampling_key": sampling_key,
        "pca_basis_key": pca_basis_key,
        "nmf_basis_key": nmf_basis_key,
        "graph_key":  graph_key,
        "indices": indices
    }


def _compute_graph_features(
    adata: ad.AnnData,
    sampling_key: str | None = None,
    feature_key: str = "graph_features",
    k: int = 10,
    eps: float = 1e-8,
) -> None:
    if sampling_key is None:
        raise ValueError("expected valid sampling_key, got None")

    sampling = adata.uns.get(sampling_key)
    if sampling is None:
        raise ValueError(
            f"expected sampling for key `{sampling_key}`, got None"
        )

    pts = sampling["pts"]
    indices = sampling["indices"]

    _check_tensor(pts)
    _check_tensor(indices)

    N, D = pts.shape
    device = _get_device()

    # knn-based index
    edge_index = knn_graph(pts, k=k, loop=False)
    row, col = edge_index

    diff = pts[col] - pts[row]
    dist = diff.norm(dim=1)

    # Displacement statistics
    deg = torch.bincount(row, minlength=N).float()

    sum_d = torch.zeros(N, device=device)
    sum_d.index_add_(0, row, dist)

    sum_d2 = torch.zeros(N, device=device)
    sum_d2.index_add_(0, row, dist * dist)

    count = deg.clamp(min=1)

    mean_d = sum_d / count
    var_d = sum_d2 / count - mean_d.square()
    std_d = torch.sqrt(var_d.clamp(min=eps))

    # Density (proxy)
    rho = 1.0 / (mean_d + eps)

    # Covariance matrix
    mean_diff = torch.zeros(N, D, device=device)
    mean_diff.index_add_(0, row, diff)
    mean_diff /= count[:, None]
    centered = diff - mean_diff[row]
    outer = centered[:, :, None] * centered[:, None, :]

    C = torch.zeros(N, D, D, device=device)
    C.index_add_(0, row, outer)
    C /= count[:, None, None]

    C = torch.nan_to_num(C)

    # Eigenvalues (of covariance)
    eigvals = torch.linalg.eigvalsh(C)

    # Anisotropy
    trace = eigvals.sum(dim=1)
    l1 = eigvals[:, -1]

    anisotropy = l1 / (trace + eps)
    anisotropy = torch.nan_to_num(anisotropy)

    feature = torch.stack(
        [
            rho,          # inverse mean neighbour distance
            l1,           # dominant variance
            trace,        # total variance
            anisotropy,   # λ_max / trace
            mean_d,       # mean neighbour distance
            std_d,        # std. neighbour distance
        ],
        dim=1,
    )[indices]

    adata.uns[feature_key] = {
        "feature": feature,
        "sampling_key": sampling_key,
        "k": k,
        "indices": indices,
    }
