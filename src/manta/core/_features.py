import anndata as ad
import torch

from torch_cluster import (
    knn_graph,
    radius
)

from ..utils._gpu import (
    _standardize,
    _build_knn_graph
)
from ..utils._tensor_utils import (
    _get_device,
    _check_tensor
)

# TODO: Consider moving to @torch.inference_mode()
@torch.no_grad()
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


@torch.no_grad()
def _compute_base_features(
    adata: ad.AnnData,
    pca_basis_key: str | None = None,
    nmf_basis_key: str | None = None,
    feature_key: str = "base_features"
) -> None:
    if pca_basis_key == None:
        raise ValueError("expected valid pca_basis_key, got None")
    if nmf_basis_key == None:
        raise ValueError("expected valid nmf_basis_key, got None")
    
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

    feature = torch.cat(
        [
            pca_X,
            nmf_X
        ],
        dim=1
    )

    adata.uns[feature_key] = {
        "feature": feature,
        "pca_basis_key": pca_basis_key,
        "nmf_basis_key": nmf_basis_key,
    }


@torch.no_grad()
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
    )[indices]

    adata.uns[feature_key] = {
        "feature": feature,
        "sampling_key": sampling_key,
        "pca_basis_key": pca_basis_key,
        "nmf_basis_key": nmf_basis_key,
        "graph_key":  graph_key,
        "indices": indices
    }


@torch.no_grad()
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
    if sampling == None:
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


@torch.no_grad()
def _compute_microenvironment_features(
    adata: ad.AnnData,
    spatial_key: str | None = None,
    sampling_key: str | None = None,
    base_features_key: str | None = None,
    feature_key: str = "micro_features",
    radius: int = 50,
) -> None:
    if spatial_key is None:
        raise ValueError("expected valid spatial_key, got None")
    if sampling_key is None:
        raise ValueError("expected valid sampling_key, got None")
    if base_features_key is None:
        raise ValueError("expected valid base_features_key, got None")

    all_pts = adata.obsm.get(spatial_key)
    if all_pts == None:
        raise ValueError(
            f"expected tensor for key `{spatial_key}`, got None"
        )

    sampling = adata.uns.get(sampling_key)
    if sampling == None:
        raise ValueError(
            f"expected sampling for key `{sampling_key}`, got None"
        )

    pts = sampling["pts"]
    indices = sampling["indices"]
    _check_tensor(pts)
    _check_tensor(indices)

    base_features = adata.uns.get(base_features_key)
    if base_features is None:
        raise ValueError(
            f"expected gene features for key `{base_features}`, got None"
        )

    feature = base_features['feature']
    _check_tensor(feature)

    device = _get_device()

    # MAKE SURE DIMENSIONALITY IS CORRECT
    if all_pts.shape[-1] != pts.shape[-1]:
        raise ValueError(
            f"dimensionality of all points ({all_pts.shape[-1]}) must match dimensionality of sampled points ({pts.shape[-1]})"
        )

    row, col = radius(
        x=all_pts,      # ALL points
        y=pts,          # subsampled points
        r=radius
    )

    N_s, _ = pts.shape          # N_s = # subsampled points
    _, D_b = feature.shape      # D_b = # pca + # nmf components

    micro_feature = torch.zeros((N_s, D_b), device, dtype=torch.float32)

    # Sum neighbour features
    micro_feature.index_add_(0, row, feature[col])

    # Divide by # neighbours
    counts = torch.bincount(row, minlength=N_s).clamp(min=1)
    micro_feature /= counts.unsqueeze(1)

    adata.uns[feature_key] = {
        "feature": feature,
        "spatial_key": spatial_key, 
        "sampling_key": sampling_key,
        "base_features_key": base_features_key,
        "radius": radius,
        "indices": indices,
    }


def _compute_features(
    adata: ad.AnnData,

    spatial_key: str | None = None,
    sampling_key: str |  None = None,

    graph_key: str = "graph",
    base_features_key: str = "base_features",
    gene_features_key: str = "gene_features",
    graph_features_key: str = "graph_features",
    micro_features_key: str = "micro_features",
    feature_key: str = "section_features",

    pca_basis_key: str | None = None,
    nmf_basis_key: str | None = None,
    graph_k: int = 6,
    graph_alpha: float = 2.0,
    graph_features_k: int = 10,
    micro_env_radius: float = 50.0
) -> None:
    if sampling_key is None:
        raise ValueError("expected valid sampling_key, got None")

    sampling = adata.uns.get(sampling_key)
    if sampling == None:
        raise ValueError(
            f"expected sampling for key `{sampling_key}`, got None"
        )

    indices = sampling["indices"]
    _check_tensor(indices)
    
    _compute_graph(
        adata=adata,
        sampling_key=sampling_key,
        graph_key=graph_key,
        k=graph_k,
        alpha=graph_alpha
    )

    # TODO: Consider moving into micro-environment routine
    #       The result of this function isn't needed downstream
    _compute_base_features(
        adata=adata,
        pca_basis_key=pca_basis_key,
        nmf_basis_key=nmf_basis_key,
        feature_key=base_features_key
    )

    _compute_gene_features(
        adata=adata,
        sampling_key=sampling_key,
        graph_key=graph_key,
        pca_basis_key=pca_basis_key,
        nmf_basis_key=nmf_basis_key,
        feature_key=gene_features_key
    )

    _compute_graph_features(
        adata=adata,
        sampling_key=sampling_key,
        feature_key=graph_features_key,
        k=graph_features_k
    )

    _compute_microenvironment_features(
        adata=adata,
        spatial_key=spatial_key,
        sampling_key=sampling_key,
        base_features_key=base_features_key,
        feature_key=micro_features_key,
        radius=micro_env_radius
    )

    gene_features       = adata.uns.get(gene_features_key)
    graph_features      = adata.uns.get(graph_features_key)
    micro_features      = adata.uns.get(micro_features_key)

    feature_raw = torch.cat(
        [
            gene_features['feature'],
            graph_features['feature'],
            micro_features['feature']
        ],
        dim=1
    )
    feature = _standardize(x=feature_raw)

    adata.uns[feature_key] = {
        "feature": feature,
        "indices": indices
    }