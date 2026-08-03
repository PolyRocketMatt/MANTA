import numpy as np
import torch

from typing import Iterable, Literal, Optional, Tuple
from torch_cluster import knn_graph

from ..utils._tensor_utils import (
    TensorLike,
    _get_device,
    _as_tensor
)


@torch.no_grad()
def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def _pairwise_sqeuclid_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1).unsqueeze(0)

    return torch.clamp(a2 + b2 - 2.0 * (a @ b.t()), min=0.0)


@torch.no_grad()
def _pairwise_dist(
    a: torch.Tensor, 
    b: torch.Tensor,
    dist_fn: Literal["euclidean", "cosine"] = "euclidean"
) -> torch.Tensor:
    if dist_fn == "euclidean":
        return _pairwise_sqeuclid_dist(a, b)
    elif dist_fn == "cosine":
        return _pairwise_cosine_dist(a, b)
    else:
        raise ValueError(
            f"no distance function named `{dist_fn}`"
        )



@torch.no_grad()
def _pairwise_cosine_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
     a = _l2_normalize(a, eps=eps)
     b = _l2_normalize(b, eps=eps)

     return 1.0 - a @ b.t()


@torch.no_grad()
def _chunked_range(n: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield start, end


@torch.no_grad()
def _standardize(
    x: TensorLike,
    eps: float = 1e-8
) -> torch.Tensor:
    x = _as_tensor(
        x=x,
        device=_get_device
    )
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=True)
    return (x - mean) / (std + eps)


@torch.no_grad()
def _stochastic_nmf(
    x: TensorLike,
    n_components: int = 25,
    max_epochs: int = 20,
    batch_size: int = 1024,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = _as_tensor(
        x=x,
        device=_get_device()
    )

    n_samples, n_features = x.shape

    # Initialize factors
    W = torch.rand((n_samples, n_components), device=_get_device())
    H = torch.rand((n_components, n_features), device=_get_device())

    def get_batch(idx):
        Xb = x[idx]
        return Xb

    # Training loop
    for _ in range(max_epochs):
        perm = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]

            Xb = get_batch(idx)
            Wb = W[idx]

            # Update H
            numerator = Wb.T @ Xb
            denominator = (Wb.T @ Wb) @ H + eps
            H *= numerator / denominator

            # Update W (batch only)
            numerator = Xb @ H.T
            denominator = Wb @ (H @ H.T) + eps
            W[idx] = Wb * (numerator / denominator)

    return W, H
    

@torch.no_grad()
def _build_knn_graph(
    x: torch.Tensor,
    k: int = 10,
    alpha: float = 2.0,
    batch: torch.Tensor = None,
    mutual: bool = True,
    loop: bool = False,
    device: str = "cpu"
) -> Tuple[torch.Tensor, ...]:
    N = x.size(0)

    # kNN Graph (initial edges)
    edge_index = knn_graph(x, k=k, batch=batch, loop=loop)
    row, col = edge_index[0], edge_index[1]

    # Compute distance for each edge
    diff = x[row] - x[col]
    dist = (diff * diff).sum(dim=1)

    perm = torch.argsort(row)
    row_s = row[perm]
    dist_s = dist[perm]

    counts = torch.bincount(row_s, minlength=N)
    max_k = counts.max().item()
    pad = torch.full((N, max_k), float('inf'), device=device)
    idx = torch.zeros(N, device=device, dtype=torch.long)

    for i in range(row_s.size(0)):
        r = row_s[i]
        pad[r, idx[r]] = dist_s[i]
        idx[r] += 1

    # Choose rank for "median-like" statistic
    rank = torch.clamp(counts // 2, min=0)

    # Gather k-th element per row via topk
    vals, _ = torch.topk(pad, k=max_k, dim=1, largest=False)

    # Finally, compute the median
    med = vals[torch.arange(N, device=device), rank]

    # Map median back to edges
    med_edge = med[row]

    # Distance-based threhsold filtering
    mask = dist <= alpha * med_edge
    row, col, dist = row[mask], col[mask], dist[mask]

    # Mutual edges only if requested!
    if mutual:
        rev = torch.stack([col, row], dim=0)
        edges = torch.cat([torch.stack([row, col], dim=0), rev], dim=1)

        # Remove duplicates
        edges = torch.unique(edges, dim=1)

        row, col = edges[0], edges[1]

    row = row.flatten()
    col = col.flatten()

    # Adjacency matrix A
    A = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        torch.ones(row.size(0), device=device),
        size=(N,N),
        device=device
    ).coalesce()

    # Degree matrix D
    deg = torch.sparse.sum(A, dim=1).to_dense()
    D = torch.diag(deg)

    # Transition matrix P
    inv_deg = torch.zeros_like(deg)
    inv_deg[deg > 0] = 1.0 / deg[deg > 0]
    values = A.values() * inv_deg[A.indices()[0]]
    P = torch.sparse_coo_tensor(
        A.indices(),
        values,
        size=A.size(),
        device=device
    ).coalesce()

    return A, D, P, torch.stack([row, col], dim=0)


@torch.no_grad()
def _kmeans(
    x: torch.Tensor,
    n_clusters: int,
    n_iter: int = 25,
    dist_fn: Literal["euclidean", "cosine"] = "euclidean",
    sample_size: Optional[int] = 50_000,
    batch_size: int = 4096,
    tolerance: float = 1e-4,
    seed: int = 0,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    LLoyd k-means clustering (only use for large N)
    """
    if not torch.is_floating_point(x):
        x = x.to(torch.float32)

    N, _ = x.shape
    device = x.device
    generator = torch.Generator().manual_seed(seed=seed)

    if sample_size is not None and sample_size < N:
        perm = torch.randperm(N, device=device, generator=generator)[:sample_size]
        fit_x = x[perm]
    else:
        fit_x = x

    M, _ = fit_x.shape
    if n_clusters >= M:
        # Degenerate > more clusters requested than samples
        labels = torch.arange(N, device=device) % max(1, n_clusters)
        centroids = x[:n_clusters].clone()
        return labels, centroids

    # kmeans++-like initialization
    centroids = fit_x[torch.randperm(M, device=device, generator=generator)[:n_clusters]].clone()
    for _ in range(n_iter):
        # Fit to existing centroids
        assign_chunks = []
        for s, e in _chunked_range(M, batch_size):  
            dist = _pairwise_dist(fit_x[s:e], centroids, dist_fn)
            assign_chunks.append(torch.argmin(dist, dim=1))
        assign = torch.cat(assign_chunks, dim=0)

        # Recalculate centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_clusters, device=device, dtype=fit_x.dtype)
        new_centroids.index_add_(0, assign, fit_x)
        counts.index_add_(0, assign, torch.ones_like(assign, dtype=fit_x.dtype))
        counts = counts.clamp_min(1.0)
        new_centroids = new_centroids / counts.unsqueeze(1)

        # Re-seed empty clusters if any exist
        empty = (counts <= 1.0)
        if empty.any():
            repl = fit_x[torch.randperm(M, device=device, generator=generator)[:int(empty.sum().item())]]
            new_centroids[empty] = repl

        shift = torch.norm(new_centroids - centroids) / (torch.norm(centroids) + eps)
        centroids = new_centroids

        # Convergence/Early
        if shift.item() < tolerance:
            break

    # Final assignment (for all N)
    labels = torch.empty(N, device=device, dtype=torch.long)
    for s, e in _chunked_range(N, batch_size):
        dist = _pairwise_dist(fit_x[s:e], centroids, dist_fn)
        labels[s:e] = torch.argmin(dist, dim=1)

    return labels, centroids