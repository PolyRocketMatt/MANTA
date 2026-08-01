import numpy as np
import torch

from typing import Tuple, Union
from torch_cluster import knn_graph

from ..utils._tensor_utils import (
    TensorLike,
    _get_device,
    _as_tensor
)


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