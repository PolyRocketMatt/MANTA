import anndata as ad
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...utils._tensor_utils import (
    TensorLike,
    _get_device,
    _as_tensor,
    _check_tensor
)


def _voxelize(
    pts: torch.Tensor,
    bin_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    origin = pts.min(dim=0).values
    voxel_idx = torch.floor((pts - origin) / bin_size).to(torch.int64)

    return voxel_idx, origin


def _aggregate(
    adata: ad.AnnData,
    bin_size: int,
    min_points: int = 10,
    normalize_expression: bool = True,
    spatial_key: str = 'spatial_manta',
    expression_key: str | None = None
) -> None:
    device = _get_device()

    if expression_key is None:
        X = adata.X
    else:
        X = adata.obsm.get(expression_key)

    # Handle sparse matrix
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = _as_tensor(X, dtype=torch.float32, device=device)

    pts = _as_tensor(adata.obsm.get(spatial_key), dtype=torch.float32, device=device)
    voxel_idx, origin = _voxelize(pts, bin_size)

    keys, inverse, counts = torch.unique(
        voxel_idx,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )

    n_voxels = keys.shape[0]

    # Aggregate by voxel
    expr_sum = torch.zeros((n_voxels, X.shape[1]), dtype=X.dtype, device=device,)
    pts_sum = torch.zeros((n_voxels, pts.shape[1]), dtype=pts.dtype, device=device,)

    expr_sum.index_add_(0, inverse, X)
    pts_sum.index_add_(0, inverse, pts)

    counts_float = counts.to(X.dtype)

    expr_mean = expr_sum / counts_float[:, None]
    pts_mean = pts_sum / counts_float[:, None]

    if normalize_expression:
        norms = torch.linalg.vector_norm(expr_mean, dim=1, keepdim=True,)
        norms = torch.clamp(norms, min=1.0)
        expr_mean = expr_mean / norms

    mask = counts >= min_points

    adata.uns[f"voxel_{bin_size}"] = {
        "keys": keys[mask],
        "expr": expr_mean[mask],
        "centroid": pts_mean[mask],
        "counts": counts_float[mask],
        "origin": origin,
    }


def _match_voxels(
    source_expr: torch.Tensor,
    target_expr: torch.Tensor,
    mutual: bool = True,
    min_similarity: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Because we're checking similarity, make sure expression IS normalized
    src = source_expr / (torch.linalg.vector_norm(source_expr, dim=1, keepdim=True) + 1e-12)
    tgt = target_expr / (torch.linalg.vector_norm(target_expr, dim=1, keepdim=True) + 1e-12)

    sim = src @ tgt.T  # (n_src, n_tgt)
    src_to_tgt = torch.argmax(sim, dim=1)

    if mutual:
        tgt_to_src = torch.argmax(sim, dim=0)
        is_mutual = (tgt_to_src[src_to_tgt] == torch.arange(src.shape[0], device=src.device))
        src_idx = torch.where(is_mutual)[0]
    else:
        src_idx = torch.arange(src.shape[0], device=src.device)

    tgt_idx = src_to_tgt[src_idx]
    sims = sim[src_idx, tgt_idx]
    mask = sims >= min_similarity

    return src_idx[mask], tgt_idx[mask], sims[mask]


def _kabsch_umeyama(
    source_pts: torch.Tensor,
    target_pts: torch.Tensor,
    weights: torch.Tensor | None = None,
    allow_scaling: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _get_device()
    src_pts = _as_tensor(source_pts, dtype=torch.float32, device=device)
    tgt_pts = _as_tensor(target_pts, dtype=src_pts.dtype, device=device)

    if src_pts.shape != tgt_pts.shape:
        raise ValueError("source_pts and target_pts must have the same shape")

    N, d = src_pts.shape

    if weights is None:
        w = torch.ones(N, dtype=src_pts.dtype, device=device)
    else:
        w = _as_tensor(weights, dtype=src_pts.dtype, device=device)

    w = w / w.sum()

    mu_src = (w[:, None] * src_pts).sum(dim=0)
    mu_tgt = (w[:, None] * tgt_pts).sum(dim=0)

    src_c = src_pts - mu_src
    tgt_c = tgt_pts - mu_tgt

    # Weighted cross-covariance
    cov = (tgt_c * w[:, None]).T @ src_c

    U, D, Vh = torch.linalg.svd(cov)
    S = torch.eye(d, dtype=src_pts.dtype, device=device)

    if torch.linalg.det(U) * torch.linalg.det(Vh) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vh

    if allow_scaling:
        var_src = (w[:, None] * src_c.square()).sum()

        if var_src > 0:
            s = torch.trace(torch.diag(D) @ S) / var_src
        else:
            s = torch.ones((), dtype=src_pts.dtype, device=device)
    else:
        s = torch.ones((), dtype=src_pts.dtype, device=device)

    t = mu_tgt - s * (R @ mu_src)

    return R, t, s


def _apply_transform(
    pts: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor
) -> torch.Tensor:
    return (s * (R @ pts.T)).T + t


@dataclass
class RansacResult:
    R: torch.Tensor
    t: torch.Tensor
    s: float
    inliers: torch.Tensor
    n_iters: int
    threshold: float


def _ransac(
    source_pts: torch.Tensor,
    target_pts: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    n_iters: int = 2000,
    threshold: Optional[float] = None,
    min_samples: Optional[int] = None,
    allow_scaling: bool = False,
    generator: Optional[torch.Generator] = None
) -> RansacResult:
    device = _get_device()
    src_pts = _as_tensor(source_pts, dtype=torch.float32, device=device)
    tgt_pts = _as_tensor(target_pts, dtype=torch.float32, device=device)

    N, d = src_pts.shape

    if min_samples is None:
        min_samples = d + 1
    if N < min_samples:
        raise ValueError(f"need >= {min_samples} correspondences, got {N}")

    if threshold is None:
        spread = torch.linalg.norm(tgt_pts - tgt_pts.mean(dim=0), dim=1)
        median_spread = torch.median(spread)
        threshold = (
            0.1 * median_spread.item()
            if median_spread > 0
            else 1.0
        )

    if weights is None:
        w_full = torch.ones(N, dtype=src_pts.dtype, device=device)
    else:
        w_full = _as_tensor(weights, dtype=src_pts.dtype, device=device)

    best_inliers = None
    best_count = -1

    for _ in range(n_iters):
        sample = torch.randperm(N, generator=generator, device=device)[:min_samples]

        try:
            R, t, s = _kabsch_umeyama(
                source_pts=src_pts[sample],
                target_pts=tgt_pts[sample],
                allow_scaling=allow_scaling
            )
        except torch.linalg.LinAlgError:
            continue

        pred = _apply_transform(src_pts, R, t, s)
        residuals = torch.linalg.norm(pred - tgt_pts, dim=1)
        inliers = residuals < threshold
        count = int(inliers.sum().item())

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or int(best_inliers.sum().item()) < min_samples:
        raise RuntimeError(
            f"RANSAC failed to find a consistent inlier set "
            f"(best consensus = {best_count}/{N}, need >= {min_samples})"
        )

    R, t, s = _kabsch_umeyama(
        source_pts=src_pts[best_inliers],
        target_pts=tgt_pts[best_inliers],
        weights=w_full[best_inliers],
        allow_scaling=allow_scaling
    )

    return RansacResult(
        R=R,
        t=t,
        s=s,
        inliers=best_inliers,
        n_iters=n_iters,
        threshold=threshold
    )