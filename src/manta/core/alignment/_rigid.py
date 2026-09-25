import anndata as ad
import numpy as np
import torch

from dataclasses import dataclass
from typing import Optional


def _voxelize(
    pts: np.array,
    bin_size: int,
) -> tuple[np.array, np.array]:
    origin = pts.min(axis=0)
    voxel_idx = np.floor((pts - origin) / bin_size).astype(np.int64)
    
    return voxel_idx, origin


def _aggregate(
    adata: ad.AnnData,
    bin_size: int,
    min_points: int = 10,
    normalize_expression: bool = True,
    spatial_key: str = 'spatial_manta',
    expression_key: str | None = None
) -> None:
    if expression_key is None:
        X = adata.X
    else:
        X = adata.obsm.get(expression_key)

    # Handle sparse matrix
    if hasattr(X, "toarray"):
        X = X.toarray() 
    X = np.asarray(X, dtype=float)

    pts = adata.obsm.get(spatial_key)
    pts = np.asarray(pts, dtype=float)
    
    voxel_idx, origin = _voxelize(pts, bin_size)

    keys, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
    n_voxels = keys.shape[0]

    expr_sum = np.zeros((n_voxels, X.shape[1]))
    pts_sum = np.zeros((n_voxels, pts.shape[1]))

    np.add.at(expr_sum, inverse, X)
    np.add.at(pts_sum, inverse, pts)

    expr_mean = expr_sum / counts[:, None]
    pts_mean = pts_sum / counts[:, None]

    if normalize_expression:
        norms = np.linalg.norm(expr_mean, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        expr_mean = expr_mean / norms

    mask = counts >= min_points

    adata.uns[f"voxel_{bin_size}"] = {
        "keys": keys[mask],
        "expr": expr_mean[mask],
        "centroid": pts_mean[mask],
        "counts": counts[mask].astype(float),
        "origin": origin
    }

    
def _match_voxels(
    source_expr: np.ndarray,
    target_expr: np.ndarray,
    mutual: bool = True,
    min_similarity: float = 0.0
) -> None:
    # Because we're checking similarity, make sure expression IS normalized
    src = source_expr / (np.linalg.norm(source_expr, axis=1, keepdims=True) + 1e-12)
    tgt = target_expr / (np.linalg.norm(target_expr, axis=1, keepdims=True) + 1e-12)

    sim = src @ tgt.T # (n_src, n_tgt)
    src_to_tgt = np.argmax(sim, axis=1)

    if mutual:
        tgt_to_src = np.argmax(sim, axis=0)
        is_mutual = tgt_to_src[src_to_tgt] == np.arange(src.shape[0])
        src_idx = np.where(is_mutual)[0]
    else:
        src_idx = np.arange(src.shape[0])

    tgt_idx = src_to_tgt[src_idx]
    sims = sim[src_idx, tgt_idx]
    mask = sims >= min_similarity

    return src_idx[mask], tgt_idx[mask], sims[mask]


def _kabsch_umeyama(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    weights: Optional[np.ndarray] = None,
    allow_scaling: bool = True
): 
    src_pts = np.asarray(source_pts, dtype=float)
    tgt_pts = np.asarray(target_pts, dtype=float)

    assert src_pts.shape == tgt_pts.shape

    N, d = src_pts.shape

    if weights is None:
        w = np.ones(N)
    else:
        w = np.asarray(weights, dtype=float)

    w = w / w.sum()

    mu_src = (w[:, None] * src_pts).sum(axis=0)
    mu_tgt = (w[:, None] * tgt_pts).sum(axis=0)

    src_c = src_pts - mu_src
    tgt_c = tgt_pts - mu_tgt

    cov = (tgt_c * w[:, None]).T @ src_c # weighed cross-covariance
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(d)

    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if allow_scaling:
        var_src = (w[:, None] * src_c ** 2).sum()
        s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 0 else 1.0
    else:
        s = 1.0

    t = mu_tgt - s * (R @ mu_src)
    return R, t, s


def _apply_transform(
    pts: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: np.ndarray
) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    return (s * (R @ pts.T)).T + t


@dataclass
class RansacResult:
    R: np.ndarray
    t: np.ndarray
    s: float
    inliers: np.ndarray
    n_iters: int
    threshold: float


def _ransac(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_iters: int = 2000,
    threshold: Optional[float] = None,
    min_samples: Optional[int] = None,
    allow_scaling: bool = False,
    generator: Optional[np.random.Generator] = None
) -> RansacResult:
    generator = np.random.default_rng() if generator is None else generator
    src_pts = np.asarray(source_pts, dtype=float)
    tgt_pts = np.asarray(target_pts, dtype=float)

    N, d = src_pts.shape

    if min_samples is None:
        min_samples = d + 1
    if N < min_samples:
        raise ValueError(f"need >= {min_samples} correspondences, got {N}")

    if threshold is None:
        spread = np.linalg.norm(tgt_pts - tgt_pts.mean(axis=0), axis=1)
        threshold = 0.1 * float(np.median(spread)) if np.median(spread) > 0 else 1.0

    w_full = np.ones(N) if weights is None else np.asarray(weights, dtype=float)

    best_inliers = None
    best_count = -1

    for _ in range(n_iters):
        sample = generator.choice(N, size=min_samples, replace=False)

        try:
            R, t, s = _kabsch_umeyama(
                source_pts=src_pts[sample],
                target_pts=tgt_pts[sample],
                allow_scaling=allow_scaling
            )
        except np.linalg.LinAlgError:
            continue

        pred = _apply_transform(src_pts, R, t, s)
        residuals = np.linalg.norm(pred - tgt_pts, axis=1)
        inliers = residuals < threshold
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < min_samples:
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