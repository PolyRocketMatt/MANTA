import anndata as ad
import numpy as np
import torch

from typing import Any, Dict, List, Optional

from ..core.alignment._rigid_cuda import (
    _aggregate,
    _match_voxels,
    _apply_transform,
    _ransac
)
from ..utils._progress import (
    _get_progress,
    _update_progress,
)
from ..utils._tensor_utils import (
    TensorLike,
    _get_device,
    _as_tensor,
    _from_tensor,
    _check_tensor
)


def rigid(
    source: ad.AnnData,
    target: ad.AnnData,
    voxel_scales: Optional[List[int]] = None,
    min_points_per_voxel: int = 25,
    min_similarity: float = 0.0,
    mutual_matching: bool = True,
    ransac_iters: int = 2000,
    ransac_treshold_frac: float = 0.5,
    allow_scaling: bool = False,
    weight_by_voxel_count: bool = True,
    seed: int = 42,
    spatial_key: str = "spatial_manta",
    expression_key: str | None = None,
):
    device = _get_device()
    dtype = torch.float32

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    if voxel_scales is None:
        pts = np.asarray(target.obsm.get(spatial_key), dtype=float)
        extent = (pts.max(axis=1) - pts.min(axis=1)).max()
        voxel_scales = [extent / f for f in [5, 10, 20, 40]]

    d = np.asarray(source.obsm.get(spatial_key)).shape[1]
    R_total = torch.eye(d, dtype=dtype, device=device)
    t_total = torch.zeros(d, dtype=dtype, device=device)
    s_total = torch.tensor(1.0, dtype=dtype, device=device)

    history: List[Dict[str, Any]] = []
    last_inlier_pair = None

    source.obsm["rigid"] = source.obsm.get(spatial_key)
    target.obsm["rigid"] = target.obsm.get(spatial_key)

    progress, _ = _get_progress(
        steps=len(voxel_scales) + 1,
        desc="Rigid Alignment"
    )

    for bin_size in sorted(voxel_scales, reverse=True):
        _update_progress(
            progress=progress, 
            message=f"Bin Size: {bin_size}"
        )

        # Revoxelize using the CURRENT aggregated transform
        src_transformed = _apply_transform(
            _as_tensor(
                source.obsm.get(spatial_key),
                dtype=dtype, 
                device=device
            ), 
            R_total, 
            t_total, 
            s_total
        )

        source_tmp = source.copy()
        source_tmp.obsm["rigid"] = _from_tensor(src_transformed)

        _aggregate(source_tmp, bin_size, min_points_per_voxel, spatial_key="rigid", expression_key=expression_key)
        _aggregate(target, bin_size, min_points_per_voxel, spatial_key="rigid", expression_key=expression_key)

        voxel_key = f"voxel_{bin_size}"
        src_vox = source_tmp.uns.get(voxel_key)
        tgt_vox = target.uns.get(voxel_key)

        src_idx, tgt_idx, sims = _match_voxels(
            src_vox["expr"],
            tgt_vox["expr"],
            mutual=mutual_matching,
            min_similarity=min_similarity
        )

        if len(src_idx) < d + 1:
            history.append(
                {
                    "bin_size": bin_size,
                    "status": "skipped; too few matches",
                    "n_matches": int(len(src_idx))
                }
            )

            continue


        src_pts = src_vox["centroid"][src_idx]
        tgt_pts = tgt_vox["centroid"][tgt_idx]
        match_weights = None
        if weight_by_voxel_count:
            match_weights = torch.minimum(
                src_vox["counts"][src_idx],
                tgt_vox["counts"][tgt_idx]
            ) 

        try:
            result = _ransac(
                source_pts=src_pts,
                target_pts=tgt_pts,
                weights=match_weights,
                n_iters=ransac_iters,
                threshold=ransac_treshold_frac * bin_size,
                allow_scaling=allow_scaling,
                generator=generator
            )
        except RuntimeError as e:
            history.append(
                {
                    "bin_size": bin_size,
                    "status": f"failed: {e}",
                    "n_matches": int(len(src_idx))
                }
            )

            continue

        # Compose delta (fit on already-transformed points) onto the running total
        R_total, t_total, s_total = (
            result.R @ R_total,
            result.s * (result.R @ t_total) + result.t,
            result.s * s_total
        )

        last_inlier_pair = (src_pts[result.inliers], tgt_pts[result.inliers])
        history.append(
            {
                "bin_size": bin_size,
                "status": "ok",
                "n_matches": int(len(src_idx)),
                "n_inliers": int(result.inliers.sum()),
                "inlier_ratio": float(result.inliers.float().mean()),
                "mean_cosine_sim_inliers": float(sims[result.inliers].mean())
            }
        )

    aligned_pts = _apply_transform(
        _as_tensor(source.obsm.get(spatial_key), dtype=dtype, device=device), 
        R_total, 
        t_total, 
        s_total
    )

    source.obsm["rigid"] = _from_tensor(aligned_pts)
    target.obsm["rigid"] = target.obsm.get(spatial_key)

    source.uns[f"rigid_alignment"] = {
        "R": R_total,
        "t": t_total,
        "s": s_total,
        "aligned_pts": aligned_pts,
        "history": history,
        "last_inlier_pair": last_inlier_pair
    }

    _update_progress(
        progress=progress, 
        message="Finished"
    )


def non_rigid():
    pass