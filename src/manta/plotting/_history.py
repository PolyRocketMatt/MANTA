import anndata as ad
import matplotlib.pyplot as plt
import numpy as np


def rigid_history(adata: ad.AnnData):   
    history = adata.uns["rigid_alignment"]["history"]
    voxel_scales = np.array([h["bin_size"] for h in history])

    n_matches = np.array([
        h.get("n_matches", np.nan)
        for h in history
    ], dtype=float)

    n_inliers = np.array([
        h.get("n_inliers", np.nan)
        for h in history
    ], dtype=float)

    inlier_ratio = np.array([
        h.get("inlier_ratio", np.nan)
        for h in history
    ], dtype=float)

    mean_cosine_sim = np.array([
        h.get("mean_cosine_sim_inliers", np.nan)
        for h in history
    ], dtype=float)

    metrics = [
        (n_matches, "Number of matches", "Matches"),
        (n_inliers, "Number of inliers", "Inliers"),
        (inlier_ratio, "Inlier ratio", "Inlier ratio"),
        (mean_cosine_sim, "Mean cosine similarity", "Cosine similarity"),
    ]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(18, 6),
        sharey=True,
        constrained_layout=True
    )
    
    for ax, (values, title, xlabel) in zip(axes, metrics):
        ax.plot(values, voxel_scales, marker="o")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel("Bin size")
    
    # Metrics with natural [0, 1] range
    axes[2].set_xlim(0, 1)
    axes[3].set_xlim(0, 1)
    
    fig.suptitle("Rigid Alignment Across Voxel Scales", fontsize=14)
    plt.show()