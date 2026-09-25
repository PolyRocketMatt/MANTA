from __future__ import annotations

import anndata as ad
import math
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import hsv_to_rgb
from pathlib import Path
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.decomposition import PCA
from typing import Literal, Optional

from ..utils._tensor_utils import _from_tensor


def _default_dataset_colors(n: int) -> list:
    """Generate visually distinct colors for datasets."""
    cmap = plt.get_cmap("tab20c")

    if n <= cmap.N:
        return [cmap(i) for i in range(n)]

    return [cmap(x) for x in np.linspace(0, 1, n)]


def _cluster_color_map(
    cluster_centroids: np.ndarray,
    cluster_keys: list,
    saturation: float = 0.80,
    value: float = 0.90,
) -> dict:
    n_clusters = len(cluster_keys)

    if n_clusters == 0:
        return {}

    if n_clusters == 1:
        return {
            cluster_keys[0]: hsv_to_rgb(
                (0.0, saturation, value)
            )
        }

    # Pairwise Euclidean distances between cluster centroids.
    diff = (
        cluster_centroids[:, None, :]
        - cluster_centroids[None, :, :]
    )
    distances = np.sqrt(np.sum(diff**2, axis=-1))

    # Minimum spanning tree.
    mst = minimum_spanning_tree(distances).toarray()

    # Make the graph symmetric.
    mst = np.maximum(mst, mst.T)

    # Choose a deterministic root.
    root = int(
        np.lexsort(
            (
                cluster_centroids[:, 1]
                if cluster_centroids.shape[1] > 1
                else np.zeros(n_clusters),
                cluster_centroids[:, 0],
            )
        )[0]
    )

    # Traverse MST.
    adjacency = [
        np.flatnonzero(mst[i] > 0).tolist()
        for i in range(n_clusters)
    ]

    order = []
    visited = set()

    def dfs(node: int) -> None:
        visited.add(node)
        order.append(node)

        # Deterministic child ordering.
        children = [
            child for child in adjacency[node]
            if child not in visited
        ]

        children.sort()
        for child in children:
            dfs(child)

    dfs(root)

    # Defensive fallback in case numerical issues produce disconnected
    # components.
    for i in range(n_clusters):
        if i not in visited:
            dfs(i)

    # Leave a small gap in hue space so that the final and first colors
    # are not forced to be neighbors.
    hues = np.linspace(
        0.0,
        0.90,
        n_clusters,
        endpoint=False,
    )

    colors = {}

    for hue, cluster_idx in zip(hues, order):
        colors[cluster_keys[cluster_idx]] = hsv_to_rgb(
            (hue, saturation, value)
        )

    return colors


def embedding(
    adatas: list[ad.AnnData],
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,

    clustering_key: str = "embedding_clustering",
    embedding_key: str = "embedding",

    # If "global", cluster ID 3 in adata 0 and cluster ID 3 in
    # adata 1 represent the same cluster.
    #
    # If "dataset", they are treated as distinct clusters.
    cluster_scope: Literal["global", "dataset"] = "global",

    # UMAP parameters.
    pca_components: Optional[int] = 50,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.3,
    umap_metric: str = "euclidean",
    umap_random_state: int = 0,

    # Cluster colors
    cluster_saturation: float = 0.80,
    cluster_value: float = 0.90,

    # Cluster-grid figure
    cluster_figsize_per_panel: tuple[float, float] = (4.0, 4.0),
    cluster_marker_size: float = 10.0,
    cluster_alpha: float = 0.7,

    # Latent-space figure
    latent_figsize: tuple[float, float] = (12.0, 5.5),
    latent_marker_size: float = 4.0,
    latent_alpha: float = 0.65,

    # General plotting options
    axis_fontsize: int = 12,
    title_fontsize: int = 12,
    legend_fontsize: int = 10,

    filename: Optional[str] = None,
    out: Optional[Path] = None,
    dpi: int = 300,

    grid: bool = True,
    grid_alpha: float = 0.25,
    equal_aspect: bool = False,
    tight_layout: bool = True,
    transparent_bg: bool = False,
    show_axis: bool = True,

    show_cluster_legend: bool = False,
    show_dataset_legend: bool = True,

    show: bool = True,
) -> None:
    if not adatas:
        raise ValueError("expected at least one AnnData object")

    n_adatas = len(adatas)

    if labels is None:
        labels = [f"Sample {i}" for i in range(n_adatas)]

    if len(labels) != n_adatas:
        raise ValueError(
            f"expected {n_adatas} labels, got {len(labels)}"
        )

    if colors is None:
        colors = _default_dataset_colors(n_adatas)

    if len(colors) != n_adatas:
        raise ValueError(
            f"expected {n_adatas} colors, got {len(colors)}"
        )

    if cluster_scope not in {"global", "dataset"}:
        raise ValueError(
            "cluster_scope must be either 'global' or 'dataset'"
        )


    embeddings = []
    points = []
    hard_clusters = []

    dataset_indices = []

    for dataset_idx, adata in enumerate(adatas):
        clustering = adata.uns.get(clustering_key)

        if clustering is None:
            raise ValueError(
                f"missing adata.uns['{clustering_key}']"
            )

        if "hard_cluster_ids" not in clustering:
            raise ValueError(
                f"missing '{clustering_key}['hard_cluster_ids']'"
            )

        if "pts" not in clustering:
            raise ValueError(
                f"missing '{clustering_key}['pts']'"
            )

        if embedding_key not in adata.uns:
            raise ValueError(
                f"missing adata.uns['{embedding_key}']"
            )

        latent = _from_tensor(adata.uns[embedding_key]["embedding"])
        cluster_ids = _from_tensor(clustering["hard_cluster_ids"]).astype(int)
        coords = _from_tensor(clustering["pts"])

        if latent.ndim != 2:
            raise ValueError(
                f"{embedding_key} must be 2D, got shape {latent.shape}"
            )

        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"'{clustering_key}['pts']' must have shape (n, >=2)"
            )

        if latent.shape[0] != len(cluster_ids):
            raise ValueError(
                f"adata {dataset_idx}: latent embedding has "
                f"{latent.shape[0]} observations but "
                f"hard_cluster_ids has {len(cluster_ids)}"
            )

        if coords.shape[0] != len(cluster_ids):
            raise ValueError(
                f"adata {dataset_idx}: pts has "
                f"{coords.shape[0]} observations but "
                f"hard_cluster_ids has {len(cluster_ids)}"
            )

        embeddings.append(latent.astype(np.float32, copy=False))
        points.append(coords[:, :2].astype(np.float32, copy=False))
        hard_clusters.append(cluster_ids)
        dataset_indices.append(
            np.full(len(cluster_ids), dataset_idx, dtype=int)
        )

    # All latent spaces need the same dimensionality.
    latent_dims = {x.shape[1] for x in embeddings}

    if len(latent_dims) != 1:
        raise ValueError(
            "all AnnData objects must have embeddings with the same "
            f"number of dimensions; got {sorted(latent_dims)}"
        )

    # Concatenate everything for the global UMAP.
    X = np.concatenate(embeddings, axis=0)
    cluster_ids_all = np.concatenate(hard_clusters, axis=0)
    dataset_ids_all = np.concatenate(dataset_indices, axis=0)

    # Determine cluster centroids
    #
    # We derive them from the actual latent embeddings rather than blindly
    # trusting the saved centroids. This makes the color assignment
    # consistent with the embeddings that are actually plotted.
    if cluster_scope == "global":
        cluster_keys = sorted(
            np.unique(cluster_ids_all).tolist()
        )

        cluster_centroids = np.vstack(
            [
                X[cluster_ids_all == cluster_id].mean(axis=0)
                for cluster_id in cluster_keys
            ]
        )

    else:
        # Dataset-specific cluster identity.
        cluster_keys = []
        centroid_list = []

        for dataset_idx in range(n_adatas):
            mask_dataset = dataset_ids_all == dataset_idx

            for cluster_id in sorted(
                np.unique(cluster_ids_all[mask_dataset]).tolist()
            ):
                mask = (
                    mask_dataset
                    & (cluster_ids_all == cluster_id)
                )

                cluster_keys.append(
                    (dataset_idx, int(cluster_id))
                )

                centroid_list.append(
                    X[mask].mean(axis=0)
                )

        cluster_centroids = np.vstack(centroid_list)

    cluster_colors = _cluster_color_map(
        cluster_centroids=cluster_centroids,
        cluster_keys=cluster_keys,
        saturation=cluster_saturation,
        value=cluster_value,
    )

    # Convert per-cell cluster IDs into colors.
    point_colors = []

    for i, cluster_id in enumerate(cluster_ids_all):
        dataset_idx = dataset_ids_all[i]

        if cluster_scope == "global":
            key = int(cluster_id)
        else:
            key = (int(dataset_idx), int(cluster_id))

        point_colors.append(cluster_colors[key])

    point_colors = np.asarray(point_colors)

    try:
        import umap.umap_ as umap
    except ImportError as exc:
        raise ImportError(
            "This function requires `umap-learn`. "
            "Install it with `pip install umap-learn`."
        ) from exc

    X_umap_input = X

    # PCA before UMAP is useful for high-dimensional latent spaces.
    # Setting pca_components=None disables this reduction.
    if (
        pca_components is not None
        and X.shape[1] > pca_components
    ):
        pca = PCA(
            n_components=pca_components,
            random_state=umap_random_state,
        )
        X_umap_input = pca.fit_transform(X)

    print(f"Computing UMAP...")
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_random_state,
    )

    X_umap = reducer.fit_transform(X_umap_input)
    ncols = min(5, n_adatas)
    nrows = math.ceil(n_adatas / ncols)

    panel_width, panel_height = cluster_figsize_per_panel

    fig_clusters, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            panel_width * ncols,
            panel_height * nrows,
        ),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for dataset_idx, (adata, coords, cluster_ids) in enumerate(
        zip(adatas, points, hard_clusters)
    ):
        ax = axes_flat[dataset_idx]

        # Use colors corresponding to cluster IDs.
        if cluster_scope == "global":
            colors_this = np.asarray(
                [
                    cluster_colors[int(cluster_id)]
                    for cluster_id in cluster_ids
                ]
            )
        else:
            colors_this = np.asarray(
                [
                    cluster_colors[
                        (dataset_idx, int(cluster_id))
                    ]
                    for cluster_id in cluster_ids
                ]
            )

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors_this,
            s=cluster_marker_size,
            alpha=cluster_alpha,
            linewidths=0,
            rasterized=True,
        )

        ax.set_title(
            labels[dataset_idx],
            fontsize=title_fontsize,
        )

        if show_axis:
            ax.set_xlabel(
                "x",
                fontsize=axis_fontsize,
            )
            ax.set_ylabel(
                "y",
                fontsize=axis_fontsize,
            )

            if grid:
                ax.grid(
                    True,
                    alpha=grid_alpha,
                )

            if equal_aspect:
                ax.set_aspect(
                    "equal",
                    adjustable="box",
                )
        else:
            ax.axis("off")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused panels.
    for ax in axes_flat[n_adatas:]:
        ax.axis("off")

    # Optional cluster legend.
    if show_cluster_legend:
        from matplotlib.lines import Line2D

        handles = []

        if cluster_scope == "global":
            legend_items = [
                (cluster_id, cluster_colors[cluster_id])
                for cluster_id in cluster_keys
            ]
        else:
            legend_items = [
                (
                    f"{labels[dataset_idx]} / {cluster_id}",
                    cluster_colors[(dataset_idx, cluster_id)],
                )
                for dataset_idx, cluster_id in cluster_keys
            ]

        for cluster_name, color in legend_items:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color,
                    markeredgecolor="none",
                    markersize=6,
                    label=str(cluster_name),
                )
            )

        fig_clusters.legend(
            handles=handles,
            fontsize=legend_fontsize,
            loc="lower center",
            ncol=min(5, len(handles)),
            bbox_to_anchor=(0.5, -0.01),
        )

    if tight_layout:
        fig_clusters.tight_layout()

    fig_latent, axes_latent = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=latent_figsize,
    )

    ax_cluster = axes_latent[0]
    ax_dataset = axes_latent[1]

    # Left: clusters.
    ax_cluster.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=point_colors,
        s=latent_marker_size,
        alpha=latent_alpha,
        linewidths=0,
        rasterized=True,
    )

    for cluster_id in sorted(np.unique(cluster_ids_all)):
        mask = cluster_ids_all == cluster_id

        x_center = np.mean(X_umap[mask, 0])
        y_center = np.mean(X_umap[mask, 1])

        ax_cluster.text(
            x_center,
            y_center,
            str(cluster_id),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            zorder=10,
        )

    ax_cluster.set_title(
        "Latent space — Clusters",
        fontsize=title_fontsize,
    )

    ax_cluster.set_xlabel(
        "UMAP 1",
        fontsize=axis_fontsize,
    )
    ax_cluster.set_ylabel(
        "UMAP 2",
        fontsize=axis_fontsize,
    )

    # Right: dataset identity.
    for dataset_idx, label in enumerate(labels):
        mask = dataset_ids_all == dataset_idx

        ax_dataset.scatter(
            X_umap[mask, 0],
            X_umap[mask, 1],
            c=[colors[dataset_idx]],
            s=latent_marker_size,
            alpha=latent_alpha,
            linewidths=0,
            rasterized=True,
            label=label,
        )

    ax_dataset.set_title(
        "Latent space — Dataset Partitioning",
        fontsize=title_fontsize,
    )

    ax_dataset.set_xlabel(
        "UMAP 1",
        fontsize=axis_fontsize,
    )
    ax_dataset.set_ylabel(
        "UMAP 2",
        fontsize=axis_fontsize,
    )

    if show_dataset_legend:
        ax_dataset.legend(
            fontsize=legend_fontsize,
            loc="best",
            frameon=True,
        )

    for ax in axes_latent:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if grid:
            ax.grid(
                True,
                alpha=grid_alpha,
            )

        if equal_aspect:
            ax.set_aspect(
                "equal",
                adjustable="box",
            )

        if not show_axis:
            ax.axis("off")

    if tight_layout:
        fig_latent.tight_layout()

    if filename is not None:
        if out is None:
            out = Path(".")

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig_clusters.savefig(
            out / f"{filename}_clusters.png",
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent_bg,
        )

        fig_latent.savefig(
            out / f"{filename}_latent.png",
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent_bg,
        )

    if show:
        plt.show()

    plt.close(fig_clusters)
    plt.close(fig_latent)


def cluster(
    adatas: list[ad.AnnData],
    clusterId: int,
    labels: Optional[list[str]] = None,

    clustering_key: str = "embedding_clustering",

    title: Optional[str] = None,
    xlabel: str = "",
    ylabel: str = "",

    # "Off" = observations not belonging to the selected cluster
    off_color: str = "#D9D9D9",

    # "On" = observations belonging to the selected cluster
    on_color: str = "#2563EB",

    axis_fontsize: int = 12,
    title_fontsize: int = 12,
    legend_fontsize: int = 10,

    filename: Optional[str] = None,
    out: Optional[Path] = None,
    dpi: int = 300,

    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,

    figsize_per_panel: tuple[float, float] = (4.0, 4.0),
    marker: str = "o",
    alpha: float = 0.7,
    marker_size: float = 12.0,

    grid: bool = True,
    grid_alpha: float = 0.25,
    equal_aspect: bool = False,
    tight_layout: bool = True,
    transparent_bg: bool = False,
    show_axis: bool = True,
    show_legend: bool = True,

    show: bool = True,
) -> None:
    if not adatas:
        raise ValueError("expected at least one AnnData object")

    n_adatas = len(adatas)

    if labels is None:
        labels = [f"Sample {i}" for i in range(n_adatas)]

    if len(labels) != n_adatas:
        raise ValueError(
            f"expected {n_adatas} labels, got {len(labels)}"
        )

    all_coords = []
    all_cluster_ids = []

    for dataset_idx, adata in enumerate(adatas):

        clustering = adata.uns.get(clustering_key)

        if clustering is None:
            raise ValueError(
                f"adata {dataset_idx} is missing "
                f"uns['{clustering_key}']"
            )

        if "pts" not in clustering:
            raise ValueError(
                f"adata {dataset_idx}: "
                f"uns['{clustering_key}'] is missing 'pts'"
            )

        if "hard_cluster_ids" not in clustering:
            raise ValueError(
                f"adata {dataset_idx}: "
                f"uns['{clustering_key}'] is missing "
                f"'hard_cluster_ids'"
            )

        coords = _from_tensor(clustering["pts"])
        cluster_ids = _from_tensor(
            clustering["hard_cluster_ids"]
        ).astype(int)

        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"adata {dataset_idx}: expected pts with shape "
                f"(n_obs, >=2), got {coords.shape}"
            )

        if coords.shape[0] != len(cluster_ids):
            raise ValueError(
                f"adata {dataset_idx}: pts contains "
                f"{coords.shape[0]} observations, while "
                f"hard_cluster_ids contains {len(cluster_ids)}"
            )

        all_coords.append(coords[:, :2])
        all_cluster_ids.append(cluster_ids)


    ncols = min(5, n_adatas)
    nrows = math.ceil(n_adatas / ncols)

    panel_width, panel_height = figsize_per_panel

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            panel_width * ncols,
            panel_height * nrows,
        ),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for dataset_idx in range(n_adatas):

        ax = axes_flat[dataset_idx]

        coords = all_coords[dataset_idx]
        cluster_ids = all_cluster_ids[dataset_idx]

        is_cluster = cluster_ids == clusterId

        # Plot all observations outside the selected cluster first.
        ax.scatter(
            coords[~is_cluster, 0],
            coords[~is_cluster, 1],
            color=off_color,
            s=marker_size,
            marker=marker,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
            label="Other clusters",
        )

        # Plot the selected cluster on top.
        ax.scatter(
            coords[is_cluster, 0],
            coords[is_cluster, 1],
            color=on_color,
            s=marker_size,
            marker=marker,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
            label=f"Cluster {clusterId}",
        )

        # Number of selected observations.
        n_selected = int(is_cluster.sum())
        n_total = len(is_cluster)

        ax.set_title(
            f"{labels[dataset_idx]}\n"
            f"Cluster {clusterId}: "
            f"{n_selected:,} / {n_total:,}",
            fontsize=title_fontsize,
            pad=10,
        )

        if show_axis:
            ax.set_xlabel(
                xlabel,
                fontsize=axis_fontsize,
            )

            ax.set_ylabel(
                ylabel,
                fontsize=axis_fontsize,
            )

            if grid:
                ax.grid(
                    True,
                    alpha=grid_alpha,
                )

            if equal_aspect:
                ax.set_aspect(
                    "equal",
                    adjustable="box",
                )

            if xlim is not None:
                ax.set_xlim(xlim)

            if ylim is not None:
                ax.set_ylim(ylim)

        else:
            ax.axis("off")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Only show one legend for the whole figure.
        if show_legend and dataset_idx == 0:
            ax.legend(
                fontsize=legend_fontsize,
                loc="best",
                frameon=True,
            )

    for ax in axes_flat[n_adatas:]:
        ax.axis("off")

    if title:
        fig.suptitle(
            title,
            fontsize=title_fontsize + 2,
            y=1.02,
        )

    if tight_layout:
        fig.tight_layout()

    if filename is not None:

        if out is None:
            out = Path(".")

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            out / f"{filename}.png",
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent_bg,
        )

    if show:
        plt.show()

    plt.close(fig)