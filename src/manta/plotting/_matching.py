from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import Optional

from ..utils._tensor_utils import _from_tensor


def matching(
    source: ad.AnnData,
    target: ad.AnnData,

    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,

    spatial_key: str = "spatial_manta",
    matching_key: str = "matching",

    title: Optional[str] = None,
    xlabel: str = "",
    ylabel: str = "",

    axis_fontsize: int = 12,
    title_fontsize: int = 12,
    legend_fontsize: int = 12,

    filename: Optional[str] = None,
    out: Optional[Path] = None,
    dpi: int = 300,

    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,

    figsize: tuple[float, float] = (6.0, 5.0),

    marker: str = "o",
    alpha: float = 0.35,
    marker_size: float = 8.0,

    edgecolor: Optional[str] = None,

    # Barycentric projection appearance.
    projection_color: str = "#84cc16",
    projection_marker: str = "o",
    projection_size: float = 10.0,
    projection_alpha: float = 0.8,

    # Arrow appearance.
    arrow_color: str = "#111827",
    arrow_alpha: float = 0.20,
    arrow_width: float = 0.0015,
    arrow_headwidth: float = 3.5,
    arrow_headlength: float = 4.5,
    arrow_headaxislength: float = 3.5,

    # Maximum number of arrows to draw. The highest-confidence
    # matches are retained. Set to None to draw all arrows.
    max_arrows: Optional[int] = None,

    # Temperature for converting scores to weights:
    #
    #   weights = softmax(scores / temperature)
    #
    # temperature=1.0 corresponds to the usual softmax.
    temperature: float = 1.0,

    grid: bool = True,
    grid_alpha: float = 0.25,
    equal_aspect: bool = False,
    tight_layout: bool = True,
    transparent_bg: bool = False,
    show_axis: bool = True,
    show_legend: bool = True,

    show: bool = True,
) -> None:
    if labels is None:
        labels = ["Source", "Target"]

    if len(labels) != 2:
        raise ValueError(
            f"expected 2 labels [source, target], got {len(labels)}"
        )

    if colors is None:
        colors = [
            "#2563EB",  # blue
            "#D9D9D9",  # gray
        ]

    if len(colors) != 2:
        raise ValueError(
            f"expected 2 colors [source, target], got {len(colors)}"
        )

    if temperature <= 0:
        raise ValueError(
            f"temperature must be > 0, got {temperature}"
        )

    source_coords = source.obsm.get(spatial_key)
    target_coords = target.obsm.get(spatial_key)

    if source_coords is None:
        raise ValueError(
            f"source.obsm['{spatial_key}'] was not found"
        )

    if target_coords is None:
        raise ValueError(
            f"target.obsm['{spatial_key}'] was not found"
        )

    source_coords = _from_tensor(source_coords)
    target_coords = _from_tensor(target_coords)

    embedding = source.uns.get("embedding")
    embedding_indices = None
    if embedding is not None:
        embedding_indices = _from_tensor(embedding["indices"]).astype(int)

    if source_coords.ndim != 2 or source_coords.shape[1] < 2:
        raise ValueError(
            "source spatial coordinates must have shape "
            f"(n_obs, >= 2), got {source_coords.shape}"
        )

    if target_coords.ndim != 2 or target_coords.shape[1] < 2:
        raise ValueError(
            "target spatial coordinates must have shape "
            f"(n_obs, >= 2), got {target_coords.shape}"
        )

    source_coords = source_coords[:, :2]
    target_coords = target_coords[:, :2]

    matching = source.uns.get(matching_key)

    if matching is None:
        raise ValueError(
            f"source.uns['{matching_key}'] was not found"
        )

    required_keys = [
        "source_idx",
        "target_idx",
        "scores",
    ]

    for key in required_keys:
        if key not in matching:
            raise ValueError(
                f"source.uns['{matching_key}'] is missing '{key}'"
            )

    source_idx = _from_tensor(
        matching["source_idx"]
    ).astype(int)

    target_idx = _from_tensor(
        matching["target_idx"]
    ).astype(int)

    scores = _from_tensor(
        matching["scores"]
    ).astype(float)

    if source_idx.ndim != 1:
        raise ValueError(
            f"source_idx must be 1D, got shape {source_idx.shape}"
        )

    if target_idx.ndim != 2:
        raise ValueError(
            f"target_idx must be 2D, got shape {target_idx.shape}"
        )

    if scores.ndim != 2:
        raise ValueError(
            f"scores must be 2D, got shape {scores.shape}"
        )

    if target_idx.shape != scores.shape:
        raise ValueError(
            "target_idx and scores must have the same shape; "
            f"got {target_idx.shape} and {scores.shape}"
        )

    if len(source_idx) != target_idx.shape[0]:
        raise ValueError(
            "source_idx and target_idx disagree on the number "
            f"of source points: {len(source_idx)} vs "
            f"{target_idx.shape[0]}"
        )

    
    if np.any(source_idx < 0) or np.any(
        source_idx >= len(source_coords)
    ):
        raise IndexError(
            "matching contains source indices outside "
            "source spatial coordinates"
        )

    if np.any(target_idx < 0) or np.any(
        target_idx >= len(target_coords)
    ):
        raise IndexError(
            "matching contains target indices outside "
            "target spatial coordinates"
        )

    scaled_scores = scores / temperature

    row_max = np.max(
        scaled_scores,
        axis=1,
        keepdims=True,
    )

    exp_scores = np.exp(
        scaled_scores - row_max
    )

    weights = exp_scores / np.sum(
        exp_scores,
        axis=1,
        keepdims=True,
    )

    target_for_source = target_coords[target_idx]

    projections = np.sum(
        target_for_source * weights[..., None],
        axis=1,
    )

    # Source positions corresponding to each row in matching.
    matched_source_coords = source_coords[source_idx]

    confidence = np.max(
        weights,
        axis=1,
    )

    n_matches = len(source_idx)

    if max_arrows is None or max_arrows >= n_matches:
        arrow_indices = np.arange(n_matches)

    elif max_arrows <= 0:
        arrow_indices = np.array([], dtype=int)

    else:
        # Highest-confidence matches first.
        arrow_indices = np.argsort(
            confidence
        )[::-1][:max_arrows]

    # Sort selected arrows by source index so their rendering order
    # is deterministic.
    arrow_indices = np.sort(arrow_indices)

    arrow_sources = matched_source_coords[arrow_indices]
    arrow_targets = projections[arrow_indices]

    arrow_dx = (
        arrow_targets[:, 0]
        - arrow_sources[:, 0]
    )

    arrow_dy = (
        arrow_targets[:, 1]
        - arrow_sources[:, 1]
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        target_coords[:, 0],
        target_coords[:, 1],
        c=colors[1],
        s=marker_size,
        marker=marker,
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=0.5 if edgecolor else 0.0,
        rasterized=True,
        label=labels[1],
    )

    if embedding_indices is not None:
        ax.scatter(
            source_coords[embedding_indices, 0],
            source_coords[embedding_indices, 1],
            c=colors[0],
            s=marker_size,
            marker=marker,
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=0.5 if edgecolor else 0.0,
            rasterized=True,
            label=labels[0],
        )
    else:
        ax.scatter(
            source_coords[:, 0],
            source_coords[:, 1],
            c=colors[0],
            s=marker_size,
            marker=marker,
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=0.5 if edgecolor else 0.0,
            rasterized=True,
            label=labels[0],
        )

    ax.scatter(
        projections[:, 0],
        projections[:, 1],
        c=projection_color,
        s=projection_size,
        marker=projection_marker,
        alpha=projection_alpha,
        edgecolors="none",
        rasterized=True,
        label="Barycentric projection",
    )

    if len(arrow_indices) > 0:

        ax.quiver(
            arrow_sources[:, 0],
            arrow_sources[:, 1],
            arrow_dx,
            arrow_dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=arrow_color,
            alpha=arrow_alpha,
            width=arrow_width,
            headwidth=arrow_headwidth,
            headlength=arrow_headlength,
            headaxislength=arrow_headaxislength,
            pivot="tail",
            zorder=4,
        )

    if not show_axis:
        ax.axis("off")

    else:
        ax.set_xlabel(
            xlabel,
            fontsize=axis_fontsize,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=axis_fontsize,
        )

        if title:
            ax.set_title(
                title,
                fontsize=title_fontsize,
                pad=12,
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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(
            fontsize=legend_fontsize,
            loc="best",
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