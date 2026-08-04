import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import List, Optional

from ..utils._tensor_utils import _from_tensor


def sampling(
    adatas: List[ad.AnnData],
    labels: Optional[List[str]],
    colors: Optional[List[str]],
    sampling_key: str | None = None,
    
    title: Optional[str] = None,
    xlabel: str = "",
    ylabel: str = "",

    axis_fontsize: int = 12,
    title_fontsize: int = 12,
    legend_fontsize: int = 12,

    filename: Optional[str] = None,
    out: Path = None,
    dpi: int = 300,

    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,

    figsize: tuple[float, float] = (6.0, 4.5),
    marker: str = "o",
    alpha: float = 0.5,
    marker_size: float = 12.0,
    edgecolor: Optional[str] = None,
    grid: bool = True,
    grid_alpha: float = 0.25,
    equal_aspect: bool = False,
    tight_layout: bool = True,
    transparent_bg: bool = False,
    show_axis: bool = True,
    show_legend: bool = True,
) -> None:
    if not sampling_key:
        raise ValueError("expected valid sampling_key, got None")
    if not colors:
        cmap = plt.get_cmap("tab20c")
        n = len(adatas)

        if n <= cmap.N:
            colors = [cmap(i) for i in range(n)]
        else:
            colors = [cmap(x) for x in np.linspace(0, 1, n)]

    if len(adatas) != len(colors):
        raise ValueError(
            f"expected {len(adatas)} colors, got {len(colors)}"
        )
    
    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = []

    for i, adata in enumerate(adatas):
        sampling = adata.uns.get(sampling_key)
        pts = sampling['pts']

        if pts is None:
            raise ValueError(
                f"expected coordinate array for key `{sampling_key}`, got None"
            )

        coords = _from_tensor(coords)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors[i],
            s=marker_size,
            marker=marker,
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=0.5 if edgecolor else 0.0,
        )

        if show_legend:
            legend_handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker=marker,
                    linestyle="",
                    color=colors[i],
                    label=labels[i] if labels else f'Sample {i}',
                )
            )

    if not show_axis:
        ax.axis("off")
    else:
        ax.set_xlabel(xlabel, fontsize=axis_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_fontsize)

        if title:
            ax.set_title(title, fontsize=title_fontsize, pad=12)

        if grid:
            ax.grid(True, alpha=grid_alpha)

        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(
            handles=legend_handles,
            fontsize=legend_fontsize,
            loc="best",
        )

    if tight_layout:
        fig.tight_layout()

    if filename:
        out.mkdir(parents=True, exist_ok=True)
        suffix = "png"
        fig.savefig(
            out / f"{filename}.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent_bg,
        )
    elif filename and show_if_save:
        plt.show()
    else:
        plt.show()

    plt.close()

