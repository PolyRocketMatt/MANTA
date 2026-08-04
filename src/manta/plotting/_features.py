import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import List, Optional

from ..utils._tensor_utils import _from_tensor


def density(
    adatas: List[ad.AnnData],
    labels: Optional[List[str]],
    density_key: str | None = None,

    xlabel: str = "",
    ylabel: str = "",

    axis_fontsize: int = 12,
    title_fontsize: int = 12,

    filename: Optional[str] = None,
    out: Path = None,
    dpi: int = 300,

    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,

    figsize: tuple[float, float] = (10.0, 4.0),

    cmap: str = "viridis",

    marker: str = "o",
    marker_size: float = 12.0,
    edgecolor: Optional[str] = None,

    grid: bool = True,
    grid_alpha: float = 0.25,
    equal_aspect: bool = False,
    tight_layout: bool = True,
    transparent_bg: bool = False,
    show_axis: bool = True,
) -> None:

    if density_key is None:
        raise ValueError("expected valid density_key, got None")

    if labels is not None and len(labels) != len(adatas):
        raise ValueError(
            f"expected {len(adatas)} labels, got {len(labels)}"
        )

    coords_list = []
    density_list = []
    gradient_list = []

    # Collect data and convert gradients to magnitudes
    for adata in adatas:
        rho = adata.uns.get(density_key)

        if rho is None:
            raise ValueError(
                f"expected key `{density_key}` in adata.uns"
            )

        coords = rho.get("pts")
        densities = rho.get("rho")
        gradients = rho.get("dv")

        if coords is None:
            raise ValueError(
                f"expected coordinates for key `{density_key}`, got None"
            )

        if densities is None:
            raise ValueError(
                f"expected densities for key `{density_key}`, got None"
            )

        if gradients is None:
            raise ValueError(
                f"expected gradients for key `{density_key}`, got None"
            )

        coords = _from_tensor(coords)
        densities = np.asarray(_from_tensor(densities))

        gradients = np.asarray(_from_tensor(gradients))

        # Convert vector field to magnitude
        if gradients.ndim == 2:
            gradients = np.linalg.norm(
                gradients,
                axis=1,
            )

        coords_list.append(coords)
        density_list.append(densities)
        gradient_list.append(gradients)

    density_norm = Normalize(
        vmin=min(x.min() for x in density_list),
        vmax=max(x.max() for x in density_list),
    )

    gradient_norm = Normalize(
        vmin=min(x.min() for x in gradient_list),
        vmax=max(x.max() for x in gradient_list),
    )

    n = len(adatas)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=figsize,
        squeeze=False,
    )

    density_artists = []
    gradient_artists = []

    for i in range(n):

        ax_density = axes[i, 0]
        ax_gradient = axes[i, 1]

        coords = coords_list[i]

        density_artist = ax_density.scatter(
            coords[:, 0],
            coords[:, 1],
            c=density_list[i],
            cmap=cmap,
            norm=density_norm,
            s=marker_size,
            marker=marker,
            alpha=1.0,
            edgecolors=edgecolor,
            linewidths=0.5 if edgecolor else 0.0,
        )

        gradient_artist = ax_gradient.scatter(
            coords[:, 0],
            coords[:, 1],
            c=gradient_list[i],
            cmap=cmap,
            norm=gradient_norm,
            s=marker_size,
            marker=marker,
            alpha=1.0,
            edgecolors=edgecolor,
            linewidths=0.5 if edgecolor else 0.0,
        )

        density_artists.append(density_artist)
        gradient_artists.append(gradient_artist)

        if labels is not None:
            ax_density.set_ylabel(
                labels[i],
                fontsize=axis_fontsize,
            )

    axes[0, 0].set_title(
        "Density",
        fontsize=title_fontsize,
    )

    axes[0, 1].set_title(
        "Density Deriv.",
        fontsize=title_fontsize,
    )

    for ax_row in axes:
        for ax in ax_row:

            if not show_axis:
                ax.axis("off")
                continue

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

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Add colorbars matching subplot heights
    for i in range(n):

        divider = make_axes_locatable(
            axes[i, 0]
        )

        cax = divider.append_axes(
            "right",
            size="5%",
            pad=0.05,
        )

        fig.colorbar(
            density_artists[i],
            cax=cax,
        )

        divider = make_axes_locatable(
            axes[i, 1]
        )

        cax = divider.append_axes(
            "right",
            size="5%",
            pad=0.05,
        )

        fig.colorbar(
            gradient_artists[i],
            cax=cax,
        )

    if tight_layout:
        fig.tight_layout()

    if filename is not None:

        if out is None:
            raise ValueError(
                "out must be provided when filename is specified"
            )

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
    else:
        plt.show()

    plt.close(fig)


def multichannel(
    adatas: List[ad.AnnData],
    labels: Optional[List[str]],
    spatial_key: str | None = None,
    channel_key: str | None = None,

    xlabel: str = "",
    ylabel: str = "",

    axis_fontsize: int = 12,
    title_fontsize: int = 12,

    filename: Optional[str] = None,
    out: Path = None,
    dpi: int = 300,

    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,

    figsize: tuple[float, float] = (10.0, 12.0),

    cmap: str = "viridis",

    marker: str = "o",
    marker_size: float = 12.0,
    edgecolor: Optional[str] = None,

    grid: bool = True,
    grid_alpha: float = 0.25,
    equal_aspect: bool = False,
    tight_layout: bool = True,
    transparent_bg: bool = False,
    show_axis: bool = True,
) -> None:
    if labels is not None and len(labels) != len(adatas):
        raise ValueError(
            f"expected {len(adatas)} labels, got {len(labels)}"
        )

    coords = []
    components = []

    # Collect value matrices
    for adata in adatas:
        if spatial_key not in adata.obsm:
            raise ValueError(
                f"expected `{spatial_key}` in adata.obsm"
            )
        if channel_key not in adata.uns:
            raise ValueError(
                f"expected `{channel_key}` in adata.uns"
            )

        coords.append(np.asarray(adata.obsm[spatial_key]))
        components.append(np.asarray(_from_tensor(adata.uns[channel_key]['values'])))

    n_components = components[0].shape[1]
    for comp in components:
        if comp.shape[1] != n_components:
            raise ValueError(
                "all adatas must contain the same number of components"
            )

    n_adatas = len(adatas)

    # Normalize each component independently across datasets
    norms = []

    for k in range(n_components):

        values = np.concatenate(
            [
                comp[:, k]
                for comp in components
            ]
        )

        norms.append(
            Normalize(
                vmin=values.min(),
                vmax=values.max(),
            )
        )

    fig, axes = plt.subplots(
        nrows=n_components,
        ncols=n_adatas,
        figsize=figsize,
        squeeze=False,
    )

    artists = np.empty(
        (n_components, n_adatas),
        dtype=object,
    )

    for k in range(n_components):

        for i in range(n_adatas):

            ax = axes[k, i]

            artist = ax.scatter(
                coords[i][:, 0],
                coords[i][:, 1],
                c=components[i][:, k],
                cmap=cmap,
                norm=norms[k],
                s=marker_size,
                marker=marker,
                alpha=1.0,
                edgecolors=edgecolor,
                linewidths=0.5 if edgecolor else 0.0,
            )

            artists[k, i] = artist

            if k == 0:
                ax.set_title(
                    labels[i] if labels else f"Sample {i}",
                    fontsize=title_fontsize,
                )

            if i == 0:
                ax.set_ylabel(
                    f"Channel {k + 1}",
                    fontsize=axis_fontsize,
                )

            if not show_axis:
                ax.axis("off")
                continue

            ax.set_xlabel(
                xlabel,
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

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if xlim is not None:
                ax.set_xlim(xlim)

            if ylim is not None:
                ax.set_ylim(ylim)

    # Colorbar only on first column
    for k in range(n_components):

        ax = axes[k, 0]

        divider = make_axes_locatable(ax)

        cax = divider.append_axes(
            "right",
            size="5%",
            pad=0.05,
        )

        fig.colorbar(
            artists[k, 0],
            cax=cax,
        )

    if tight_layout:
        fig.tight_layout()

    if filename is not None:

        if out is None:
            raise ValueError(
                "out must be provided when filename is specified"
            )

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

    else:
        plt.show()

    plt.close(fig)