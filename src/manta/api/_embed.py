import anndata as ad
import torch

from torch_geometric.nn import radius
from typing import List, Literal, Optional

from ..core.encoders._graphsage_encoder import _embed
from ..core._features import (
    _compute_graph,
    _compute_base_features,
    _compute_gene_features,
    _compute_graph_features,
    _compute_microenvironment_features
)

from ..utils._gpu import _standardize
from ..utils._progress import (
    _get_progress,
    _update_progress,
)
from ..utils._tensor_utils import (
    _get_device,
    _as_tensor
)


def _compute_features(
    adata: ad.AnnData,

    spatial_key: str | None = None,
    sampling_key: str |  None = None,

    graph_key: str = "graph",
    base_features_key: str = "base_features",
    gene_features_key: str = "gene_features",
    graph_features_key: str = "graph_features",
    micro_features_key: str = "micro_features",
    feature_key: str = "section_features",

    pca_basis_key: str | None = None,
    nmf_basis_key: str | None = None,
    graph_k: int = 6,
    graph_alpha: float = 2.0,
    graph_features_k: int = 10,
    micro_env_radius: float = 50.0
) -> None:
    if sampling_key is None:
        raise ValueError("expected valid sampling_key, got None")

    sampling = adata.uns.get(sampling_key)
    if sampling is None:
        raise ValueError(
            f"expected sampling for key `{sampling_key}`, got None"
        )

    device = _get_device()
    indices = _as_tensor(sampling["indices"], dtype=torch.int64, device=device)

    _compute_graph(
        adata=adata,
        sampling_key=sampling_key,
        graph_key=graph_key,
        k=graph_k,
        alpha=graph_alpha
    )

    
    # TODO: Consider moving into micro-environment routine
    #       The result of this function isn't needed downstream
    _compute_base_features(
        adata=adata,
        pca_basis_key=pca_basis_key,
        nmf_basis_key=nmf_basis_key,
        feature_key=base_features_key
    )

    _compute_gene_features(
        adata=adata,
        graph_key=graph_key,
        pca_basis_key=pca_basis_key,
        nmf_basis_key=nmf_basis_key,
        feature_key=gene_features_key
    )
 
    _compute_graph_features(
        adata=adata,
        graph_key=graph_key,
        sampling_key=sampling_key,
        feature_key=graph_features_key,
        k=graph_features_k
    )

    _compute_microenvironment_features(
        adata=adata,
        spatial_key=spatial_key,
        sampling_key=sampling_key,
        base_features_key=base_features_key,
        feature_key=micro_features_key,
        micro_env_radius=micro_env_radius
    )

    gene_features       = adata.uns.get(gene_features_key)
    graph_features      = adata.uns.get(graph_features_key)
    micro_features      = adata.uns.get(micro_features_key)

    feature_raw = torch.cat(
        [
            gene_features['feature'],
            graph_features['feature'],
            micro_features['feature']
        ],
        dim=1
    )

    feature = _standardize(x=feature_raw)

    adata.uns[feature_key] = {
        "feature": feature,
        "indices": indices
    }


def embed(
    adatas: List[ad.AnnData],
    
    spatial_key: str | None = None,
    sampling_key: str |  None = None,

    pca_basis_key: str | None = None,
    nmf_basis_key: str | None = None,
    graph_k: int = 6,
    graph_alpha: float = 2.0,
    graph_features_k: int = 10,
    micro_env_radius: float = 50.0,

    hidden_dim: int = 128,
    decoder_hidden_dim: int = 64,
    num_layers: int = 2,
    activation: Literal["relu", "gelu"] = "gelu",
    dropout: float = 0.0,
    p_drop: float = 0.1,
    eta: float = 0.01,

    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    lambda_recon: float = 1.0,

    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 25,
    steps_per_epoch: int = 250,
    batch_size_per_graph: int = 256,
    grad_clip: Optional[float] = 1.0,
    shuffle_graph_order: bool = True,
    seed: int = 42,

    template_k: int = 30,
    template_iter: int = 25,
    template_temperature: float = 1.0,
) -> None:
    graph_key = "graph"
    feature_key = "features"
    embedding_key = "embedding"

    feature_progress, _ = _get_progress(
        steps=len(adatas) + 1,
        desc="Computing Features"
    )

    for adata in adatas:
        _update_progress(
            progress=feature_progress, 
            message=f"Computing Features"
        )

        _compute_features(
            adata=adata,

            spatial_key=spatial_key,
            sampling_key=sampling_key,
        
            graph_key=graph_key,
            base_features_key="base_features",
            gene_features_key="gene_features",
            graph_features_key="graph_features",
            micro_features_key="micro_features",
            feature_key=feature_key,
        
            pca_basis_key=pca_basis_key,
            nmf_basis_key=nmf_basis_key,
            graph_k=graph_k,
            graph_alpha=graph_alpha,
            graph_features_k=graph_features_k,
            micro_env_radius=micro_env_radius
        )

    _update_progress(
        progress=feature_progress, 
        message=f"Finished"
    )

    """
    _embed(
        adatas=adatas,

        sampling_key=sampling_key,
        graph_key=graph_key,
        feature_key=feature_key,
        embedding_key=embedding_key,

        hidden_dim=hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout=dropout,
        p_drop=p_drop,
        eta=eta,

        sim_coeff=sim_coeff,
        var_coeff=var_coeff,
        cov_coeff=cov_coeff,
        lambda_recon=lambda_recon,

        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size_per_graph=batch_size_per_graph,
        grad_clip=grad_clip,
        shuffle_graph_order=shuffle_graph_order,
        seed=seed,

        template_k=template_k,
        template_iter=template_iter,
        template_temperature=template_temperature,

        eps=1e-8
    )
    """