import anndata as ad
import torch



def _match(
    source: ad.AnnData,
    target: ad.AnnData,

    embedding_key: str | None = None,

    top_n_clusters: int = 5,
    top_k_matches: int = 10,
    batch_size: int = 4096,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    temperature: float = 1.0,

    # OT Hyperparameters
    epsilon: float = 0.05,
    rho_src: float = 1.0,
    rho_tgt: float = 1.0,
    num_sinkhorn_iters: int = 50,
    eps: float = 1e-8    
):
    src_embedding = source.uns.get(embedding_key)
    tgt_embedding = target.uns.get(embedding_key)
    
    if src_embedding is None:
        raise ValueError(
            f"expected embedding to be of type `dict`, got `None`"
        )
    if tgt_embedding is None:
        raise ValueError(
            f"expected embedding to be of type `dict`, got `None`"
        )

    
