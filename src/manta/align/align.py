import anndata as ad


from ..align._preprocessing import _preprocess


def preprocess(
    source: ad.AnnData,
    target: ad.AnnData,
    batch_key: str = "batch",
    n_components: int = 25,
    pca_basis_key: str = "X_pca",
    nmf_basis_key: str = "X_nmf",
    gene_key: str = "gene",
    spatial_key: str = "spatial",
    key_added: str = "spatial_manta",
) -> None:
    _preprocess(
        source=source,
        target=target,
        batch_key=batch_key,
        n_components=n_components,
        pca_basis_key=pca_basis_key,
        nmf_basis_key=nmf_basis_key,
        gene_key=gene_key,
        spatial_key=spatial_key,
        key_added=key_added
    )
    

def align_rigid(
    source: ad.AnnData,
    target: ad.AnnData,
):
    pass


def align_elastic():
    pass