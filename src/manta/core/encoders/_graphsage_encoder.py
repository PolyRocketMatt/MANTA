import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import (
    List,
    Literal, 
    Optional, 
    Sequence, 
    Tuple
)

from ._graph_utils import (
    _GraphRepresentation,
    _sample_pooled_batch
)
from ...utils._gpu import (
    _chunked_range,
    _pairwise_dist,
    _kmeans
)
from ...utils._progress import (
    _get_progress,
    _update_progress,
    _update_postfix
)
from ...utils._tensor_utils import (
    _get_device,
    _check_tensor,
    _off_diag
)


class _FeatureAugmentor(nn.Module):
    def __init__(self,
                 p_drop: float = 0.1,
                 eta: float = 0.01) -> None:
        super().__init__()
        self.p_drop = p_drop
        self.eta = eta


    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand_like(x) > self.p_drop).to(torch.float32)
        noise = torch.randn_like(x) * self.eta
        return mask * x + noise


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._augment(x=x), self._augment(x=x)


class _ProjectionHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.0) -> None: 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 64,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _GraphSAGELayer(nn.Module):
    def __init__(self,
                 hidden_dim: int = 128,
                 activation: Literal["relu", "gelu"] = "gelu",
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Linear(3 * hidden_dim, hidden_dim) # self, m1 and m2 all have same hidden dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(
                f"unknown activation function `{activation}`"
            )


    def forward(self, h: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        m1 = torch.sparse.mm(P, h)
        m2 = torch.sparse.mm(P, m1)

        out = torch.cat([h, m1, m2], dim=-1)
        out = self.net(out)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        return h + out


class _Encoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 activation: Literal["relu", "gelu"] = "gelu", 
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.projector = _ProjectionHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.layers = nn.ModuleList(
            [
                _GraphSAGELayer(
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        h = self.projector(x)
        for layer in self.layers:
            h = layer(h, P)
        z = h

        return z


class _MantaEncoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 decoder_hidden_dim: int = 64,
                 num_layers: int = 2,
                 activation: Literal["relu", "gelu"] = "gelu",
                 dropout: float = 0.0,
                 p_drop: float = 0.1,
                 eta: float = 0.01) -> None:
        super().__init__()
        self.augmentor = _FeatureAugmentor(
            p_drop=p_drop,
            eta=eta
        )
        self.encoder = _Encoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout
        )
        self.decoder = _Decoder(
            in_dim=hidden_dim,
            out_dim=in_dim,
            hidden_dim=decoder_hidden_dim,
            dropout=dropout
        )


    def forward(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.augmentor(x)

        z1 = self.encoder(x1, P)
        z2 = self.encoder(x2, P)

        x1_rec = self.decoder(z1)

        return {
            "x1": x1,
            "x2": x2,
            "z1": z1,
            "z2": z2,
            "x1_rec": x1_rec
        }


    def infer(self, x: torch.Tensor, P: torch.Tensor):
        return self.encoder(x, P)


class _MantaEncoderLoss(nn.Module):
    def __init__(self,
                 sim_coeff: float = 25.0,
                 var_coeff: float = 25.0,
                 cov_coeff: float = 1.0,
                 lambda_recon: float = 1.0,
                 target_std: float = 1.0,
                 eps: float = 1e-8) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.lambda_recon = lambda_recon
        self.target_std = target_std
        self.eps = eps


    def _invariance_loss(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(z1, z2)


    def _variance_loss(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        var = z.var(dim=0, unbiased=False)
        std = torch.sqrt(var + self.eps)
        return torch.mean((F.relu(self.target_std - std)) ** 2)


    def _covariance_loss(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, keepdim=True) + self.eps)
        cov = (z.T @ z) / z.size(0)
        return _off_diag(cov).pow(2).sum() / z.size(1)


    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        x_true: torch.Tensor,
        x_rec: torch.Tensor,
    ) -> dict:
        # VICReg
        inv = self._invariance_loss(z1, z2)
        var = 0.5 * (self._variance_loss(z1) + self._variance_loss(z2))
        cov = 0.5 * (self._covariance_loss(z1) + self._covariance_loss(z2))

        # Reconstruction
        recon = F.mse_loss(x_rec, x_true)

        total = (
            self.sim_coeff * inv
            + self.var_coeff * var
            + self.cov_coeff * cov
            + self.lambda_recon * recon
        )

        return {
            "loss": total,
            "inv": inv.detach(),
            "var": var.detach(),
            "cov": cov.detach(),
            "recon": recon.detach()
        }


def _train(
    model: _MantaEncoder,
    graphs: Sequence[_GraphRepresentation],
    loss_fn: _MantaEncoderLoss,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 25,
    batch_size_per_graph: int = 256,
    steps_per_epoch: int = 250,
    grad_clip: Optional[float] = 1.0,
    shuffle_graph_order: bool = True,
    seed: int = 42
) -> dict:
    device = _get_device()
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs,
        eta_min=0.0
    )
    generator = torch.Generator(device).manual_seed(seed=seed)
    progress, set_postfix = _get_progress(
        bar=epochs,
        desc="Training"
    )

    history = {
        "loss": [],
        "inv": [],
        "var": [],
        "cov": [],
        "recon": []
    }

    # Move graphs ONCE to GPU
    prepared_graphs = [
        _GraphRepresentation(
            x=g.x.to(device, non_blocking=True),
            P=g.P.coalesce().to(device, non_blocking=True)
            if g.P.is_sparse
            else g.P.to(device, non_blocking=True)
        )
        for g in graphs
    ]

    for i in range(epochs):
        _update_progress(
            progress=progress,
            message=f"Training (Epoch {i + 1}/{epochs})"
        )

        model.train()

        epoch_loss = 0.0
        epoch_inv = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        epoch_recon = 0.0

        for _ in range(steps_per_epoch):
            # Graph order
            if shuffle_graph_order:
                graph_order = torch.randperm(len(prepared_graphs)).tolist()
            else:
                graph_order = list(range(len(prepared_graphs)))

            # Sampling batch
            batch = _sample_pooled_batch(
                graphs=prepared_graphs,
                nodes_per_graph=batch_size_per_graph,
                graph_order=graph_order,
                generator=generator
            )

            x = batch.x
            P = batch.P

            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            out = model(x, P)
            loss_dict = loss_fn(
                z1=out['z1'],
                z2=out['z2'],
                x_true=x,
                x_rec=out['x1_rec'],
                P=P
            )
            loss = loss_dict['loss']

            # Backward
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip
                )

            optimizer.step()

            # Bookkeeping
            epoch_loss += loss.item()
            epoch_inv += loss_dict['inv'].item()
            epoch_var += loss_dict['var'].item()
            epoch_cov += loss_dict['cov'].item()
            epoch_recon += loss_dict['recon'].item()

        # Scheduler update
        scheduler.step()

        # Averaging over epochs
        denom = float(steps_per_epoch)
        epoch_loss /= denom
        epoch_inv /= denom
        epoch_var /= denom
        epoch_cov /= denom
        epoch_recon /= denom

        # More bookkeeping
        history["loss"].append(epoch_loss)
        history["inv"].append(epoch_inv)
        history["var"].append(epoch_var)
        history["cov"].append(epoch_cov)
        history["recon"].append(epoch_recon)  

        _update_postfix(set_postfix, f"{epoch_loss:.3f}") 

    return {
        "model": model,
        "history": history
    }         


@torch.no_grad()
def _encode(
    adata: ad.AnnData,
    model: _MantaEncoder,
    sampling_key: str | None = None,
    graph_key: str | None = None,
    feature_key: str | None = None,
    embedding_key: str | None = None
) -> None:
    if model == None:
        raise ValueError("expected model to be of type `_MantaEncoder`, got `None`")

    if sampling_key == None:
        raise ValueError("expected valid sampling_key, got None")
    if graph_key == None:
        raise ValueError("expected valid graph_key, got None")
    if feature_key == None:
        raise ValueError("expected valid feature_key, got None")
    if embedding_key == None:
        raise ValueError("expected valid embedding_key, got None")

    sampling = adata.uns.get(sampling_key)
    if sampling == None:
        raise ValueError(
            f"expected sampling to be of type `dict`, got `None`"
        )

    s_pts = sampling['pts']
    s_indices = sampling['indices']
    _check_tensor(s_pts)
    _check_tensor(s_indices)

    graph = adata.uns.get(graph_key)
    if graph == None:
        raise ValueError(
            f"expected graph to be of type `dict`, got `None`"
        )

    P = graph['P']
    _check_tensor(P)

    feature = adata.uns.get(feature_key)
    if feature == None:
        raise ValueError(
            f"expected feature to be of type `dict`, got `None`"
        )    

    s_features = feature['feature']
    _check_tensor(s_features)

    # Encode with model (using inference)
    embedding = model.infer(s_features, P)

    adata.uns[embedding_key] = {
        "embedding": embedding,
        "pts": s_pts,
        "indices": s_indices
    }

    return embedding


@torch.no_grad()
def _build_latent_template(
    embedding: torch.Tensor,
    k: int,
    n_iter: int = 25
) -> torch.Tensor:
    _, centroids = _kmeans(
        x=embedding,
        n_clusters=k,
        n_iter=n_iter,
        dist_fn="euclidean"
    )
    return centroids


@torch.no_grad()
def _assign_to_latent_template(
    embedding: torch.Tensor,
    centroids: torch.Tensor,
    temperature: float = 1.0,
    batch_size: int = 4096,
    eps: float = 1e-8
) -> Tuple[torch.Tensor,...]:
    N, D = embedding.shape
    k = centroids.shape[0]
    device = embedding.device

    hard_cluster_ids = torch.empty(N, device=device, dtype=torch.long)
    soft_cluster_probs = torch.empty(N, k, device=device, dtype=torch.float32)

    hard_quantized = torch.empty(N, D, device=device, dtype=torch.float32)
    soft_quantized = torch.empty(N, D, device=device, dtype=torch.float32)

    inv_temp = 1.0 / max(temperature, eps)

    for s, e in _chunked_range(N, batch_size):
        dist = _pairwise_dist(embedding[s:e], centroids, "euclidean")
        logits = -dist * inv_temp
        probs = torch.softmax(logits, dim=1)

        hard_ids = probs.argmax(dim=1)

        hard_cluster_ids[s:e] = hard_ids
        soft_cluster_probs[s:e] = probs
        hard_quantized[s:e] = centroids[hard_ids]
        soft_quantized[s:e] = probs @ centroids

    return hard_cluster_ids, soft_cluster_probs, hard_quantized, soft_quantized


def _create_graph_representation(
    adata: ad.AnnData,
    graph_key: str | None = None,
    feature_key: str | None  = None,
) -> _GraphRepresentation:
    if graph_key == None:
        raise ValueError("expected valid graph_key, got None")
    if feature_key == None:
        raise ValueError("expected valid feature_key, got None")

    graph = adata.uns.get(graph_key)
    if graph == None:
        raise ValueError(
            f"expected graph to be of type `dict`, got `None`"
        )

    P = graph['P']
    _check_tensor(P)

    feature = adata.uns.get(feature_key)
    if feature == None:
        raise ValueError(
            f"expected feature to be of type `dict`, got `None`"
        )    

    s_features = feature['feature']
    _check_tensor(s_features)

    return _GraphRepresentation(s_features, P)


def _embed(
    adatas: List[ad.AnnData],

    sampling_key: str | None = None,
    graph_key: str | None = None,
    feature_key: str | None = None,
    embedding_key: str | None = None,

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

    eps: float = 1e-8
) -> None:
    graph_representations = [
        _create_graph_representation(
            adata=adata,
            graph_key=graph_key,
            feature_key=feature_key
        )
        for adata in adatas
    ]
    device = graph_representations[0].x.device

    # Make sure all graph representations have the same dimensionality
    expected_in_dims = graph_representations[0].x.shape[1]
    for g_repr in graph_representations:
        if g_repr.x.shape[1] != expected_in_dims:
            raise ValueError(
                f"expected input dimension to be {expected_in_dims}, got {g_repr.x.shape[1]} instead"
            )

    # Model
    model = _MantaEncoder(
        in_dim=expected_in_dims,
        hidden_dim=hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout=dropout,
        p_drop=p_drop,
        eta=eta
    )

    # Loss
    loss_fn = _MantaEncoderLoss(
        sim_coeff=sim_coeff,
        var_coeff=var_coeff,
        cov_coeff=cov_coeff,
        lambda_recon=lambda_recon,
        target_std=1.0,
        eps=eps
    )

    # Training
    result = _train(
        model=model,
        graphs=graph_representations,
        loss_fn=loss_fn,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size_per_graph=batch_size_per_graph,
        steps_per_epoch=steps_per_epoch,
        grad_clip=grad_clip,
        shuffle_graph_order=shuffle_graph_order,
        seed=seed
    )

    model = result['model']
    model.eval()

    # Inference on all (subsampled) points
    embeddings = []
    indices = [0]
    for adata in adatas:
        # Encode with model
        embedding = _encode(
            adata=adata,
            model=model,
            sampling_key=sampling_key,
            graph_key=graph_key,
            feature_key=feature_key,
            embedding_key=embedding_key
        )

        embeddings.append(embedding)
        indices.append(embedding.shape[0])

    # Stack all embeddings into one tensor to train k-means template
    stacked_embedding = torch.cat(embeddings, dim=0)

    # Train k-means clustering in latent space
    template_centroids = _build_latent_template(
        embedding=stacked_embedding,
        k=template_k,
        n_iter=template_iter
    )

    template_mappings = []
    for embedding in embeddings:
        mapping = _assign_to_latent_template(
            embedding=embedding,
            centroids=template_centroids,
            temperature=template_temperature,
            eps=eps
        )

        template_mappings.append(mapping)

    # Precompute centroid kernel matrix
    centroid_distances = torch.cdist(template_centroids, template_centroids) # [k, k]
    mask = ~torch.eye(centroid_distances.shape[0], device=device, dtype=torch.bool)
    tau = 0.5 * centroid_distances[mask].mean()
    diff = template_centroids[:, None, :] - template_centroids[None, :, :]
    Kmat = torch.exp(-(diff.pow(2).sum(-1)) / (2 * tau ** 2))

    clustering_key = f"{embedding_key}_clustering"
    for adata, template_mapping in zip(adatas, template_mappings):
        graph = adata.uns.get(graph_key)    # Already checked against None
        g_pts = graph['pts']
        g_indices = graph['indices']

        adata.uns[clustering_key] = {
            "hard_cluster_ids": template_mapping[0],
            "soft_cluster_probs": template_mapping[1],
            "hard_quantized": template_mapping[2],
            "soft_quantized": template_mapping[3],
            "centroids": template_centroids,
            "K": Kmat,
            "pts": g_pts,
            "indices": g_indices
        }