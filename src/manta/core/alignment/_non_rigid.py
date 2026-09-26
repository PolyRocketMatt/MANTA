import anndata as ad
import math
import torch
import torch.nn.functional as F

from typing import List, Tuple

from ...utils._gpu import (
    _chunked_range
)
from ...utils._tensor_utils import (
    _get_device,
    _as_tensor
)


NEG_INF = float("-inf")


def _scatter_logsumexp(
    values: torch.Tensor,
    index: torch.Tensor,
    size: int
) -> torch.Tensor:
    device = _get_device()
    max_vals = values.new_full((size,), NEG_INF, device=device)
    max_vals.scatter_reduce_(0, index, values, reduce="amax", include_self=True)

    # Shifted exp-sum
    shift = max_vals[index]
    shifted = (values - shift).exp()
    exp_sum = values.new_zeros(size)
    exp_sum.scatter_add_(0, index, shifted)

    return max_vals + exp_sum.clamp(min=1e-40).log()


def _build_sparse_transport_costs(
    src_pts: torch.Tensor, src_z: torch.Tensor, src_probs: torch.Tensor,
    tgt_pts: torch.Tensor, tgt_z: torch.Tensor, tgt_probs: torch.Tensor,
    tgt_cluster: torch.Tensor,
    cluster_buckets: List[torch.Tensor],
    K: torch.Tensor,
    top_n_clusters: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    batch_size: int = 4096,
    eps: float = 1e-8
) -> Tuple[torch.Tensor,...]: 
    device = _get_device()
    
    N = src_pts.shape[0]
    C = src_probs.shape[1]
    k = min(top_n_clusters, C)

    all_src, all_tgt, all_cost = [], [], []
    for s, e in _chunked_range(N, batch_size):
        xs = src_pts[s:e]
        zs = src_z[s:e]
        ps = src_probs[s:e]

        # Select top-n clusters
        top_clusters = torch.topk(ps, k=k, dim=1, largest=True, sorted=False).indices
        unique_c = torch.unique(top_clusters.reshape(-1))
        candidate_buckets = [
            cluster_buckets[int(c)]
            for c in unique_c.tolist()
            if cluster_buckets[int(c)].numel() > 0
        ]

        # No candidate buckets found any point in the range
        if not candidate_buckets:
            continue

        candidate_idx = torch.unique(torch.cat(candidate_buckets))

        xt = tgt_pts[candidate_idx]
        zt = tgt_z[candidate_idx]
        pt = tgt_probs[candidate_idx]

        candidate_cluster = tgt_cluster[candidate_idx]

        # Compute geometric penalties for points in candidate clusters
        xx = (xs ** 2).sum(1, keepdim=True)
        yy = (xt ** 2).sum(1).unsqueeze(0)
        dist2 = (xx + yy - 2.0 * (xs @ xt.T)).clamp_min(0.0)    # Matrix-like "dot" product
    
        # Use a soft geometric falloff, not a hard filter
        sigma = dist2.mean().sqrt().clamp_min(eps)

        # TODO: Make sure this geometric penalty actually WORKS?
        # If not, resort to "geometric_penalty = -dist2"
        geometric_pentalty  = torch.exp(-dist2 / (2.0 * sigma ** 2))
        embedding_penalty   = zs @ zt.T
        cluster_penalty     = (ps @ K) @ pt.T

        # Standardize the penalties (unbiased, as otherwise std() potentially returns 0, resulting in NaN)
        geometric_pentalty  = (geometric_pentalty - geometric_pentalty.mean()) / (geometric_pentalty.std(unbiased=False).clamp_min(eps))
        embedding_penalty   = (embedding_penalty - embedding_penalty.mean()) / (embedding_penalty.std(unbiased=False).clamp_min(eps))
        cluster_penalty     = (cluster_penalty - cluster_penalty.mean()) / (cluster_penalty.std(unbiased=False).clamp_min(eps))

        # Combine scores based on weights
        # In older implementation, alpha weighs embedding and beta the spatial score, this is kept for support.
        scores = alpha * embedding_penalty + beta * geometric_pentalty + gamma * cluster_penalty

        # OLD
        # scores = scores + 0.5 * torch.log(geometric_pentalty.clamp_min(eps))

        # Cluster-based candidate mask
        candidate_mask = (top_clusters[:, :, None] == candidate_cluster[None, None, :]).any(dim=1)

        # Prefer targets in candidate clusters and which are geometrically close
        soft_mask = candidate_mask.float() * torch.exp(-dist2 / (2 * sigma**2))

        # Get mapping
        s_idx, t_idx = torch.where(soft_mask > 1e-3)
        if s_idx.numel() == 0:
            continue    # No valid mappings

        all_src.append((s + s_idx).to(torch.int32))
        all_tgt.append(candidate_idx[t_idx].to(torch.int32))
        all_cost.append(-scores[s_idx, t_idx])

    if not all_src:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        return empty, empty, torch.zeros(0, dtype=torch.float32, device=device)

    src_pairs = torch.cat(all_src)
    tgt_pairs = torch.cat(all_tgt)
    costs = torch.cat(all_cost)

    return src_pairs, tgt_pairs, costs


def _unbalanced_entropic_ot(
    src_pairs: torch.Tensor,
    tgt_pairs: torch.Tensor,
    costs: torch.Tensor,
    N: int,
    M: int,
    rho_src: float,
    rho_tgt: float,
    epsilon: float,
    num_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = _get_device()
    
    src_idx = src_pairs.long()
    tgt_idx = tgt_pairs.long()

    # Uniform marginal log-weights
    log_a = torch.full((N,), -math.log(N), dtype=torch.float32, device=device)
    log_b = torch.full((M,), -math.log(M), dtype=torch.float32, device=device)

    # Damping factors
    tau_src = rho_src / (rho_src + epsilon)
    tau_tgt = rho_tgt / (rho_tgt + epsilon)

    # Dual potentials
    f = torch.zeros(N, dtype=torch.float32, device=device)
    g = torch.zeros(M, dtype=torch.float32, device=device)

    inv_eps = 1.0 / epsilon
    for _ in range(num_iters):
        lse_src = _scatter_logsumexp(
            values=(g[tgt_idx] - costs) * inv_eps,
            index=src_idx,
            size=N
        )
        f = tau_src * (epsilon * (log_a - lse_src))

        lse_tgt = _scatter_logsumexp(
            values=(f[src_idx] - costs) * inv_eps,
            index=tgt_idx,
            size=M
        )
        g = tau_tgt * (epsilon * (log_b - lse_tgt))

    return f, g


def _extract_topk(
    src_pairs: torch.Tensor,
    tgt_pairs: torch.Tensor,
    log_pi: torch.Tensor,
    tgt_idx: torch.Tensor,
    N: int,
    top_k: int = 5,
    temperature: float = 1.0
) -> Tuple[torch.Tensor,...]:
    device = _get_device()

    # Two-pass lexicographic sort
    #   Pass 1 - Sort by log-pi (descending)
    order = torch.argsort(log_pi, descending=True)
    s_src   = src_pairs[order].long()
    s_tgt   = tgt_pairs[order].long()
    s_lp    = log_pi[order]

    #   Pass 2 - Stable sort by source s.t. same-source edges stay in log-pi order
    order2 = torch.argsort(s_src, stable=True)
    s_src   = s_src[order2]
    s_tgt   = s_tgt[order2]
    s_lp    = s_lp[order2]

    E = s_src.shape[0]

    # Within-group rank using cummax boundary trick
    positions = torch.arange(E, device=device)
    is_new = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        s_src[1:] != s_src[:-1]
    ])

    group_start = torch.cummax(torch.where(is_new, positions, positions.new_zeros(1)), dim=0).values
    rank = positions - group_start

    # Keep only top-k entries per source
    keep_mask = rank < top_k
    k_src   = s_src[keep_mask]
    k_tgt   = tgt_idx[s_tgt[keep_mask]]
    k_lp    = s_lp[keep_mask]
    k_rank  = rank[keep_mask]

    # Scatter into dense output tensors
    out_tgt = torch.full((N, top_k), -1, dtype=torch.long, device=device)
    out_scores = torch.full((N, top_k), NEG_INF, dtype=torch.float32, device=device)

    out_tgt[k_src, k_rank] = k_tgt
    out_scores[k_src, k_rank] = k_lp

    # Softmaxed probabilities over retained top-k
    valid = out_tgt >= 0
    soft = out_scores.masked_fill(~valid, -1e9) / temperature
    probs = torch.softmax(soft, dim=1).masked_fill(~valid, 0.0)
    probs = torch.where(valid.any(dim=1, keepdim=True), probs, torch.zeros_like(probs))

    return out_tgt, out_scores, probs


def _match(
    source: ad.AnnData,
    target: ad.AnnData,

    embedding_key: str | None = None,
    clustering_key: str | None = None,

    top_n_clusters: int = 5,
    top_k_matches: int = 10,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    temperature: float = 1.0,

    # OT Hyperparameters
    epsilon: float = 0.05,
    rho_src: float = 1.0,
    rho_tgt: float = 1.0,
    num_sinkhorn_iters: int = 50,
):
    if embedding_key is None:
        raise ValueError("expected valid embedding_key, got None")
    if clustering_key is None:
        raise ValueError("expected valid clustering_key, got None")
    
    device = _get_device()

    # Extract embedding
    src_embedding_dict = source.uns.get(embedding_key)
    tgt_embedding_dict = target.uns.get(embedding_key)
    
    if src_embedding_dict is None:
        raise ValueError(
            f"expected embedding to be of type `dict`, got `None`"
        )
    if tgt_embedding_dict is None:
        raise ValueError(
            f"expected embedding to be of type `dict`, got `None`"
        )

    src_embedding = _as_tensor(src_embedding_dict["embedding"], dtype=torch.float32, device=device)
    tgt_embedding = _as_tensor(tgt_embedding_dict["embedding"], dtype=torch.float32, device=device)

    src_z = F.normalize(src_embedding, dim=-1)
    tgt_z = F.normalize(tgt_embedding, dim=-1)
    src_pts = _as_tensor(src_embedding_dict["pts"], dtype=torch.float32, device=device)
    tgt_pts = _as_tensor(tgt_embedding_dict["pts"], dtype=torch.float32, device=device)
    src_idx = _as_tensor(src_embedding_dict["indices"], dtype=torch.int64, device=device)
    tgt_idx = _as_tensor(tgt_embedding_dict["indices"], dtype=torch.int64, device=device)


    # Extract clustering
    src_clustering_dict = source.uns.get(clustering_key)
    tgt_clustering_dict = target.uns.get(clustering_key)

    if src_clustering_dict is None:
        raise ValueError(
            f"expected clustering to be of type `dict`, got `None`"
        )
    if tgt_clustering_dict is None:
        raise ValueError(
            f"expected clustering to be of type `dict`, got `None`"
        )

    src_clustering_soft = _as_tensor(src_clustering_dict["soft_cluster_probs"], dtype=torch.float32, device=device)
    tgt_clustering_soft = _as_tensor(tgt_clustering_dict["soft_cluster_probs"], dtype=torch.float32, device=device)
    K = _as_tensor(src_clustering_dict["K"], dtype=torch.float32, device=device)

    N = src_pts.shape[0]
    M = tgt_pts.shape[0]
    C = src_clustering_soft.shape[1]

    # Assigns target observations to the highest probability cluster
    tgt_max_prob_idx = torch.argmax(tgt_clustering_soft, dim=1)
    cluster_buckets = [torch.where(tgt_max_prob_idx == c)[0] for c in range(C)]

    # Sparse candidate edges
    src_pairs, tgt_pairs, costs = _build_sparse_transport_costs(
        src_pts=src_pts, src_z=src_z, src_probs=src_clustering_soft,
        tgt_pts=tgt_pts, tgt_z=tgt_z, tgt_probs=tgt_clustering_soft,
        tgt_cluster=tgt_max_prob_idx,
        cluster_buckets=cluster_buckets,
        K=K,
        top_n_clusters=top_n_clusters,
        alpha=alpha, beta=beta, gamma=gamma,
    )

    # Degenerate case
    if src_pairs.numel() == 0:
        raise ValueError(f"no matching could be inferred")

    # Unbalanced sinkhorn to account for growth/shrinkage.
    f, g = _unbalanced_entropic_ot(
        src_pairs=src_pairs,
        tgt_pairs=tgt_pairs,
        costs=costs,
        N=N,
        M=M,
        rho_src=rho_src,
        rho_tgt=rho_tgt,
        epsilon=epsilon,
        num_iters=num_sinkhorn_iters
    )

    # Transport mass per edge
    log_pi = (f[src_pairs.long()] + g[tgt_pairs.long()] - costs) / epsilon

    # Top-k extraction
    out_tgt, out_scores, out_probs = _extract_topk(
        src_pairs=src_pairs,
        tgt_pairs=tgt_pairs,
        log_pi=log_pi,
        tgt_idx=tgt_idx,
        N=N,
        top_k=top_k_matches,
        temperature=temperature
    )

    transport_dict = {
        "source_idx": src_idx,
        "target_idx": out_tgt,
        "scores": out_scores,
        "probs": out_probs,
        "P": log_pi
    } 

    source.uns["matching"] = transport_dict
    target.uns["matching"] = transport_dict

    return transport_dict