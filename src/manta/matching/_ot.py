import math
import torch

from ..utils._gpu import _chunked_range

from typing import Tuple


def _scatter_logsumexp(
    values: torch.Tensor,
    index: torch.Tensor,
    size: int
) -> torch.Tensor:
    device = values.device
    max_vals = values.new_full((size,), float("-inf"), device=device)
    max_vals.scatter_reduce_(0, index, values, reduce="amax", include_self=True)

    # Shifted exp-sum
    shift = max_vals[index]
    shifted = (values - shift).exp()
    exp_sum = values.new_zeros(size)
    exp_sum.scatter_add_(0, index, shifted)

    return max_vals + exp_sum.clamp(min=1e-40).log()


@torch.no_grad()
def _sparse_unbalanced_sinkhorn(
    src_x: torch.Tensor,
    tgt_x: torch.Tensor,
    src_z: torch.Tensor,
    tgt_z: torch.Tensor,

    alpha: float = 1.0,
    beta: float = 1.0,

    n_candidates: int = 32,

    epsilon: float = 0.05,
    src_rho: float = 1.0,
    tgt_rho: float = 1.0,

    n_iters: int = 50,
    batch_size: int = 8192
) -> Tuple[torch.Tensor,...]:
    """
    Sparse unbalanced entropic OT between two point clouds

    Candidate edges: for each source point, the top-`n_candidates`
    target points by embedding cosine similarity.
    """
    device = src_x.device
    N, D = src_x.shape
    M = tgt_x.shape[0]

    src_z = torch.nn.functional.normalize(src_z, dim=-1)
    tft_z = torch.nn.functional.normalize(tgt_z, dim=-1)

    src_list, tgt_list, cost_list = [], [], []
    for s, e in _chunked_range(N, batch_size):
        zs = src_z[s:e]
        ps = src_x[s:e]

        sim = zs @ tgt_z.t()
        _, top_j = torch.topk(sim, k=min(n_candidates, M), dim=1)

        # Spatial distance
        pt = tgt_x[top_j]
        d2 = ((pt - ps.unsqueeze(1)) ** 2).sum(dim=-1)

        # Z-score distances, otherwise they blow up cost
        spatial_cost  = (d2 - d2.mean()) / (d2.std(unbiased=False).clamp_min(1e-8))

        # Cost: alpha * (1 - cosine) + beta * d2
        embedding_cost = 1.0 - sim.gather(1, top_j)
        costs = alpha * embedding_cost + beta * spatial_cost

        src_idx_local = torch.arange(s, e, device=device).unsqueeze(1).expand(-1, top_j.size(1))
        src_list.append(src_idx_local.reshape(-1))
        tgt_list.append(top_j.reshape(-1))
        cost_list.append(costs.reshape(-1))

    src_idx = torch.cat(src_list).long()
    tgt_idx = torch.cat(tgt_list).long()
    costs = torch.cat(cost_list).float()

    # Unbalanced sinkhorn
    log_a = torch.full((N,), -math.log(N), device=device)
    log_b = torch.full((M,), -math.log(M), device=device)

    src_tau = src_rho / (src_rho + epsilon)
    tgt_tau = tgt_rho / (tgt_rho + epsilon)

    f = torch.zeros(N, device=device)
    g = torch.zeros(M, device=device)

    inv_eps = 1.0 / epsilon
    for _ in range(n_iters):
        lse_src = _scatter_logsumexp((g[tgt_idx] - costs) * inv_eps, src_idx, N)
        f = src_tau * (epsilon * (log_a - lse_src))
        
        lse_tgt = _scatter_logsumexp((f[src_idx] - costs) * inv_eps, tgt_idx, M)
        g = tgt_tau * (epsilon * (log_b - lse_tgt))

    log_pi = (f[src_idx] + g[tgt_idx] - costs) * inv_eps
    return src_idx, tgt_idx, log_pi


def _compute_displacement_targets(
    src_x: torch.Tensor,
    tgt_x: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    log_pi: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Barycentric projection of transport plan onto target coordinates.
    """
    device = src_x.device
    N, D = src_x.shape

    w = log_pi.exp()
    tgt_weighted = tgt_x[tgt_idx] * w.unsqueeze(1)
    acc = torch.zeros(N, D, device=device)
    acc.index_add_(0, src_idx, tgt_weighted)
    row_sums = torch.zeros(N, device=device)
    row_sums.index_add_(0, src_idx, w)

    if normalize:
        acc = acc / row_sums.clamp(1e-12).unsqueeze(-1)

    return acc - src_x
