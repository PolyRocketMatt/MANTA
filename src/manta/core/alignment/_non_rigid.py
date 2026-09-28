import anndata as ad
import math
import time
import torch
import torch.nn.functional as F

from dataclasses import dataclass, field
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import List, Literal, Tuple

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


def _log_normal(
    x: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor
) -> torch.Tensor:
    return -0.5 * (math.log(2.0 * math.pi) + var.log() + (x - mean).pow(2.0) / var)


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
    src_embedding_dict: dict,
    tgt_embedding_dict: dict,

    src_clustering_dict: dict,
    tgt_clustering_dict: dict,

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
    device = _get_device()

    src_embedding = _as_tensor(src_embedding_dict["embedding"], dtype=torch.float32, device=device)
    tgt_embedding = _as_tensor(tgt_embedding_dict["embedding"], dtype=torch.float32, device=device)

    src_z = F.normalize(src_embedding, dim=-1)
    tgt_z = F.normalize(tgt_embedding, dim=-1)
    src_pts = _as_tensor(src_embedding_dict["pts"], dtype=torch.float32, device=device)
    tgt_pts = _as_tensor(tgt_embedding_dict["pts"], dtype=torch.float32, device=device)
    src_idx = _as_tensor(src_embedding_dict["indices"], dtype=torch.int64, device=device)
    tgt_idx = _as_tensor(tgt_embedding_dict["indices"], dtype=torch.int64, device=device)

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

    return transport_dict


def _cubic_bspline(u: torch.Tensor) -> torch.Tensor:
    a = u.abs()
    out = torch.zeros_like(u)

    m1 = a < 1.0
    out[m1] = (2.0 / 3.0) - a[m1].pow(2) + 0.5 * a[m1].pow(3)

    m2 = (a >= 1.0) & (a < 2.0)
    out[m2] = (2.0 - a[m2]).pow(3) / 6.0

    return out


def _cubic_bspline_derivative(u: torch.Tensor) -> torch.Tensor:
    a = u.abs()
    out = torch.zeros_like(u)

    m1 = a < 1
    out[m1] = -2.0 * u[m1] + 1.5 * u[m1] * a[m1]

    m2 = (a >= 1.0) & (a < 2.0)
    out[m2] = -(2.0 - a[m2]).pow(2) / 2.0 * u[m2].sign()

    return out


# TODO - This needs to be converted to a dimensionless representation to work in 3D/4D
def _build_design_matrix(
    x: torch.Tensor,
    origin: torch.Tensor,
    h: float,
    l_x: int,
    l_y: int
) -> torch.Tensor:
    device = _get_device()
    u = (x - origin.unsqueeze(0)) / h

    l_idx = torch.arange(l_x, dtype=torch.float32, device=device)
    m_idx = torch.arange(l_y, dtype=torch.float32, device=device)

    # Evaluate B-spline kernel along each axis
    b_x = _cubic_bspline(u[:, 0:1] - l_idx.unsqueeze(0))
    b_y = _cubic_bspline(u[:, 1:2] - m_idx.unsqueeze(0))

    # 2D outer product per node
    B = (b_x.unsqueeze(2) * b_y.unsqueeze(2)).reshape(-1, l_x * l_y)
    return B


# TODO - This needs to be converted to a dimensionless representation to work in 3D/4D
def _build_design_matrix_derivative(
    x: torch.Tensor,
    origin: torch.Tensor,
    h: float,
    l_x: int,
    l_y: int,
    axis: int
) -> torch.Tensor:
    device = _get_device()
    u = (x - origin.unsqueeze(0)) / h

    l_idx = torch.arange(l_x, dtype=torch.float32, device=device)
    m_idx = torch.arange(l_y, dtype=torch.float32, device=device)

    b_x = _cubic_bspline(u[:, 0:1] - l_idx.unsqueeze(0))
    b_y = _cubic_bspline(u[:, 1:2] - m_idx.unsqueeze(0))

    db_x = _cubic_bspline_derivative(u[:, 0:1] - l_idx.unsqueeze(0))
    db_y = _cubic_bspline_derivative(u[:, 1:2] - m_idx.unsqueeze(0))

    if axis == 0:
        dB = ((db_x / h).unsqueeze(2) * b_y.unsqueeze(1)).reshape(-1, l_x * l_y)
    else:
        dB = (b_x.unsqueeze(2) * (db_y / h).unsqueeze(1)).reshape(-1, l_x * l_y)
    return dB


def _build_diff_operators(l_x: int, l_y: int) -> dict:
    device = _get_device()
    P = l_x * l_y
    I_x = torch.eye(l_x, dtype=torch.float32, device=device)
    I_y = torch.eye(l_y, dtype=torch.float32, device=device)

    def _diff1(L: int) -> torch.Tensor:
        D =  torch.zeros(max(L - 1, 0), L, dtype=torch.float32, device=device)
        if L > 1:
            idx = torch.arange(L - 1, device=device)
            D[idx, idx]     = -1.0
            D[idx, idx + 1] =  1.0
        return D

    def _diff2(L: int) -> torch.Tensor:
        D = torch.zeros(max(L - 2, 0), L, dtype=torch.float32, device=device)
        if L > 2:
            idx = torch.arange(L - 1, device=device)
            D[idx, idx]     =  1.0
            D[idx, idx + 1] = -2.0
            D[idx, idx + 2] = 1.0
        return D

    ops: dict = {}

    if l_x > 1 and l_y > 1:
        ops["l1x"] = torch.kron(_diff1(l_x), I_y)
        ops["l1y"] = torch.kron(I_x, _diff1(l_y))
    else:
        ops["L1x"] = torch.zeros(0, P, dtype=torch.float32, device=device)
        ops["L1y"] = torch.zeros(0, P, dtype=torch.float32, device=device)

    if l_x > 2 and l_y > 2:
        ops["l2x"] = torch.kron(_diff2(l_x), I_y)
        ops["l2y"] = torch.kron(I_x, _diff2(l_y))
    else:
        ops["l2x"] = torch.zeros(0, P, dtype=torch.float32, device=device)
        ops["l2y"] = torch.zeros(0, P, dtype=torch.float32, device=device)

    return ops


def _build_gmrf_prior(
    l_x: int,
    l_y: int,
    shape: Literal["bending", "membrane", "combined"] = "bending"
) -> torch.Tensor:
    device = _get_device()
    P = l_x * l_y
    ops = _build_diff_operators(l_x, l_y)

    L0 =  torch.zeros(P, P, dtype=torch.float32, device=device)

    if shape in ("membrane", "combined"):
        L0 += ops["l1x"].T @ ops["l1x"] + ops["l1y"].T @ ops["l1y"]

    if shape in ("bending", "combined"):
        L0 += ops["l2x"].T @ ops["l2x"] + ops["l2y"].T @ ops["l2y"]

    # Ridge to make sure L0 is positive-definite
    L0 += 1e-5 * torch.eye(P, dtype=torch.float32, device=device)

    return L0


def _geman_mcclure_weight(
    residual_sq: torch.Tensor,
    kappa: float,
    w_min: float = 0.0
) -> torch.Tensor:
    kappa = max(float(kappa), 1e-12)
    w = (kappa ** 2) / (kappa + residual_sq).pow(2).clamp(min=1e-24)
    return w.clamp(min=w_min) if w_min > 0.0 else w


@dataclass
class _NRRegistrationResult:
    """
    Container for outputs from _ProbabilisticRegistration.fit()
    """

    # Posterior field parameters
    mu_x:           torch.Tensor    # [P]   posterior mean of x-component control points
    mu_y:           torch.Tensor    # [P]   posterior mean of y-component control points
    v_x:            torch.Tensor    # [P]   posterior diagonal variance of phi_x
    v_y:            torch.Tensor    # [P]   posterior diagonal variance of phi_y

    # Per-node inlier confidence
    r:              torch.Tensor    # [N]   per-point responsibilities in [0, 1]

    # Learned hyperparameters
    pi0:            float           # final learned inlier rate
    sigma_in:       float           # final learned inlier noise std
    alpha:          float           # final learned regularisation scale

    # Diagnostics
    elbo_hist:      list            # ELBO per iteration
    delta_hist:     list            # In-between deformation at each CAVI iteration
    diff_hist:      list            # Error of source to target at each CAVI iteration
    inlier_hist:    list            # Percentage of being an inlier at each CAVI iteration
    elapsed:        float           # Wall-clock (in seconds) for the full fit

    # Grid metadata
    l_x:            int
    l_y:            int
    origin:         torch.Tensor    # [2]
    h:              float

    # Discontinuity-aware diagnostics (None unless discontinuity_aware=True)
    tear_weight_x:  torch.Tensor | None = None  # [rows(l1x)]   final tear gate, x-membrane edges
    tear_weight_y:  torch.Tensor | None = None  # [rows(l1y)]   final tear gate, y-membrane edges
    fold_weight_x:  torch.Tensor | None = None  # [rows(l2x)]   final fold gate, x-bending edges
    fold_weight_z:  torch.Tensor | None = None  # [rows(l2y)]   final fold gate, y-bending edges
    tear_gate_hist: list = field(default_factory=list)  # mean tear weight per iteration
    fold_gate_hist: list = field(default_factory=list)  # mean fold weight per iteration


class _ProbabilisticRegistration:
    def __init__(
        self,
        l_x: int = 32,
        l_y: int = 32,
        regularisation_shape: Literal["bending", "membrane", "combined"] = "bending",

        pi0_init: float = 0.8,
        sigma_in_init: float | None = None,
        alpha_init: float = 1.0,
        alpha_max: float = 1e6,
        
        barrier_alpha: float = 1e-3,
        barrier_lr: float = 0.5,
        barrier_steps: int = 3,

        tolerance: float = 1e-4,
        patience: int = 5,
        min_iters: int = 5,
        n_iters: int = 20,

        discontinuity_aware: bool = False,
        allow_tears: bool = True,
        allow_folds: bool = True,
        kappa_tear: float | None = None,
        kappa_fold: float | None = None,
        fold_barrier_suppression: bool = True
    ):
        self.l_x                    = l_x
        self.l_y                    = l_y
        self.reg_shape              = regularisation_shape

        self.pi0_init               = pi0_init
        self.sigma_in_init          = sigma_in_init
        self.alpha_init             = alpha_init
        self.alpha_max              = alpha_max

        self.barrier_alpha          = barrier_alpha
        self.barrier_lr             = barrier_lr
        self.barrier_steps          = barrier_steps

        self.tolerance              = tolerance
        self.patience               = patience
        self.min_iters              = min_iters
        self.n_iters                = n_iters

        self.discontinuity_aware    = discontinuity_aware
        self.allow_tears            = bool(discontinuity_aware and allow_tears)
        self.allow_folds            = bool(discontinuity_aware and allow_folds)
        self.kappa_tear             = kappa_tear
        self.kappa_fold             = kappa_fold
        self.barrier_suppression    = fold_barrier_suppression


    def _apply_internal_deformation(
        self,
        x: torch.Tensor,
        origin: torch.Tensor,
        h: float,
        l_x: int,
        l_y: int,
        mu_x: torch.Tensor,
        mu_y: torch.Tensor
    ) -> torch.Tensor:
        B = _build_design_matrix(
            x=x,
            origin=origin,
            h=h,
            l_x=l_x,
            l_y=l_y
        )

        return x + torch.stack([B @ mu_x, B @ mu_y], dim=1)


    def apply_deformation(
        self,
        x: torch.Tensor,
        result: _NRRegistrationResult
    ) -> torch.Tensor:
        return self._apply_internal_deformation(
            x=x,
            origin=result.origin,
            h=result.h,
            l_x=result.l_x,
            l_y=result.l_y,
            mu_x=result.mu_x,
            mu_y=result.mu_y
        )


    def posterior_displacement_std(
        self,
        x: torch.Tensor,
        result: _NRRegistrationResult
    ) -> torch.Tensor:
        B = _build_design_matrix(
            x=x,
            origin=result.origin,
            h=result.h,
            l_x=result.l_x,
            l_y=result.l_y
        )

        B2 = B.pow(2)
        std_x = (B2 @ result.v_x).sqrt()
        std_y = (B2 @ result.v_y).sqrt()

        return torch.stack([std_x, std_y], dim=1)


    def fit(
        self,
        src_x:          torch.Tensor,
        tgt_x:          torch.Tensor,
        tgt_scores:     torch.Tensor,
        use_softmax:    bool = True
    ) -> _NRRegistrationResult:
        device = _get_device()
        t0 = time.perf_counter()

        src_x       = src_x.to(torch.float32)
        tgt_x       = tgt_x.to(torch.float32) 
        tgt_scores  = tgt_scores.to(torch.float32)

        # Compute displaacement targets
        delta = self._compute_displacement_targets(
            src_x=src_x,
            tgt_x=tgt_x,
            tgt_scores=tgt_scores,
            use_softmax=use_softmax
        )

        delta_x, delta_y = delta[:, 0], delta[:, 1]
        N = delta.shape[0]

        # Uniform outlier constants (precomputed once, here)
        omega_x = float((delta_x.max() - delta_x.min()).clamp(min=1e-6).item())
        omega_y = float((delta_y.max() - delta_y.min()).clamp(min=1e-6).item())
        log_ell_out = -(math.log(omega_x) + math.log(omega_y))

        # B-spline grid
        origin, h = self._compute_grid(src_x)
        B = _build_design_matrix(
            x=src_x,
            origin=origin,
            h=h,
            l_x=self.l_x,
            l_y=self.l_y
        )
        B2 = B.pow(2)

        # Regularisation structure
        P = self.l_x * self.l_y
        ops = None,
        kappa_tear = kappa_fold = None
        if self.discontinuity_aware:
            ops = _build_diff_operators(l_x=self.l_x, l_y=self.l_y)
            default_kappa = 0.1 * (omega_x ** 2 + omega_y ** 2)
            kappa_tear = float(self.kappa_tear) if self.kappa_tear is not None else default_kappa
            kappa_fold = float(self.kappa_fold) if self.kappa_fold is not None else default_kappa
            L_shape         = None
            L_shape_diag    = None
            w               = None
        else:
            L_shape         = _build_gmrf_prior(
                l_x=self.l_x,
                l_y=self.l_y,
                shape=self.reg_shape
            ) 
            L_shape_diag    = L_shape.diagonal().clone()

        # Initialize variational parameters
        mu_x    = torch.zeros(P, dtype=torch.float32, device=device)
        mu_y    = torch.zeros(P, dtype=torch.float32, device=device)

        _diag0  = L_shape_diag if L_shape_diag is not None else (_build_gmrf_prior(self.l_x, self.l_y, "bending").diagonal())
        v_x     = (1.0 / (self.alpha_init * _diag0).clamp(min=1e-8)).clone()
        v_y     = (1.0 / (self.alpha_init * _diag0).clamp(min=1e-8)).clone()
        r       = torch.full((N,), self.pi0_init, dtype=torch.float32, device=device)
        pi0     = self.pi0_init
        alpha   = self.alpha_init

        # Use data spread if not supplied by user
        if self.sigma_in_init is not None: 
            sigma2_in = float(self.sigma_in_init)
        else:
            sigma2_in = (omega_x ** 2 + omega_y ** 2) / 8.0

        #  Diagnostics
        elbo_history:       list[float]         = []
        delta_history:      list[torch.Tensor]  = []
        diff_history:       list[torch.Tensor]  = []
        inlier_history:     list[torch.Tensor]  = []
        tear_gate_history:  list[float]         = []
        fold_gate_history:  list[float]         = []

        # Convergence bookkeeping
        elbo_prev           = None
        rel_improve         = 0.0
        no_improve_count    = 0

        # CAVI Loop
        with tqdm(total=self.n_iters, desc="Maximizing ELBO") as pbar:
            for it in range(self.n_iters):
                # [1] Responsibilities (expectation step)
                r = self._update_responsibilities(
                    B=B,
                    B2=B2,
                    mu_x=mu_x,
                    mu_y=mu_y,
                    v_x=v_x,
                    v_y=v_y,
                    delta_x=delta_x,
                    delta_y=delta_y,
                    pi0=pi0,
                    sigma2_in=sigma2_in,
                    log_ell_out=log_ell_out
                )
                inlier_history.append(r)

                # [2] Inlier rate 
                pi0 = float(r.mean().clamp(1e-3, 1.0 - 1e-3))

                if self.discontinuity_aware:
                    w = self._update_edge_weights(
                        mu_x=mu_x,
                        mu_y=mu_y,
                        v_x=v_x,
                        v_y=v_y,
                        ops=ops,
                        kappa_tear=kappa_tear,
                        kappa_fold=kappa_fold
                    )

                    L_shape = self._build_weighted_gmrf(ops, w)
                    L_shape_diag = L_shape.diagonal().clone()

                    if self.allow_tears:
                        tear_cat = torch.cat([w["w1x"], w["w1y"]])
                        tear_gate_history.append(float(tear_cat.mean()) if tear_cat.numel() > 0 else 1.0)
                    if self.allow_folds:
                        fold_cat = torch.cat([w["w2x"], w["w2y"]])
                        tear_gate_history.append(float(fold_cat.mean()) if fold_cat.numel() > 0 else 1.0)

                # [3] Field posterior (both x- and y-components)
                mu_x, v_x = self._update_field(
                    B=B,
                    B2=B2,
                    r=r,
                    delta=delta_x,
                    L0=L_shape,
                    L0_diag=L_shape_diag,
                    alpha=alpha,
                    sigma2_in=sigma2_in
                )

                mu_y, v_y = self._update_field(
                    B=B,
                    B2=B2,
                    r=r,
                    delta=delta_y,
                    L0=L_shape,
                    L0_diag=L_shape_diag,
                    alpha=alpha,
                    sigma2_in=sigma2_in
                )

                # [4] Lean sigma2_in
                sigma2_in = float(self._update_sigma_in(
                    B=B,
                    B2=B2,
                    mu_x=mu_x,
                    mu_y=mu_y,
                    v_x=v_x,
                    v_y=v_y,
                    r=r,
                    delta_x=delta_x,
                    delta_y=delta_y
                ))

                # [5] Learn alpha
                alpha = float(self._update_alpha(
                    mu_x=mu_x,
                    mu_y=mu_y,
                    v_x=v_x,
                    v_y=v_y,
                    L0=L_shape,
                    L0_diag=L_shape_diag,
                    P=P,
                    alpha_max=self.alpha_max
                ))

                # [6] Jacobian barrier
                if self.barrier_alpha > 0.0:
                    barrier_weight = None
                    if self.discontinuity_aware and self.allow_folds and self.barrier_suppression:
                        node_suppression = self._fold_suppression_by_node(ops, w, P)
                        barrier_weight = B @ node_suppression

                    mu_x, mu_y = self._jacobian_barrier_step(
                        x=src_x,
                        origin=origin,
                        h=h,
                        mu_x=mu_x,
                        mu_y=mu_y,
                        barrier_weight=barrier_weight
                    )

                # [7] ELBO
                elbo = self._compute_elbo(
                    B=B,
                    B2=B2,
                    mu_x=mu_x,
                    mu_y=mu_y,
                    v_x=v_x,
                    v_y=v_y,
                    r=r,
                    pi0=pi0,
                    delta_x=delta_x,
                    delta_y=delta_y,
                    L0=L_shape,
                    L0_diag=L_shape_diag,
                    alpha=alpha,
                    sigma2_in=sigma2_in,
                    log_ell_out=log_ell_out
                )
                elbo_val = float(elbo.item())
                elbo_history.append(elbo_val)

                if elbo_prev is not None:
                    rel_improve = abs(elbo_val - elbo_prev) / (abs(elbo_val) + 1e-8)
                    no_improve_count = no_improve_count + 1 if rel_improve < self.tolerance else 0
                elbo_prev = elbo_val

                # Diagnostic deformation snapshots
                delta_src = self._apply_internal_deformation(
                    x=src_x.clone(),
                    origin=origin,
                    h=h,
                    l_x=self.l_x,
                    l_y=self.l_y,
                    mu_x=mu_x,
                    mu_y=mu_y
                )
                delta_history.append(delta_src)

                diff_src = self._compute_displacement_targets(
                    src_x=delta_src,
                    tgt_x=tgt_x,
                    tgt_scores=tgt_scores,
                    use_softmax=True
                )
                diff_history.append(diff_src)

                postfix = {
                    "ELBO":       f"{elbo_val:.4f}",
                    "ΔELBO":      f"{rel_improve:.2e}" if elbo_prev is not None else "NA",
                    "pi0":        f"{pi0:.3f}",
                    "σ_in":       f"{math.sqrt(max(sigma2_in, 0.0)):.3f}",
                    "α":          f"{alpha:.3f}",
                    "stop_count": f"{no_improve_count}/{self.patience}",
                }

                if self.allow_tears:
                    postfix["tear_w"] = f"{tear_gate_history[-1]:.2f}"
                if self.allow_folds:
                    postfix["fold_w"] = f"{fold_gate_history[-1]:.2f}"
                pbar.set_postfix(postfix)
                pbar.update(1)

                if it >= self.min_iters and no_improve_count >= self.patience:
                    pbar.set_description("Converged (ELBO)")
                    break

        elapsed = time.perf_counter()
        sigma_in_final = math.sqrt(max(sigma2_in, 0.0))
        extra_log = ""
        if self.allow_tears:
            extra_log += f" | mean_tear_w={tear_gate_history[-1]:.2f}"
        if self.allow_folds:
            extra_log += f" | mean_fold_w={fold_gate_history[-1]:.2f}"
        print(
            f"[Registration v3] {elapsed:.2f}s | "
            f"ELBO={elbo_history[-1]:.2f} | inlier_rate={pi0:.3f} | "
            f"σ_in={sigma_in_final:.2f} | α={alpha:.3f} | "
            f"n_inliers≈{int(r.sum())}/{N}{extra_log}"
        )

        return _NRRegistrationResult(
            mu_x=mu_x,
            mu_y=mu_y,
            v_x=v_x,
            v_y=v_y,

            r=r,
            
            pi0=pi0,
            sigma_in=sigma_in_final,
            alpha=alpha,
            
            elbo_hist=elbo_history,
            delta_hist=delta_history,
            diff_hist=diff_history,
            inlier_hist=inlier_history,
            elapsed=elapsed,
            
            l_x=self.l_x,
            l_y=self.l_y,
            origin=origin,
            h=h,

            tear_weight_x=w["w1x"] if (self.discontinuity_aware and self.allow_tears) else None,
            tear_weight_y=w["w1y"] if (self.discontinuity_aware and self.allow_tears) else None,
            fold_weight_x=w["w2x"] if (self.discontinuity_aware and self.allow_folds) else None,
            fold_weight_y=w["w2y"] if (self.discontinuity_aware and self.allow_folds) else None,
            tear_gate_hist=tear_gate_history,
            fold_gate_hist=fold_gate_history,
        )


    @staticmethod
    def _compute_displacement_targets(
        src_x: torch.Tensor,
        tgt_x: torch.Tensor,
        tgt_scores: torch.Tensor,
        use_softmax: bool = True
    ) -> torch.Tensor:
        w = F.softmax(tgt_scores, dim=1) if use_softmax else tgt_scores
        y_hat = (w.unsqueeze(2) * tgt_x).sum(dim=1)
        return y_hat - src_x


    def _compute_grid(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        xy_min = x.min(dim=0).values
        xy_max = x.max(dim=0).values

        extent_x = (xy_max[0] - xy_min[0]).item()
        extent_y = (xy_max[1] - xy_min[1]).item()
        hx = extent_x / max(self.l_x - 3, 1)
        hy = extent_y / max(self.l_y - 3, 1)
        h = float(max(hx, hy, 1e-6))

        origin = xy_min - 1.5 * h
        return origin, h


    # CAVI Update 1 - Responsibilities
    def _update_responsibilities(
        self,
        B:              torch.Tensor,
        B2:             torch.Tensor,
        mu_x:           torch.Tensor,
        mu_y:           torch.Tensor,
        v_x:            torch.Tensor,
        v_y:            torch.Tensor,
        delta_x:        torch.Tensor,
        delta_y:        torch.Tensor,
        pi0:            float,
        sigma2_in:      float,
        log_ell_out:    float
    ) -> torch.Tensor:
        """
        Computed in log-space using logaddexp for numerical stability
        """
        pred_x = B @ mu_x
        pred_y = B @ mu_y

        pvar_x = sigma2_in + B2 @ v_x
        pvar_y = sigma2_in + B2 @ v_y

        log_ell_in = (
            _log_normal(delta_x, pred_x, pvar_x)
            + _log_normal(delta_y, pred_y, pvar_y)
        )

        log_pi0     = math.log(max(pi0,       1e-12))
        log1mpi0    = math.log(max(1.0 - pi0, 1e-12))

        log_num     = log_pi0 + log_ell_in
        log_denom   = torch.logaddexp(log_num, torch.full_like(log_num, log1mpi0 + log_ell_out))

        return (log_num - log_denom).exp().clamp(1e-6, 1.0 - 1e-6)


    # CAVI Update 3 - Field posterior
    @staticmethod
    def _update_field(
        B:              torch.Tensor,
        B2:             torch.Tensor,
        r:              torch.Tensor,
        delta:          torch.Tensor,
        L0:             torch.Tensor,
        L0_diag:        torch.Tensor,
        alpha:          float,
        sigma2_in:      float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rB      = r.unsqueeze(1) * B
        BtRB    = rB.T @ B
        H       = BtRB / sigma2_in + alpha * L0

        rhs     = (rB * delta.unsqueeze(1)).sum(dim=0) / sigma2_in
        mu      = torch.linalg.solve(H, rhs)

        data_prec   = (r.unsqueeze(1) * B2).sum(dim=0) / sigma2_in
        v           = 1.0 / (alpha * L0_diag + data_prec).clamp(1e-12)

        return mu, v


    # CAVI Update 4 - sigma-in
    def _update_sigma_in(
        B:              torch.Tensor,
        B2:             torch.Tensor,
        mu_x:           torch.Tensor,
        mu_y:           torch.Tensor,
        v_x:            torch.Tensor,
        v_y:            torch.Tensor,
        r:              torch.Tensor,
        delta_x:        torch.Tensor,
        delta_y:        torch.Tensor
    ) -> torch.Tensor:
        pred_x  = B @ mu_x
        pred_y  = B @ mu_y
        pvar_x  = B2 @ v_x
        pvar_y  = B2 @ v_y

        sq_res_x = (delta_x - pred_x).pow(2) + pvar_x
        sq_res_y = (delta_y - pred_y).pow(2) + pvar_y

        numerator   = (r * (sq_res_x + sq_res_y)).sum()
        denominator = 2.0 * r.sum().clamp(min=1e-8)

        return (numerator / denominator).clamp(min=1e-8)


    # CAVI Update 5 - alpha
    @staticmethod
    def _update_alpha(
        mu_x:           torch.Tensor,
        mu_y:           torch.Tensor,
        v_x:            torch.Tensor,
        v_y:            torch.Tensor,
        L0:             torch.Tensor,
        L0_diag:        torch.Tensor,
        P:              int,
        alpha_max:      float
    ) -> torch.Tensor:
        device = _get_device()
        quad_x  = mu_x @ (L0 @ mu_x)
        quad_y  = mu_y @ (L0 @ mu_y)
        trace_x = (L0_diag * v_x).sum()
        trace_y = (L0_diag * v_y).sum()

        denom = (quad_x + quad_y + trace_x + trace_y).clamp(min=1e-12)
        alpha = float(P) / denom

        return alpha.clamp(max=alpha_max) if isinstance(alpha, torch.Tensor) else torch.Tensor(alpha, dtype=torch.float32, device=device).clamp(max=alpha_max)


    def _update_edge_weights(
        self,
        mu_x:           torch.Tensor,
        mu_y:           torch.Tensor,
        v_x:            torch.Tensor,
        v_y:            torch.Tensor,
        ops:            dict,
        kappa_tear:     float,
        kappa_fold:     float
    ) -> dict:
        device = _get_device()
        
        def _edge_weight(D: torch.Tensor, kappa: float) -> torch.Tensor:
            if D.shape[0] == 0:
                return torch.zeros(0, dtype=torch.float32, device=device)
            t_x = D @ mu_x
            t_y = D @ mu_y
            var_x = D.pow(2) @ v_x
            var_y = D.pow(2) @ v_y

            s = t_x.pow(2) + t_y.pow(2) + var_x + var_y
            return _geman_mcclure_weight(residual_sq=s, kappa=kappa)

        w: dict = {}

        if self.allow_tears:
            w["w1x"] = _edge_weight(ops["l1x"], kappa_tear)
            w["w1y"] = _edge_weight(ops["l1y"], kappa_tear)
        else:
            w["w1x"] = torch.zeros(ops["l1x"].shape[0], dtype=torch.float32, device=device)
            w["w1y"] = torch.zeros(ops["l1y"].shape[0], dtype=torch.floar32, device=device)

        if self.allow_folds:
            w["w2x"] = _edge_weight(ops["l2x"], kappa_fold)
            w["w2y"] = _edge_weight(ops["l2y"], kappa_fold)
        else:
            w["w2x"] = torch.ones(ops["l2x"].shape[0], dtype=torch.float32, device=device)
            w["w2y"] = torch.ones(ops["l2y"].shape[0], dtype=torch.float32, device=device)
        
        return w


    @staticmethod
    def _build_weighted_gmrf(
        ops:            dict,
        w:              dict
    ) -> torch.Tensor:
        device = _get_device()
        P = ops["l1x"].shape[1] if ops["l1x"].shape[1] > 0 else ops["l2x"].shape[1]
        L = torch.zeros(P, P, dtype=torch.float32, device=device)

        for key_D, key_w in (("l1x", "w1x"), ("l1y", "w1y"), ("l2x", "w2x"), ("l2y", "w2y")):
            D = ops[key_D]
            if D.shape[0] == 0:
                continue
        
            wv = w[key_w]
            L += D.T @ (D * wv.unsqueeze(1))
        
        L += 1e-5 * torch.eye(P, dtype=torch.float32, device=device)
        return L


    @staticmethod
    def _fold_suppression_by_node(
        ops:            dict,
        w:              dict,
        P:              int
    ) -> torch.Tensor:
        device = _get_device()
        suppression = torch.ones(P, dtype=torch.float32, device=device)

        for key_D, key_w in (("l2x", "w2x"), ("l2y", "w2y")):
            D = ops[key_D]
            if D.shape[0] == 0:
                continue

            wv = w[key_w]
            rows, cols = torch.nonzero(D, as_tuple=True)
            vals = wv[rows]
            suppression.scatter_reduce_(0, cols, vals, reduce="amin", include_self=True)
        
        return suppression


    # CAVI Update 6 - Jacobian barrier
    def _jacobian_barrier_step(
        self,
        x:              torch.Tensor,
        origin:         torch.Tensor,
        h:              float,
        mu_x:           torch.Tensor,
        mu_y:           torch.Tensor,
        barrier_weight: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        l_x, l_y = self.l_x, self.l_y

        dBdx = _build_design_matrix_derivative(
            x=x,
            origin=origin,
            h=h,
            l_x=l_x,
            l_y=l_y,
            axis=0
        )
        dBdy = _build_design_matrix_derivative(
            x=x,
            origin=origin,
            h=h,
            l_x=l_x,
            l_y=l_y,
            axis=1
        )

        gamma = self.barrier_alpha if barrier_weight is None else self.barrier_alpha * barrier_weight

        for _ in range(self.barrier_steps):
            mux = mu_x.detach().requires_grad_(True)
            muy = mu_y.detach().requires_grad_(True)

            # Construct Jacobian
            J00 = 1.0 + dBdx @ mux
            J01 =       dBdy @ mux
            J10 =       dBdx @ muy
            J11 = 1.0 + dBdy @ muy

            det_J = J00 * J11 - J01 * J10

            barrier_loss = -(gamma * det_J.clamp(min=1e-4).log()).sum()
            barrier_loss.backward()

            with torch.no_grad():
                mu_x = (mu_x - self.barrier_lr * mux.grad()).detach()
                mu_y = (mu_y - self.barrier_lr * muy.grad()).detach()

        return mu_x, mu_y


    @staticmethod
    def _compute_elbo(
        B:              torch.Tensor,
        B2:             torch.Tensor,
        mu_x:           torch.Tensor,
        mu_y:           torch.Tensor,
        v_x:            torch.Tensor,
        v_y:            torch.Tensor,
        r:              torch.Tensor,
        pi0:            float,
        delta_x:        torch.Tensor,
        delta_y:        torch.Tensor,
        L0:             torch.Tensor,
        L0_diag:        torch.Tensor,
        alpha:          float,
        sigma2_in:      float,
        log_ell_out:    float     
    ) -> torch.Tensor:
        log2pi  = math.log(2.0 * math.pi)
        eps     = 1e-10

        pred_x  = B @ mu_x
        pred_y  = B @ mu_y

        pvar_x  = B2 @ v_x
        pvar_y  = B2 @ v_y

        bias2_x     = (delta_x - pred_x).pow(2)
        bias2_y     = (delta_y - pred_y).pow(2)
        inlier_ll   = r * (
            -float(log2pi + math.log(sigma2_in))
            - 0.5 * (bias2_x + pvar_x + bias2_y + pvar_y) / sigma2_in
        )
        outlier_ll = (1.0 - r) * log_ell_out
        term_A = (inlier_ll + outlier_ll).sum()

        def _neg_kl_field(
            mu: torch.Tensor,
            v: torch.Tensor
        ) -> torch.Tensor:
            return 0.5 * (
                v.clamp(eps).log().sum()
                - alpha * (L0_diag * v).sum()
                - alpha * (mu @ (L0 @ mu))
            )

        term_B = _neg_kl_field(mu_x, v_x) + _neg_kl_field(mu_y, v_y)

        term_C = -(
            r * (r.clamp(eps).log() - math.log(pi0 + eps))
            + (1.0 - r) * ((1.0 - r).clamp(eps).log() - math.log(1.0 - pi0 + eps))
        ).sum()

        return term_A + term_B + term_C