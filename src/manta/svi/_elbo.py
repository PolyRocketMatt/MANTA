import math
import torch

from ..utils._gpu import _chunked_range


@torch.no_grad()
def _compute_elbo(
    stencil_idx: torch.Tensor,
    stencil_w: torch.Tensor,
    mu: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    delta: torch.Tensor,
    L0: torch.Tensor,
    L0_diag: torch.Tensor,

    pi0: float,
    alpha: float, 
    sigma2_in: float,

    log_ell_out=torch.Tensor,

    batch_size: int = 8192
) -> torch.Tensor:
    """
    Compute ELBO.
    """
    N, D = delta.shape

    log2pi = math.log(2.0 * math.pi)

    term_A = 0.0
    for s, e in _chunked_range(N, batch_size):
        idx = stencil_idx[s:e]
        w = stencil_w[s:e]

        pred = (mu[:, idx].permute(1, 2, 0) * w.unsqueeze(-1)).sum(dim=1)
        pv = (w.pow(2) * v[idx]).sum(dim=1)
        sq = (delta[s:e] - pred).pow(2)
        inlier_ll = r[s:e].unsqueeze(1) * (
            -log2pi - 0.5 * math.log(sigma2_in)
            - 0.5 * (sq + pv.unsqueeze(1).expand_as(sq)) / sigma2_in
        )
        outlier_ll = (1.0 - r[s:e]).unsqueeze(1) * log_ell_out
        term_A = term_A + (inlier_ll + outlier_ll).sum()

    eps = 1e-10
    neg_kl = 0.0
    for d in range(D):
        mu_d = mu[d]
        quad = mu_d @ (L0 @ mu_d)
        trace = (L0_diag * v).sum()
        neg_kl = neg_kl + 0.5 * (
            v.clamp(eps).log().sum() - alpha * trace - alpha * quad
        ) 
    term_B = neg_kl

    r_c = r.clamp(eps)
    term_C = -(r * (r_c.log() - math.log(pi0 + eps))
               + (1 - r) * ((1 - r).clamp(eps).log() - math.log(1 - pi0 + eps))).sum()

    return term_A + term_B + term_C