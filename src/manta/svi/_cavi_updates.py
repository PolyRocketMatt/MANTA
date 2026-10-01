import math
import numpy as np
import torch

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Literal, Tuple

from ..primitives._bspline import _eval_stencil_derivative
from ..primitives._diff_ops import _apply_diff_axis, _apply_diff_axis_sq_variance
from ..primitives._robust import _geman_mcclure_weight
from ..utils._gpu import _chunked_range


def _solve_dense(
    H: torch.Tensor, 
    rhs: torch.Tensor
) -> torch.Tensor:
    return torch.linalg.solve(H, rhs)


def _solve_sparse(
    H_dense: torch.Tensor,
    rhs_dense: torch.Tensor
) -> torch.Tensor:
    device = H_dense.device
    H_np = H_dense.detach().cpu().numpy()
    rhs_np = rhs_dense.detach().cpu().numpy()

    # Threshold small entries to sparsify
    H_np[np.abs(H_np) < 1e-8] = 0.0
    H_sp = sp.csc_matrix(H_np)
    lu = spla.splu(H_sp)
    x_np = lu.solve(rhs_np)
    return torch.from_numpy(x_np).to(dtype=H_dense.type, device=device)


def _update_field(
    stencil_idx: torch.Tensor,
    stencil_w: torch.Tensor, 
    r: torch.Tensor,
    delta: torch.Tensor,
    L0: torch.Tensor,
    L0_diag: torch.Tensor,
    alpha: float,
    sigma2_in: float, 
    P: int,
    H: torch.Tensor | None = None,
    mode: Literal["sparse", "dense", "auto"] = "auto",
    sparse_threshold: int = 8192,
    batch_size: int = 8192,
    return_H: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update the field posterior q(µ) = N(µ | m, diag(v))

    Returns:
        mu:     [D, P]  posterior mean
        v:      [P]     posterior diagonal variance
    """
    device = delta.device
    N, S = stencil_w.shape
    D = delta.shape[1]

    if mode == "auto":
        mode = "dense" if P <= sparse_threshold else "sparse"

    if H is None:
        H = torch.zeros(P, P, dtype=torch.float32, device=device)
        for s, e in _chunked_range(N, batch_size):
            idx = stencil_idx[s:e]
            w = stencil_w[s:e]
            rr = r[s:e]

            outer = w.unsqueeze(2) * w.unsqueeze(1)
            outer = outer * rr.view(-1, 1, 1)

            rows = idx.unsqueeze(2).expand(-1, S, S).reshape(-1)
            cols = idx.unsqueeze(1).expand(-1, S, S).reshape(-1)
            vals = outer.reshape(-1)

            H.index_put_((rows, cols), vals, accumulate=True)
        H = H / sigma2_in + alpha * L0

    rhs = torch.zeros(P, D, device=device)
    for s, e in _chunked_range(N, batch_size):
        idx = stencil_idx[s:e]
        w = stencil_w[s:e]
        rr = r[s:e]

        for d in range(D):
            weighted = rr * delta[s:e, d]
            contrib = (w * weighted.unsqueeze(1)).reshape(-1)
            rhs[:, d].index_add_(0, idx.reshape(-1), contrib)
    rhs = rhs / sigma2_in

    if mode == "sparse":
        mu = _solve_sparse(H, rhs)
    else:
        mu = _solve_dense(H, rhs)

    # Mean-field varance
    data_prec = torch.zeros(P, device=device)
    for s, e in _chunked_range(N, batch_size):
        idx = stencil_idx[s:e]
        w2 = stencil_w[s:e].pow(2)
        rr = r[s:e]
        contrib = (w2 * rr.unsqueeze(1)).reshape(-1)
        data_prec.index_add_(0, idx.reshape(-1), contrib)
    data_prec = data_prec / sigma2_in
    v = 1.0 / (alpha * L0_diag + data_prec).clamp(min=1e-12)

    if return_H:
        return mu.T, v, H
    return mu.T, v


@torch.no_grad()
def _update_responsibilities(
    stencil_idx: torch.Tensor,
    stencil_w: torch.Tensor,
    mu: torch.Tensor,
    v: torch.Tensor, 
    delta: torch.Tensor,
    pi0: float,
    sigma2_in: float,
    log_ell_out: torch.Tensor,
    batch_size: int = 8192
) -> torch.Tensor:
    """
    Update per-point inlier responsibilities
    """
    device = delta.device
    N, D = delta.shape

    pred = torch.zeros(N, D, device=device)
    pvar = torch.zeros(N, D, device=device)

    for s, e in _chunked_range(N, batch_size):
        idx = stencil_idx[s:e]
        w = stencil_w[s:e]

        pred_chunk = (mu[:, idx].permute(1, 2, 0) * w.unsqueeze(-1)).sum(dim=1)
        pvar_chunk = (w.pow(2).unsqueeze(-1) * v[idx]).sum(dim=1)

        pred[s:e] = pred_chunk
        pvar[s:e] = pvar_chunk

    pvar = pvar + sigma2_in
    log_ell_in = (
        -0.5 * (torch.log(2.0 * math.pi * pvar) +
                (delta - pred).pow(2) / pvar)
    ).sum(dim=-1)

    log_pi0 = math.log(max(pi0, 1e-12))
    log_pi1 = math.log(max(1.0 - pi0, 1e-12))

    log_num = log_pi0 + log_ell_in
    log_denom = torch.logaddexp(log_num, log_ell_in.new_full(), log_pi1 + log_ell_out)

    r = (log_num - log_denom).exp().clamp(1e-6, 1.0 - 1e-6)
    return r


@torch.no_grad()
def _update_sigma_in(
    stencil_idx: torch.Tensor,
    stencil_w: torch.Tensor,
    mu: torch.Tensor,
    v: torch.Tensor, 
    r: torch.Tensor,
    delta: torch.Tensor,
    batch_size: int = 8192
) -> torch.Tensor:
    """
    Update inlier variance
    """
    device = delta.device
    N, D = delta.shape
    S = stencil_w.shape[1]

    sq_res = torch.zeros(N, D, device=device)
    pvar = torch.zeros(N, D, device=device)

    for s, e in _chunked_range(N, batch_size):
        idx = stencil_idx[s:e]
        w = stencil_w[s:e]
        pred = (mu[:, idx].permute(1, 2, 0) * w.unsqueeze(-1)).sum(dim=1)

        sq_res[s:e] = (delta[s:e] - pred).pow(2)
        pvar[s:e] = (w.pow(2) * v[idx]).sum(dim=1)

    num = (r.unsqueeze(1) * (sq_res + pvar)).sum()
    denom = D * r.sum().clamp(min=1e-8)

    return (num / denom).clamp(min=1e-8)


@torch.no_grad()
def _update_alpha(
    mu: torch.Tensor,
    v: torch.Tensor,
    L0: torch.Tensor,
    L0_diag: torch.Tensor,
    alpha_max=1e6
) -> torch.Tensor:
    """
    Update alpha
    """
    N, P = mu.shape
    quad = (mu * (L0 @ mu)).sum()
    trace = (L0_diag * v).sum() * N
    denom = (quad + trace).clamp(min=1e-12)

    return float(min(P / denom, alpha_max))


def _jacobian_barrier_step(
    x: torch.Tensor,
    origin: torch.Tensor,
    h: float,
    grid_shape: Tuple[int,...],
    mu: torch.Tensor,
    lr: float = 0.5,
    steps: int = 3,
    alpha: float = 1.0,
    beta: float = 0.0,
    barrier_weight: torch.Tensor | None = None,
    density_weight: float | None = None,
    use_adam: bool = True
) -> torch.Tensor:
    device = x.device
    N, D = x.shape

    der_stencils = [
        _eval_stencil_derivative(
            x=x,
            origin=origin,
            h=h,
            grid_shape=grid_shape,
            axis=d
        )
        for d in range(D)
    ]

    eye = torch.eye(D, device=device).unsqueeze(0)
    mu_det = mu.detach().clone()

    if use_adam:
        opt = torch.optim.Adam([mu_det], lr=lr)
    else:
        opt = None

    for _ in range(steps):
        mu_req = mu_det.clone().requires_grad_(True)

        # Build J via stack (never in-pace into a non-grad tensor)
        J_rows = []
        for i in range(D):
            J_cols = []
            for j in range(D):
                idx_j, w_j = der_stencils[j]
                val = (w_j * mu_req[i, idx_j]).sum(dim=1)
                J_cols.append(val)
            J_rows.append(torch.stack(J_cols, dim=-1))
        J = torch.stack(J_rows, dim=1) + eye

        if D == 2:
            det_J = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        else:
            det_J = torch.linalg.det(J)

        det_clamped = det_J.clamp(min=1e-4)
        log_det = det_clamped.log()


        w = barrier_weight if barrier_weight is not None else 1.0
        
        # One-sided fold penalty
        loss = -(alpha * w * det_clamped.log()).sum()

        if beta > 0.0:
            if density_weight is not None:
                loss = loss + beta * (w * density_weight * log_det.pow(2)).mean()
            else:
                loss = loss + beta * (w * log_det.pow(2)).mean()

        if use_adam:
            opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                opt.step()
            mu_det = mu_det.detach().requires_grad_(False)
        else:
            grad = torch.autograd.grad(loss, mu_req)[0]
            with torch.no_grad():
                mu_det = mu_det - lr * grad

    return mu_det


@torch.no_grad()
def _update_edge_weights(
    mu: torch.Tensor,
    v: torch.Tensor,
    grid_shape: Tuple[int,...],
    kappa_tear: float,
    kappa_fold: float,
    allow_tears: bool = True,
    allow_folds: bool = True
) -> dict:
    """
    Compute per-edge weights using Geman-McClure on the expected
    squared displacement difference for membrane (tear) and 
    bending (fold) edges.
    """
    device = mu.device
    D = mu.shape[0]

    w_dict = {}

    def _edge_weight(
        axis: int,
        order: int, 
        kappa: float
    ) -> float:
        diffs = []
        var_sum = 0.0

        for d in range(D):
            t = _apply_diff_axis(
                mu=mu[d],
                grid_shape=grid_shape,
                axis=axis,
                order=order
            )
            var_d = _apply_diff_axis_sq_variance(
                v=v,
                grid_shape=grid_shape,
                axis=axis,
                order=order
            )

            diffs.append(t.pow(2))
            var_sum = var_sum + var_d

        s = sum(diffs) + var_sum
        return _geman_mcclure_weight(s, kappa)

    if allow_tears:
        for axis in range(D):
            w_dict[f"w1{axis}"] = _edge_weight(axis=axis, order=1, kappa=kappa_tear)
    else:
        for axis in range(D):
            w_dict[f"w1{axis}"] = torch.ones(0, device=device)

    if allow_folds:
        for axis in range(D):
            w_dict[f"w2{axis}"] = _edge_weight(axis=axis, order=2, kappa=kappa_fold)
    else:
        for axis in range(D):
            w_dict[f"w2{axis}"] = torch.ones(0, device=device)

    return w_dict