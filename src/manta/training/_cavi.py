import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Literal, List, Optional, Sequence, Tuple

from ..matching._ot import _compute_displacement_targets, _sparse_unbalanced_sinkhorn
from ..models._ffd import MultiscaleFFD
from ..models._encoder import ExpressionEncoder, MantaEncoderLoss, _ssl_train_step
from ..models._field import CanonicalField
from ..models._gat import HeterogeneousGAT
from ..primitives._bspline import _eval_stencil
from ..primitives._diff_ops import _build_gmrf
from ..primitives._weighted_gmrf import _build_weighted_gmrf
from ..svi._elbo import _compute_elbo
from ..svi._cavi_updates import (
    _jacobian_barrier_step,
    _update_alpha,
    _update_edge_weights,
    _update_field,
    _update_responsibilities,
    _update_sigma_in
)
from ..utils._spatial import _binned_knn
from ..utils._tensor_utils import (
    _get_device,
    _as_tensor,
)


@dataclass
class SliceData:
    x: torch.Tensor         # [N, D]
    expr: torch.Tensor      # [N, G]


class MantaModelTrainer:
    """
    Multiscale CAVI + JEPA trainer.
    """
    def __init__(
        self,

        D: int,
        G: int,
        n_slices: int,
        grid_shapes: List[Tuple[int,...]],

        latent_dim: int = 64,
        lr_nn: float = 1e-3,

        # CAVI
        pi0_init: float = 0.8,
        sigma2_in_init: Optional[float] = None,
        alpha_init: float = 1.0,
        alpha_max: float = 1e6,
        discontinuity_aware: bool = True,
        kappa_tear: Optional[float] = None,
        kappa_fold: Optional[float] = None,
        allow_tears: bool = True,
        allow_folds: bool = True,
        sparse_threshold: int = 8192,

        # Barrier + density
        barrier_alpha: float = 1.0,
        barrier_lr: float = 0.1,
        barrier_steps: int = 3,
        density_beta: float = 0.1,
        density_k: int = 8,

        # Convergence 
        tolerance: float = 1e-4,
        patience: int = 5,
        min_iters: int = 5,
        n_iters: int = 10,

        # OT
        ot_n_candidates: int = 32,
        ot_n_iters: int = 50,
        ot_epsilon: float = 0.05,
        ot_alpha: float = 1.0,
        ot_beta: float = 1.0,

        # JEPA
        jepa_weight: float = 1.0,
        jepa_steps: int = 5,
        vicreg_weight: float = 0.1,
        vicreg_var_reg: float = 1.0,
        vicreg_cov_reg: float = 1.0,

        # Freezing
        freeze_threshold: float = 0.95,

        # Regularization
        reg_shape: Literal["bending", "membrane", "combined"] = "bending",

        # SSL
        encoder_lr: float = 1e-3,
        ssl_epochs: int = 10,
        ssl_steps_per_epoch: int = 15,
        ssl_batch_size: int = 2048,
        n_rounds: int = 3,
        finetune_lr: float = 1e-4,
        finetune_weight: float = 0.1,
        finetune_steps: int = 50
    ) -> None:
        self.device = _get_device()

        self.D = D
        self.G = G
        self.K = n_slices
        self.grid_shapes = grid_shapes
        self.S = len(self.grid_shapes)

        # Hyperparameters
        self.pi0_init = pi0_init
        self.sigma2_in_init = sigma2_in_init
        self.alpha_init = alpha_init
        self.alpha_max = alpha_max
        self.discontinuity_aware = discontinuity_aware
        self.kappa_tear = kappa_tear
        self.kappa_fold = kappa_fold
        self.allow_tears = allow_tears
        self.allow_folds = allow_folds
        self.sparse_threshold = sparse_threshold

        self.barrier_alpha = barrier_alpha
        self.barrier_lr = barrier_lr
        self.barrier_steps = barrier_steps
        self.density_beta = density_beta
        self.density_k = density_k

        self.tolerance = tolerance
        self.patience = patience
        self.min_iters = min_iters
        self.n_iters = n_iters

        self.ot_n_candidates = ot_n_candidates
        self.ot_n_iters = ot_n_iters
        self.ot_epsilon = ot_epsilon
        self.ot_alpha = ot_alpha
        self.ot_beta = ot_beta

        self.jepa_weight = jepa_weight
        self.jepa_steps = jepa_steps
        self.vicreg_weight = vicreg_weight
        self.vicreg_var_reg = vicreg_var_reg
        self.vicreg_cov_reg = vicreg_cov_reg

        self.freeze_threshold = freeze_threshold
        self.reg_shape = reg_shape

        # Infer rho_src/tgt
        self.log_rho_src = nn.Parameter(torch.zeros(n_slices, device=self.device))
        self.log_rho_tgt = nn.Parameter(torch.zeros(n_slices, device=self.device))
        self.rho_lr = 0.05
        self.rho_target_inlier = 0.85

        # SSL
        self.encoder_lr = encoder_lr
        self.ssl_epochs = ssl_epochs
        self.ssl_steps_per_epoch = ssl_steps_per_epoch
        self.ssl_batch_size = ssl_batch_size
        self.n_rounds = n_rounds
        self.finetune_lr = finetune_lr
        self.finetune_weight = finetune_weight
        self.finetune_steps = finetune_steps

        # Modules
        self.field = CanonicalField(
            D=D, 
            d=latent_dim
        ).to(self.device)
        self.encoder = ExpressionEncoder(
            in_dim=G,
            latent_dim=latent_dim,
            hidden_dim=latent_dim * 4,
            n_layers=3,
            decoder_hidden_dim=latent_dim * 2,
            p_drop=0.1,
            eta=0.01
        ).to(self.device)
        self.deformation = MultiscaleFFD(
            D=D, 
            n_slices=n_slices, 
            grid_shapes=self.grid_shapes
        ).to(self.device)
        self.gat = HeterogeneousGAT(
            in_dim=latent_dim,
            hidden=latent_dim,
            out_dim=latent_dim,
            n_spot_layers=2,
            heads=4,
            k=8
        ).to(self.device)

        self.encoder_loss = MantaEncoderLoss(
            sim_coeff=25.0,
            var_coeff=25.0,
            cov_coeff=1.0,
            lambda_recon=1.0
        )

        # Registration optimizer = field + GAT only
        self.opt_reg = torch.optim.Adam(
            list(self.field.parameters()) + list(self.gat.parameters()),
            lr=lr_nn
        )

        # Encoder optimizer
        self.opt_encoder = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.encoder_lr,
            weight_decay=1e-4
        )
        self.scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.opt_encoder,
            T_max=self.ssl_epochs,
            eta_min=0.0
        )

        # Diagnostics
        self.elbo_history: List[float] = []
        self.ssl_history: List[float] = []
        self.per_slice_final: List[dict] = []

    def _apply_upto(
        self,
        x: torch.Tensor,
        slice_id: int,
        up_to_scale: int,
        origin: torch.Tensor,
        h: float
    ) -> torch.Tensor:
        for s in range(up_to_scale + 1):
            x = self.deformation.apply_scale(
                x=x,
                s=s,
                slice_id=slice_id,
                origin=origin,
                h=h
            )
        return x

    def _gat_embeddings(
        self,
        slices: List[SliceData],
        origin: torch.Tensor,
        h: float,
        scale: int
    ) -> List[torch.Tensor]:
        with torch.no_grad():
            enc_feats = [self.encoder.infer(s.expr) for s in slices]
        canon = [
            self.deformation.apply_full(
                x=s.x,
                slice_id=k,
                origin=origin,
                h=h
            )
            for k, s in enumerate(slices)
        ]
        slice_ids = list(range(len(slices)))
        return self.gat(enc_feats, canon, slice_ids, scale)

    @torch.no_grad()
    def _compute_targets(
        self,
        slices: Sequence[SliceData],
        inputs: Sequence[torch.Tensor],
        origin: torch.Tensor,
        h: float,
        scale: int,
        rho_src: torch.Tensor,
        rho_tgt: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        For each slice k, match its input to the pooled inputs of 
        all OTHER slices. Return barycentric-projected displacements.
        """
        K = len(slices)
        z_list = self._gat_embeddings(
            slices=slices,
            origin=origin,
            h=h,
            scale=scale
        )

        targets = []
        for k in range(K):
            others_idx = [j for j in range(K) if j != k]
            other_inputs = torch.cat([inputs[j] for j in others_idx], dim=0)
            other_z = torch.cat([z_list[j] for j in others_idx], dim=0)

            src_idx, tgt_idx, log_pi = _sparse_unbalanced_sinkhorn(
                src_x=inputs[k],
                tgt_x=other_inputs,
                src_z=z_list[k],
                tgt_z=other_z,
                alpha=self.ot_alpha,
                beta=self.ot_beta,
                n_candidates=self.ot_n_candidates,
                epsilon=self.ot_epsilon,
                src_rho=float(rho_src[k]),
                tgt_rho=float(rho_tgt[k]),
                n_iters=self.ot_n_iters
            )

            delta = _compute_displacement_targets(
                src_x=inputs[k],
                tgt_x=other_inputs,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                log_pi=log_pi,
                normalize=True
            )

            targets.append(delta)
        return targets

    def _ssl_pretrain_encoder(
        self,
        slices: List[SliceData],
        verbose: bool = False
    ) -> List[float]:
        """
        Train the encoder on the union of all expression matrices.
        """
        x_all = torch.cat([s.expr for s in slices], dim=0)
        N = x_all.shape[0]
        history = []

        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad_(True)

        for epoch in range(self.ssl_epochs):
            epoch_loss = 0.0

            for _ in range(self.ssl_steps_per_epoch):
                batch_idx = torch.randperm(N, device=self.device)[:self.ssl_batch_size]
                batch_x = x_all[batch_idx]

                losses = _ssl_train_step(
                    model=self.encoder,
                    loss_fn=self.encoder_loss,
                    x=batch_x,
                    optimizer=self.opt_encoder,
                    grad_clip=1.0
                )
                epoch_loss += losses["loss"]
            self.scheduler_encoder.step()
            epoch_loss /= self.ssl_steps_per_epoch
            history.append(epoch_loss)

            if verbose:
                print(f"[SSL epoch {epoch}] loss = {epoch_loss:.4f}")

        self.encoder.eval()
        self.ssl_history.extend(history)
        return history


    def _cavi_slice(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        stencil_idx: torch.Tensor,
        stencil_w: torch.Tensor,
        L0: torch.Tensor,
        L0_diag: torch.Tensor,
        frozen: torch.Tensor,
        origin: torch.Tensor,
        h: float,
        scale: float
    ) -> Tuple[torch.Tensor,...]:
        device = x.device
        N, D = x.shape
        P = 1

        for L in self.grid_shapes[scale]:
            P *= L

        # Initialize
        mu      = torch.zeros(D, P, device=device)
        v       = torch.full((P,), 0.01, device=device)
        r       = torch.full((N,), self.pi0_init, device=device)
        pi0     = self.pi0_init
        alpha   = self.alpha_init 

        if self.sigma2_in_init is not None:
            sigma2_in = float(self.sigma2_in_init)
        else:
            spread = (delta.max(0).values - delta.min(0).values).clamp_min(1e-6)
            sigma2_in = float((spread ** 2).mean().item() / 8.0)

        kappa_tear = self.kappa_tear
        kappa_fold = self.kappa_fold
        if kappa_tear is None:
            kappa_tear = 0.1 * delta.var(dim=0).sum()
        if kappa_fold is None:
            kappa_fold = 0.1 * delta.var(dim=0).sum()

        omega = (delta.max(0).values - delta.min(0).values).clamp_min(1e-6)
        log_ell_out = float(-torch.log(omega).sum().item())

        # Frozen mask -> 0 weight
        active = (~frozen).float()
        active_sum = active.sum().clamp_min(1.0)

        # Solver mode
        mode = "sparse" if P > self.sparse_threshold else "dense"

        # Local density estimate
        rho_local = None
        if self.density_beta > 0.0:
            with torch.no_grad():
                _, d2 = _binned_knn(
                    x=x,
                    k=self.density_k
                )
                median_d2 = d2.median(dim=1).values.clamp_min(1e-6)
                rho_local = 1.0 / (median_d2 ** (self.D / 2.0))
                rho_local = rho_local / rho_local.mean().clamp_min(1e-12)

        elbo_hist = []
        for it in range(self.n_iters):
            # Expectation
            r = _update_responsibilities(
                stencil_idx=stencil_idx,
                stencil_w=stencil_w,
                mu=mu,
                v=v,
                delta=delta,
                pi0=pi0,
                sigma2_in=sigma2_in,
                log_ell_out=log_ell_out
            )

            r = r * active
            pi0 = float(r.sum().item() / active_sum.item()) 
            pi0 = max(min(pi0, 1.0 - 1e-3), 1e-3)

            # Maximization
            mu, v = _update_field(
                stencil_idx=stencil_idx,
                stencil_w=stencil_w,
                r=r,
                delta=delta,
                L0=L0,
                L0_diag=L0_diag,
                alpha=alpha,
                sigma2_in=sigma2_in,
                P=P,
                mode=mode,
                sparse_threshold=self.sparse_threshold
            )

            # Update hyperparams
            sigma2_in = float(
                _update_sigma_in(
                    stencil_idx=stencil_idx,
                    stencil_w=stencil_w,
                    mu=mu,
                    v=v,
                    r=r,
                    delta=delta
                )
            )
            alpha = float(
                _update_alpha(
                    mu=mu,
                    v=v,
                    L0=L0,
                    L0_diag=L0_diag,
                    alpha_max=self.alpha_max
                )
            )

            if self.discontinuity_aware:
                w_dict = _update_edge_weights(
                    mu=mu,
                    v=v,
                    grid_shape=self.grid_shapes[scale],
                    kappa_tear=self.kappa_tear,
                    kappa_fold=self.kappa_fold,
                    allow_tears=self.allow_tears,
                    allow_folds=self.allow_folds
                )
                L0 = _build_weighted_gmrf(
                    D=D,
                    grid_shape=self.grid_shapes[scale],
                    w_dict=w_dict,
                    reg_shape=self.reg_shape,
                    device=mu.device
                )
                L0_diag = L0.diagonal().clone()

            # Jacobian barrier
            if self.barrier_alpha > 0.0:
                mu = _jacobian_barrier_step(
                    x=x,
                    origin=origin,
                    h=h,
                    grid_shape=self.grid_shapes[scale],
                    mu=mu,
                    lr=self.barrier_lr,
                    steps=self.barrier_steps,
                    alpha=self.barrier_alpha,
                    beta=self.density_beta,
                    density_weight=rho_local,
                    use_adam=True
                )

            # ELBO
            elbo = _compute_elbo(
                stencil_idx=stencil_idx,
                stencil_w=stencil_w,
                mu=mu,
                v=v,
                r=r,
                delta=delta,
                L0=L0,
                L0_diag=L0_diag,
                pi0=pi0,
                alpha=alpha,
                sigma2_in=sigma2_in,
                log_ell_out=log_ell_out
            )
            elbo_val = float(elbo.item())
            elbo_hist.append(elbo_val)

            # Convergence
            if it >= self.min_iters and len(elbo_hist) >= 2:
                rel = abs(elbo_hist[-1] - elbo_hist[-2]) / (abs(elbo_hist[-1]) + 1e-8)
                if rel < self.tolerance:
                    break
        
        return mu, v, r, math.sqrt(max(sigma2_in, 1e-8)), alpha, pi0, elbo_hist

    # Field + GAT
    def _jepa_step(
        self,
        slices: List[SliceData],
        origin: torch.Tensor,
        h: float,
        scale: int
    ) -> None: 
        K = len(slices)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        for _ in range(self.jepa_steps):
            canon = [
                self.deformation.apply_full(
                    x=s.coords, 
                    slice_id=k, 
                    origin=origin, 
                    h=h
                )
                for k, s in enumerate(slices)
            ]

            # Use frozen target from encoder
            # NO GAT here!
            with torch.no_grad():
                z_target = [self.encoder.infer(s.expr) for s in slices]

            self.opt_reg.zero_grad()

            # JEPA => field(canon) ~ z_target
            loss_field = 0.0
            for k in range(K):
                phi_k = self.field(canon[k])
                loss_field = loss_field + ((phi_k - z_target[k]) ** 2).mean()

            # VICReg on GAT output
            with torch.no_grad():
                enc_feats = [self.encoder.infer(s.expr) for s in slices]
            z_gat = self.gat(enc_feats, canon, list(range(K)), scale)

            loss_vicreg = 0.0
            for z in z_gat:
                z_c = z - z.mean(dim=0, keepdim=True)
                std = (z_c.pow(2).mean(dim=0) + 1e-4).sqrt()
                var_term = torch.clamp(1.0 - std, min=0.0).mean()
                loss_vicreg = loss_vicreg + self.vicreg_var_reg * var_term
                if z.shape[0] > 1:
                    cov = (z_c.T @ z_c) / (z.shape[0] - 1)
                    off = cov - torch.diag(torch.diagonal(cov))
                    cov_term = off.pow(2).sum() / z.shape[1]
                    loss_vicreg = loss_vicreg + self.vicreg_cov_reg * cov_term

            total_loss = (
                self.jepa_weight *  loss_field 
                + self.vicreg_weight * loss_vicreg
            )

            total_loss.backward()
            self.opt_reg.step()

        # Canonical coordinates have changed => GAT cache is stale
        self.gat.flush_cache()

        # Re-enable encoder gradients
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def _finetune_encoder(
        self,
        slices: List[SliceData],
        origin: torch.Tensor,
        h: float
    ) -> None:
        self.encoder.train()
        for p in self.encoder.parameters():
            p.requires_grad_(True)

        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.finetune_lr)

        with torch.no_grad():
            canon = [
                self.deformation.apply_scale(
                    x=s.x,
                    slice_id=k,
                    origin=origin,
                    h=h
                )
                for k, s in enumerate(slices)
            ]
            phi = [self.field(c) for c in canon]

        x_all = torch.cat([s.expr for s in slices], dim=0)
        phi_all = torch.cat(phi, dim=0)

        for _ in range(self.finetune_steps):
            batch_idx = torch.randperm(x_all.shape[0], device=self.device)[:self.ssl_batch_size]
            batch_x     = x_all[batch_idx]
            batch_phi   = phi_all[batch_idx]

            opt.zero_grad()
            z = self.encoder.infer(batch_x)
            loss_align = ((z - batch_phi) ** 2).mean()      # SAME AS JEPA LOSS

            out = self.encoder(batch_x)
            loss_ssl = self.encoder_loss(
                z1=out["z1"],
                z2=out["z2"],
                x_true=batch_x,
                x_recon=out["x_recon"]
            )["loss"]

            total_loss = self.finetune_weights * loss_align + loss_ssl
            total_loss.backward()
            opt.step()

        self.encoder.eval()

    def fit(
        self,
        slices: List[SliceData],
        origin: torch.Tensor = None,
        h: float = None,
        verbose: bool = True
    ) -> None:
        device = self.device
        K = len(slices)
        D = self.D

        for s in slices:
            s.x = s.x.to(device)
            s.expr = s.expr.to(device)

        # Auto-infer origin/h if not provided
        if origin is None or h is None:
            all_x = torch.cat([s.x for s in slices], dim=0)
            x_min = all_x.min(dim=0).values
            x_max = all_x.max(dim=0).values
            extent = float((x_max - x_min).max().item())
            L_fine = max(self.grid_shapes[-1])
            h = extent / max(L_fine - 3, 1)
            origin = x_min - 2.0 * h

        origin = origin.to(device)

        for round_idx in range(self.n_rounds):
            if verbose:
                print(f"\n=== Round {round_idx + 1}/{self.n_rounds} ===")

            # Step 1 - SSL pretraining/refresh
            self._ssl_pretrain_encoder(slices=slices, verbose=verbose)

            # Step 2 - Rest frozen mask each round
            frozen = [
                torch.zeros(s.x.shape[0], dtype=torch.bool, device=device)
                for s in slices
            ]
            final_states = [None] * K

            # Step 3 - Multi-scale registration
            for scale in range(self.S):
                if verbose:
                    print(f"[Scale {scale}] grid={self.grid_shapes[scale]}")

                # Inputs at this scale
                if scale == 0:
                    inputs = [s.x for s in slices]
                else:
                    inputs = [
                        self._apply_upto(
                            x=s.x,
                            slice_id=k,
                            up_to_scale=scale - 1,
                            origin=origin,
                            h=h
                        )
                        for k, s in enumerate(slices)
                    ]

                # Stencils
                stencils = [
                    _eval_stencil(
                        x=input,
                        origin=origin,
                        h=h,
                        grid_shape=self.grid_shapes[scale]
                    )
                    for input in inputs
                ]

                # Prior(s)
                L0 = _build_gmrf(
                    grid_shape=self.grid_shapes[scale],
                    shape=self.reg_shape
                ).to(device)
                L0_diag = L0.diagonal().clone()

                # Compute rho's
                rho_src = F.softplus(self.log_rho_src).detach() + 1e-3
                rho_tgt = F.softplus(self.log_rho_tgt).detach() + 1e-3

                # Targets
                if verbose:
                    print(f"    Computing displacement targets")
                targets = self._compute_targets(
                    slices=slices,
                    inputs=inputs,
                    origin=origin,
                    h=h,
                    scale=scale,
                    rho_src=rho_src,
                    rho_tgt=rho_tgt
                )

                if verbose:
                    print(f"    Running JEPA <> CAVI iterations")
                for it in range(self.n_iters):
                    slice_elbos = []
                    for k in range(K):
                        mu_k, v_k, r_k, sigma_k, alpha_k, pi0_k, elbo_hist = self._cavi_slice(
                            x=inputs[k],
                            delta=targets[k],
                            stencil_idx=stencils[k][0],
                            stencil_w=stencils[k][1],
                            L0=L0,
                            L0_diag=L0_diag,
                            frozen=frozen[k],
                            origin=origin,
                            h=h,
                            scale=scale
                        )

                        with torch.no_grad():
                            self.deformation.mu[scale][k].data = mu_k
                            self.deformation.log_var[scale][k].data = torch.log(
                                v_k.unsqueeze(0).expand(D, -1).clamp_min(1e-12)
                            )

                        final_states[k] = dict(
                            r=r_k,
                            sigma=sigma_k,
                            alpha=alpha_k,
                            pi0=pi0_k
                        )
                        slice_elbos.append(elbo_hist[-1] if elbo_hist else 0.0)

                    if self.jepa_weight > 0.0:
                        self._jepa_step(
                            slices=slices,
                            origin=origin,
                            h=h,
                            scale=scale
                        )

                    mean_elbo = sum(slice_elbos) / max(K, 1)
                    self.elbo_history.append(mean_elbo)

                    if verbose:
                        print(f"    iter {it}: mean ELBO = {mean_elbo:.4f}")


                    if it >= self.min_iters and len(self.elbo_history) >= 2:
                        rel = abs(self.elbo_history[-1] - self.elbo_history[-2]) / (abs(self.elbo_history[-1]) + 1e-8)
                        if rel < self.tolerance:
                            if verbose:
                                print(f"    Coverged at iter {it}")
                            break

                # Freeze and update rho's
                for k in range(K):
                    st = final_states[k]
                    if st is not None:
                        frozen[k] = frozen[k] | (st["r"] > self.freeze_threshold)

                        mean_r = float(st["r"].mean().item())
                        delta_lr = self.rho_lr * (mean_r - self.rho_target_inlier)
                        
                        with torch.no_grad():
                            self.log_rho_src[k] += delta_lr
                            self.log_rho_tgt[k] += delta_lr

                # Flush GAT cache
                self.gat.flush_cache()

            # Step 4 - Finetune encoder
            self._finetune_encoder(
                slices=slices,
                origin=origin,
                h=h
            )

        self.per_slice_final = final_states
        return self._build_result(
            slices=slices,
            origin=origin,
            h=h
        ) 


    @torch.no_grad()
    def _build_result(
        self,
        slices: List[SliceData],
        origin: torch.Tensor,
        h: float
    ) -> dict:
        canonical, embeddings, inlier = [], [], []
        for k, s in enumerate(slices):
            c_k = self.deformation.apply_full(
                x=s.x,
                slice_id=k,
                origin=origin,
                h=h
            )

            canonical.append(c_k)
            embeddings.append(self.field(c_k))
            inlier.append(self.per_slice_final[k]["r"])

        return {
            "canonical": canonical,
            "embeddings": embeddings,
            "inlier_rates": inlier,
            "field": self.field,
            "encoder": self.encoder,
            "deformation": self.deformation,
            "origin": origin,
            "h": h,
            "elbo_hist": self.elbo_history,
            "ssl_hist": self.ssl_history,
            "rho_src": F.softplus(self.log_rho_src).detach(),
            "rho_tgt": F.softplus(self.log_rho_tgt).detach(),
        }