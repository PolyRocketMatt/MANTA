import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class FeatureAugmentor(nn.Module):
    """
    Independent dropout + Gaussian noise on the same expression vector.
    """
    def __init__(self, p_drop: float = 0.1, eta: float = 0.01):
        super().__init__()
        self.p_drop = p_drop
        self.eta = eta

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand_like(x) > self.p_drop).to(torch.float32)
        noise = torch.rand_like(x) * self.eta
        return mask * x + noise

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        return self._augment(x), self._augment(x)


class MLP(nn.Module):
    """
    Generic MLP block with GELU activation and LayerNorm.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
        final_ln: bool = False
    ) -> None:
        super().__init__()

        layers = []
        d_in = in_dim
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(d_in, hidden_dim), 
                nn.GELU(), 
                nn.LayerNorm(hidden_dim)
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))

        if final_ln:
            layers.append(nn.LayerNorm(out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpressionEncoder(nn.Module):
    """
    Pure expression to latent encoder.
    """
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 2,
        decoder_hidden_dim: int = 128,
        dropout: float = 0.0,
        p_drop: float = 0.1,
        eta: float = 0.01
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.augmentor = FeatureAugmentor(p_drop=p_drop, eta=eta)
        self.encoder = MLP(
            in_dim=in_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            final_ln=True
        )
        self.decoder = MLP(
            in_dim=latent_dim,
            out_dim=in_dim,
            hidden_dim=decoder_hidden_dim,
            n_layers=2,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> dict:
        x1, x2 = self.augmentor(x)
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        x1_recon = self.decoder(z1)

        return {
            "x1": x1,
            "x2": x2,
            "z1": z1,
            "z2": z2,
            "x_recon": x1_recon
        }

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MantaEncoderLoss(nn.Module):
    """
    VICReg + reconstruction loss
    """
    def __init__(
        self,
        sim_coeff: float = 25.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        lambda_recon: float = 1.0,
        target_std: float = 1.0,
        eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.lambda_recon = lambda_recon
        self.target_std = target_std
        self.eps = eps

    def _invariance(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(z1, z2)

    def _variance(self, z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
        return torch.mean(F.relu(self.target_std - std) ** 2)

    def _covariance(self, z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, keepdim=True) + self.eps)
        N, D = z.shape
        cov = (z.T @ z) / N
        off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        return off_diag.pow(2).sum() / D

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        x_true: torch.Tensor,
        x_recon: torch.Tensor
    ) -> dict:
        inv = self._invariance(z1, z2)
        var = 0.5 * (self._variance(z1) + self._variance(z2))
        cov = 0.5 * (self._covariance(z1) + self._covariance(z2))
        recon = F.mse_loss(x_recon, x_true)

        total = (self.sim_coeff * inv
                 + self.var_coeff * var 
                 + self.cov_coeff * cov
                 + self.lambda_recon * recon)

        return {
            "loss": total,
            "inv": inv.detach(),
            "var": var.detach(),
            "cov": cov.detach(),
            "recon": recon.detach()
        }


def _ssl_train_step(
    model: nn.Module,
    loss_fn: MantaEncoderLoss,
    x: torch.Tensor,
    optimizer,
    grad_clip: float = 1.0
) -> dict:
    """
    Single SSL training step.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    losses = loss_fn(
        z1=out["z1"],
        z2=out["z2"],
        x_true=x,
        x_recon=out["x_recon"],
    )
    losses["loss"].backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return  { k : float(v.detach().item()) for k, v in losses.items() }



class OldExpressionEncoder(nn.Module):
    def __init__(
        self,
        G: int,
        out: int = 64,
        hidden: int = 256
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out)
        )

    def forward(self, e) -> torch.Tensor:
        return self.net(e)