import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

from ...utils._tensor_utils import _off_diag


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


        