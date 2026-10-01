import torch
import torch.nn as nn

from ._mhe import MultiResolutionHashEncoding


class CanonicalField(nn.Module):
    """
    Canonical field Φ: R^D → R^d using MHE + 2-layer MLP
    """
    def __init__(
        self,
        D: int,
        d: int = 64,
        L: int = 16,
        F: int = 2,
        hidden: int = 128
    ) -> None:
        super().__init__()
        self.mhe = MultiResolutionHashEncoding(
            D=D, 
            L=L, 
            F=F
        )

        in_dim = L * F
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d)
        )

    def forward(self, c) -> torch.Tensor:
        self.mlp(self.mhe(c))