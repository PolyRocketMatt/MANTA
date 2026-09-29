import torch
import torch.nn as nn


class ExpressionEncoder(nn.Module):
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