import itertools
import torch
import torch.nn as nn


class MultiResolutionHashEncoding(nn.Module):
    """
    Multi-resolution has encoding based on 
        "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
    by Müller et al.

    Args:
        D: spatial dimension
        L: number of levels
        F: features per level
        T: hash table size
        base_res: resolution at level 0
        growth: resolution growth factor per level
    """

    def __init__(
        self,
        D: int,
        L: int = 16,
        F: int = 2,
        T: int = 2 ** 19,
        base_res: int = 16,
        growth: float = 1.5
    ) -> None:
        super().__init__()
        self.D = D
        self.L = L
        self.F = F
        self.T = T

        self.hash_tables = nn.Parameter(0.01 * torch.randn(L, T, F))
        primes = [1, 2654435761, 805459861, 3674653429]

        self.register_buffer(
            "primes",
            torch.tensor(
                primes[:D],
                dtype=torch.long
            )
        )

        self.register_buffer(
            "corners", 
            torch.tensor(
                list(itertools.product([0, 1], repeat=D)),
                dtype=torch.long
            )
        )

        self.register_buffer(
            "resolutions",
            base_res * (growth ** torch.arange(L, dtype=torch.float32))
        )

    def forward(self, x):
        N, D = x.shape
        L, F, T = self.L, self.F, self.T
        device= x.device

        # Scale coordinates per level
        xs = x.unsqueeze(1) * self.resolutions.view(1, L, 1)
        x_floor = xs.floor().long()
        x_frac = xs - x_floor.float()

        # Corner coordinates per (level, point, corner)
        corners = self.corners.view(1, 1, -1, D)            # [1, 1, 2^D, D]
        corner_coords = x_floor.unsqueeze(2) + corners      # [1, 1, 2^D, D]

        # Hash
        hashed = (corner_coords * self.primes.view(1, 1, 1, D)).sum(-1) % T

        # Multi-linear weights (in the number of dimensions)
        cf = corners.float()
        xf = x_frac.unsqueeze(2)
        w = cf * xf + (1.0 - cf) * (1.0 - xf)
        weights = w.prod(dim=-1)

        # Gather
        level_offsets = (torch.arange(L, device=device) * T).view(1, L, 1)
        flat_idx = hashed + level_offsets
        ht_flat = self.hash_tables.reshape(-1, F)
        features = ht_flat[flat_idx]

        out = (features * weights.unsqueeze(-1)).sum(dim=2)
        return out.reshape(N, L * F)