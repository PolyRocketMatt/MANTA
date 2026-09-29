import torch
import torch.nn as nn

from ..primitives._bspline import _eval_stencil

from typing import Tuple


class MultiscaleFFD(nn.Module):
    """
    Multi-scale cubic B-spline free-form deformation

    Composition of scales: φ = φ_{S-1} o ... o φ_{0}
    Each scale has an independent B-spline control grid
    
    Parameters:
        mu[s]:      [K, D, P_s] control-point mean (per slice, axis, point)
        log_var[s]: [K, D, P_s] log-variance (per slice, axis, point)
    """
    def __init__(
        self,
        D: int,
        n_slices: int,
        grid_shapes: list[Tuple[int,...]],
        ref_idx: int = None
    ) -> None:
        super().__init__()
        self.D = D
        self.K = n_slices
        self.grid_shapes = grid_shapes
        self.S = len(grid_shapes)
        self.ref_idx = ref_idx if ref_idx is not None else n_slices // 2

        self.mu = nn.ParameterList()
        self.log_var = nn.ParameterList()

        for shape in self.grid_shapes:
            P = 1
            for L in shape: 
                P *= L

            mu = torch.zeros(n_slices, D, P)
            lv = torch.full((n_slices, D, P), -4.0)

            self.mu.append(nn.Parameter(mu))
            self.log_var.append(nn.Parameter(lv))

    def variance(self, s) -> torch.Tensor:
        return self.log_var[s].exp()

    def apply_scale(
        self,
        x: torch.Tensor,
        s: int,
        slice_id: int,
        origin: torch.Tensor,
        h: float,
        stencil: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply the scale-s deformation without composition
        """
        if stencil is None:
            idx, w = _eval_stencil(
                x=x,
                origin=origin,
                h=h,
                grid_shape=self.grid_shapes[s]
            )
        else:
            idx, w = stencil

        mu_s = self.mu[s][slice_id]
        contribution = mu_s[:, idx].permute(1, 2, 0)
        delta = (w.unsqueeze(-1) * contribution).sum(dim=1)

        return x + delta

    def apply_full(
        self,
        x: torch.Tensor,
        slice_id: int,
        origin: torch.Tensor,
        h: float,
        stencils: list[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply deformation on all scales with composition
        """
        for s in range(self.S):
            stencil = stencils[s] if stencils is not None else None
            x = self.apply_scale(
                x=x,
                s=s,
                slice_id=slice_id,
                origin=origin,
                h=h,
                stencil=stencil
            )
        return x