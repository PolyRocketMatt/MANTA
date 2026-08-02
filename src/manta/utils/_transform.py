from __future__ import annotations

import torch

from dataclasses import dataclass


@dataclass(frozen=True)
class Transform:
    ndim: int
    rotation: torch.Tensor | None = None
    translation: torch.Tensor | None = None
    scale: torch.Tensor | None = None

    def __post_init__(self):
        if self.rotation is None:
            self.rotation = torch.eye(self.ndim)
            self.translation = torch.zeros(self.ndim)
            self.scale = torch.ones(self.ndim)

        if self.rotation.shape != (self.ndim, self.ndim):
            raise ValueError(
                f"rotation must have shape ({self.ndim}, {self.ndim})"
            )

        if self.translation.shape != (self.ndim,):
            raise ValueError(
                f"translation must have shape ({self.ndim,})"
            )

        if self.scale.shape != (self.ndim,):
            raise ValueError(
                f"scale must have shape ({self.ndim,})"
            )


    @staticmethod
    def identity(
        ndim: int,
        *,
        dtype=torch.float32,
        device=None
    ) -> "Transform":
        return Transform(
            rotation=torch.eye(ndim, dtype=dtype, device=device),
            translation=torch.zeros(ndim, dtype=dtype, device=device),
            scale=torch.ones(ndim, dtype=dtype, device=device),
            ndim=ndim
        )


    def compose(
        self,
        other: "Transform"
    ) -> "Transform":
        if self.ndim != other.ndim:
            raise ValueError("transform dimensionalities do not match")

        rotation = other.rotation @ self.rotation
        scale = other.scale * self.scale
        translation = (
            other.rotation @ (other.scale * self.translation) + other.translation
        )

        return Transform(
            rotation=rotation,
            translation=translation,
            scale=scale,
            ndim=self.ndim
        )


    def apply(
        self,
        points: torch.Tensor
    ) -> torch.Tensor:
        return (
            (points * self.scale) 
            @ self.rotation
            + self.translation
        )