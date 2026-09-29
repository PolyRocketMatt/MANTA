import torch


def _geman_mcclure_weight(
    residual_sq: torch.Tensor,
    kappa: float,
    w_min: float = 0.0
) -> torch.Tensor:
    kappa = max(float(kappa), 1e-12)
    w = (kappa ** 2) / (kappa + residual_sq).pow(2).clamp(1e-24)
    return w.clamp(min=w_min) if w_min > 0.0 else w