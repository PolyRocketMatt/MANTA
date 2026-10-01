import torch

from typing import List, Tuple


@torch.no_grad()
def _prealign_stack(
    slices: List[Tuple[torch.Tensor, torch.Tensor]],
    inter_slice_distance: float = 50.0
) -> List[torch.Tensor]:    
    D = slices[0][0].shape[1]

    if D == 2:
        stacked = []
        K = len(slices)
        for k, slice in enumerate(slices):
            x = slice[0]
            z = torch.full(
                (x.size(0), 1),
                (k - K // 2) * inter_slice_distance,
                device=x.device,
                dtype=x.dtype
            )

            stacked.append(torch.cat([x, z], dim=1))
        return stacked
    return slices