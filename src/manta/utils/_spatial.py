import itertools
import torch

from typing import Tuple


@torch.no_grad()
def _binned_knn(
    x: torch.Tensor,
    k: int,
    bin_size: int = None,
    max_per_offset: int = 32,
    sample_size: int = 20_000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-based exact KNN via voxel bucketing.

    Args:
        x: [N, D] coordinates
        k: number of neighbours
        bin_size: if None, estimated from data
        max_per_offset: cap on candidates per offset direction
        sample_size: subsample size for voxel size estimation
    
    Returns:
        idx:        [N, k] neighbour indices; self-loop padding if k > available
        dist_sq:    [N, k] squared distances (0 for self-loops)
    """
    N, D = x.shape
    device = x.device

    # Bin size estimation
    if bin_size is None:
        S = min(sample_size, N)
        if S < 2:
            bin_size = 1.0
        else:
            perm = torch.randperm(N, device=device)[:S]
            sub = x[perm]
            
            d = torch.cdist(sub, sub)
            d.fill_diagonal_(float("inf"))  # Infinite distance to self

            nn_d = d.min(dim=1)
            nn_d = nn_d[torch.isfinite(nn_d)]

            bin_size = float(nn_d.median().item()) * 2.0 if nn_d.numel() > 0 else 1.0
            bin_size = max(bin_size, 1e-6)

    # Bin points
    x_min = x.min(dim=0).values
    b_idx = ((x - x_min) / bin_size).floor().long() 
    n_bin = b_idx.max(dim=0).values + 1

    strides = torch.ones(D, dtype=torch.long, device=device)
    for d in range(1, D):
        strides[d] = strides[d - 1] * n_bin[d - 1]

    bin_idx = (b_idx * strides).sum(dim=-1)        
    num_bin = int(n_bin.prod().item())

    # Sort points by bin
    _, sort_idx = torch.sort(bin_idx, stable=True)

    counts = torch.zeros(num_bin + 1, dtype=torch.long, device=device)
    counts.scatter_add_(0, bin_idx, torch.ones_like(bin_idx))

    offsets = torch.zeros(num_bin + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts[:num_bin], dim=0)

    # Gather candidates
    neighbour_offsets = torch.tensor(
        list(itertools.product([-1, 0, 1], repeat=D)),
        dtype=torch.long,
        device=device
    )

    all_cand, all_mask = [], []

    for i in range(neighbour_offsets.size(0)):
        off = neighbour_offsets[i]
        nb = b_idx + off.unsqueeze(0)
        valid = ((nb >= 0) & (nb < n_bin)).all(-1)
        nb_clamped = nb.clamp(min=0, max=n_bin - 1)
        nb_idx = (nb_clamped * strides).sum(dim=-1)

        start = offsets[nb_idx]
        end = offsets[nb_idx + 1]
        count = (end - start).clamp(min=0)

        max_count = int(count.max().item())
        if max_count == 0:
            continue
        max_count = min(max_count, max_per_offset)

        arr = torch.arange(max_count, device=device)
        slots = start.unsqueeze(1) + arr.unsqueeze(0)
        mask = (slots < end.unsqueeze(1)) & valid.unsqueeze(1)
        cand_idx = sort_idx[slots.clamp(max=N - 1)]
        all_cand.append(cand_idx)                   
        all_mask.append(mask)

    if not all_cand:
        idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)
        return idx, torch.zeros(N, k, device=device)

    cand_indices = torch.cat(all_cand, dim=1)
    cand_mask = torch.cat(all_mask, dim=1)

    # Distances and top k
    cand_x = x[cand_indices]
    d2 = ((cand_x - x.unsqueeze(1)) ** 2).sum(-1)
    d2 = d2.masked_fill(~cand_mask, float("inf"))

    self_idx = torch.arange(N, device=device).unsqueeze(1)
    d2 = d2.masked_fill(cand_indices == self_idx, float("inf"))

    k_eff = min(k, d2.size(1))
    top_d2, top_pos = torch.topk(d2, k=k_eff, dim=1, largest=False)
    top_idx = cand_indices.gather(1, top_pos)

    if k_eff < k:
        pad = k - k_eff
        top_idx = torch.cat([top_idx, self_idx.expand(-1, pad)], dim=1)
        top_d2  = torch.cat([top_d2, torch.full((N, pad), float("inf"), device=device)], dim=1)

    inf_mask = ~torch.isfinite(top_d2)
    top_idx     = torch.where(inf_mask, self_idx.expand_as(top_idx), top_idx)
    top_d2      = torch.where(inf_mask, torch.zeros_like(top_d2), top_d2)

    return top_idx, top_d2