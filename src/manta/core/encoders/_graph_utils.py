import torch

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass 
class _GraphRepresentation:
    x: torch.Tensor     # [N, d] - feature matrix
    P: torch.Tensor     # [N, N] - transition matrix


@dataclass
class _GraphBatch:
    x: torch.Tensor     # [B, d] - feature matrix
    P: torch.Tensor     # [B, B] transition matrix
    idx: torch.Tensor   # [B] - original graph indices for each node


def _induced_subgraph(
    P: torch.Tensor,
    node_ids: torch.Tensor,
    num_nodes: int
) -> torch.Tensor:
    """
    Extract induced submatrix P[node_ids][:, node_ids] and remap indices.
    """
    node_ids = node_ids.to(torch.long)

    if not P.is_sparse:
        return P.index_select(0, node_ids).index_select(1, node_ids)

    P = P.coalesce()
    idx = P.indices() # (2, E)
    vals = P.values() # E

    selected = torch.zeros(num_nodes, device=P.device, dtype=torch.bool)
    selected[node_ids] = True

    keep = selected[idx[0]] & selected[idx[1]]
    idx = idx[:, keep]
    vals = vals[keep]

    remap = torch.full((num_nodes,), -1, device=P.device, dtype=torch.long)
    remap[node_ids] = torch.arange(node_ids.numel(), device=P.device)

    idx = remap[idx]
    sub_P = torch.sparse_coo_tensor(
        indices=idx,
        values=vals,
        size=(node_ids.numel(), node_ids.numel()),
        device=P.device
    ).coalesce()

    return sub_P


def _block_diag_sparse(
    matrices: Sequence[torch.Tensor]
) -> torch.Tensor:
    """
    Build a block-diagonal sparse COO tensor from sparse or dense square matrices.
    """
    if len(matrices) == 0:
        raise ValueError("matrices must be non-empty")

    if not any(matrix.is_sparse for matrix in matrices):
        # No sparse matrices > dense fallback
        sizes = [matrix.size(0) for matrix in matrices] # only one "size" needed, all square
        total = sum(sizes)
        out = matrices[0].new_zeros((total, total))
        offset = 0

        for matrix in matrices:
            n = matrix.size(0)
            out[offset:offset+n, offset:offset+n] = matrix
            offset += n

        return out

    indices_list = []
    values_list = []
    offset = 0
    device = matrices[0].device

    for matrix in matrices:
        if matrix.is_sparse:
            matrix = matrix.coalesce()
            idx = matrix.indices()
            vals = matrix.values()
        else:
            idx = matrix.nonzero(as_tuple=False).t()
            vals = matrix[idx[0], idx^[1]]

        idx = idx + offset
        indices_list.append(idx)
        values_list.append(vals)
        offset += matrix.size(0)

    indices = torch.cat(indices_list, dim=1)
    values = torch.cat(values_list, dim=0)
    out = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(offset, offset),
        device=device
    ).coalesce()

    return out


def _sample_pooled_batch(
    graphs: Sequence[_GraphRepresentation],
    nodes_per_graph: int,
    graph_order: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None
) -> _GraphBatch:
    """
    Sample the same number of nodes from each graph and build a block-diagonal
    batch to avoid cross-graph message passing.
    """
    if graph_order is None:
        graph_order = list(range(len(graphs)))
    device = graphs[0].x.device

    xs = []
    Ps = []
    ids = []
    

    for g_idx in graph_order:
        g = graphs[g_idx]
        if g.x.device != device:
            raise ValueError(
                f"expected graph node coordinates to be on `{device}`, is on `{g.x.device}`"
            )
        if g.P.device != device:
            raise ValueError(
                f"expected graph transition matrix to be on `{device}`, is on `{g.P.device}`"
            )

        n = g.x.sie(0)
        take = min(nodes_per_graph, n)

        perm = torch.randperm(
            n=n,
            generator=generator,
            device=device
        )[:take]
        x_sub = g.x.index_select(0, perm)
        P_sub = _induced_subgraph(
            P=g.P,
            node_ids=perm,
            num_nodes=n
        )

        xs.append(x_sub)
        Ps.append(P_sub)
        ids.append(
            torch.full(
                (take,),
                g_idx,
                device=device,
                dtype=torch.long
            )
        )

    x_batch = torch.cat(xs, dim=0)
    P_batch = _block_diag_sparse(matrices=Ps)
    ids_batch = torch.cat(ids, dim=0)

    return _GraphBatch(
        x=x_batch,
        P=P_batch,
        idx=ids_batch
    )