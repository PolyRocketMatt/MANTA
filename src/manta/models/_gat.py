import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple

from ..utils._spatial import _binned_knn


class SpotAttentionLayer(nn.Module):
    """
    Multi-head additive attention over a k-NN graph.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert out_dim % heads == 0

        self.H = heads
        self.Dh = out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(heads, 2 * self.Dh))

        nn.init.xavier_uniform_(self.a)

        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)
        self.act = nn.ELU()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = h.shape[0]
        Wh = self.W(h).view(N, self.H, self.Dh)
        src, tgt = edge_index[0], edge_index[1]

        src_a = (Wh[src] * self.a[:, :self.Dh]).sum(dim=-1)
        tgt_a = (Wh[tgt] * self.a[:, self.Dh:]).sum(dim=-1)
        
        e = F.leaky_relu(src_a + tgt_a, 0.2)
        e = e - e.max(dim=0, keepdim=True).values.detach()

        exp_e = self.drop(e.exp())
        denom = torch.zeros(N, self.H, device=h.device).index_add_(0, tgt, exp_e)
        alpha = exp_e / (denom[tgt] + 1e-12)

        msg = Wh[src] * alpha.unsqueeze(-1)
        out = torch.zeros(N, self.H, self.Dh, device=h.device).index_add_(0, tgt, msg)
        out = out.reshape(N, -1)

        return self.ln(self.act(out) + self.W(h))


class SliceAttentionLayer(nn.Module):
    """
    Cross-slice attention where each point get a message from
    every other slice.
    """
    def __init__(
        self,
        dim: int,
        heads: int = 4
    ) -> None:
        super().__init__()
        assert dim % heads == 0

        self.H = heads
        self.Dh = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        K = len(features)
        device = features[0].device

        u = torch.stack([f.mean(dim=0) for f in features], dim=0)
        Q = self.q(u).view(K, self.H, self.Dh)
        Kk = self.k(u).view(K, self.H, self.Dh)
        Vv = self.v(u).view(K, self.H, self.Dh)

        attention = torch.einsum("khd,jhd->khj", Q, Kk) / (self.Dh ** 0.5)
        eye = torch.eye(K, dtype=torch.bool, device=device)
        attention = attention.masked_fill(eye.unsqueeze(1), float("-inf"))
        alpha = attention.softmax(dim=-1)
        msg = torch.einsum("khj,jhd->khd", alpha, Vv).reshape(K, -1)
        return [self.ln(f + msg[k].unsqueeze(0)) for k, f in enumerate(features)]


class HeterogeneousGAT(nn.Module):
    """
    Heterogeneous graph-attention network for joint spatial-transcriptomic embedding.

    Caches k-NN edges keyed by (slice_id, scale). Call `flush_cache()` when
    coordinates change enough that a k-NN refresh is warranted.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        out_dim: int = 64,
        n_spot_layers: int = 2,
        heads: int = 4,
        k=8
    ) -> None:
        super().__init__()
        self.k = k
        self.input_proj = nn.Linear(in_dim, hidden)
        self.spot = nn.ModuleList([
            SpotAttentionLayer(
                in_dim=hidden,
                out_dim=hidden,
                heads=heads
            )
            for _ in range(n_spot_layers)
        ])
        self.slice_attention = SliceAttentionLayer(dim=hidden, heads=heads)
        self.out = nn.Linear(hidden, out_dim)

        # Cache: {(slice_id, scale): edge_index [2, E]}
        self._edge_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def flush_cache(self):
        self._edge_cache.clear()

    def _get_edges(
        self,
        x: torch.Tensor,
        slice_id: int,
        scale: int
    ) -> torch.Tensor:
        key = (slice_id, scale)
        if key in self._edge_cache:
            return self._edge_cache[key]
        N = x.shape[0]
        nbr_idx, _ = _binned_knn(
            x=x,
            k=self.k
        )
        device = x.device
        src = torch.arange(N, device=device).unsqueeze(1).expand(-1, self.k).reshape(-1)
        tgt = nbr_idx.reshape(-1)

        edge_index = torch.stack(
            [torch.cat([src, tgt]), torch.cat([tgt, src])], 
            dim=0
        )
        self._edge_cache[key] = edge_index
        return edge_index

    def forward(
        self,
        features: List[torch.Tensor],
        x: List[torch.Tensor],
        ids: List[int],
        scale: int
    ) -> List[torch.Tensor]:
        h_list = [self.input_proj(f) for f in features]

        for i in range(len(h_list)):
            edge_index = self._get_edges(
                x=x[i],
                slice_id=ids[i],
                scale=scale
            )
            h = h_list[i]

            for layer in self.spot:
                h = layer(h, edge_index)
            h_list[i] = h

        h_list = self.slice_attention(h_list)
        return [self.out(h) for h in h_list]