"""TGN-JointIDS model (Sections 3.3-3.4).

[x_v || m̃_v] -> MLP -> L × (edge-aware GAT + residual + LayerNorm)
            -> SAGPool + mean pool -> MLP_pool -> {binary head, attack-type head}
node embeddings -> linear node-participation head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool


class TGNJointIDS(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 memory_dim: int = 96, num_layers: int = 3, num_heads: int = 4,
                 dropout: float = 0.25, edge_dim: int = 5, sagpool_ratio: float = 0.6):
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # GRU memory updater: input = node embedding, hidden = memory
        self.memory_updater = nn.GRUCell(input_size=hidden_dim, hidden_size=memory_dim)

        self.input_proj = nn.Sequential(                       # Eq. 8
            nn.Linear(in_dim + memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):                            # Eq. 9
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                                      dropout=dropout, edge_dim=edge_dim, add_self_loops=False))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.sagpool = SAGPooling(hidden_dim, ratio=sagpool_ratio)   # Eq. 10
        self.pool_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Dropout(dropout))

        def _head(out_dim):
            return nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim // 2, out_dim))

        self.binary_head = _head(2)                            # Eq. 11
        self.multi_head = _head(num_classes)                   # Eq. 12
        self.node_aux_head = nn.Linear(hidden_dim, 2)          # Eq. 13
        self.dropout = nn.Dropout(dropout)

    def encode(self, data, node_memory=None):
        """Returns node_emb [N,H], graph_emb [B,H], node_ids (list|None)."""
        x = data.x.float()
        edge_index, batch = data.edge_index, data.batch
        edge_attr = data.edge_attr.float() if data.edge_attr is not None else None

        if node_memory is not None and hasattr(data, "node_ids"):
            nids = data.node_ids.cpu().tolist()
            mem = node_memory.get(nids, x.device)               # read (Eq. 6)
        else:
            nids = None
            mem = torch.zeros(x.size(0), self.memory_dim, device=x.device)

        x = self.input_proj(torch.cat([x, mem], dim=-1))
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = F.elu(conv(x, edge_index, edge_attr=edge_attr))
            x = self.dropout(norm(x + x_res))
        node_emb = x

        try:
            x_pool, _, _, batch_pool, _, _ = self.sagpool(x, edge_index, edge_attr, batch=batch)
            graph_emb = global_mean_pool(x_pool, batch_pool)
        except Exception:
            # Fallback for degenerate (tiny) graphs
            graph_emb = global_mean_pool(x, batch)
        return node_emb, self.pool_proj(graph_emb), nids

    def forward(self, data, node_memory=None):
        """Returns (bin_logits [B,2], multi_logits [B,C], node_logits [N,2],
        node_emb [N,H], node_ids)."""
        node_emb, graph_emb, nids = self.encode(data, node_memory)
        return (self.binary_head(graph_emb), self.multi_head(graph_emb),
                self.node_aux_head(node_emb), node_emb, nids)

    @torch.no_grad()
    def get_graph_embedding(self, data, node_memory=None):
        return self.encode(data, node_memory)[1]
