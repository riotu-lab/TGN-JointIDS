"""Persistent IP-keyed TGN node memory (Section 3.2, Eqs. 6-7)."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class TGNNodeMemory:
    """Dictionary M: id -> R^{d_m} with last-update step s(v).

    Lookup  (Eq. 6): m̃_v = γ^{t - s(v)} · m_v       (unseen nodes -> 0)
    Update  (Eq. 7): m_v <- α · m_v + (1-α) · GRU(h_v, m̃_v)

    The memory lives on CPU and is keyed by the integer IPv4 identifier
    stored in ``data.node_ids``, so it follows each graph through batching.
    Node embeddings are detached before the update: gradients never flow
    through the persistent state.
    """

    def __init__(self, memory_dim: int, decay: float = 0.98, alpha: float = 0.7):
        self.memory_dim = memory_dim
        self.decay = decay
        self.alpha = alpha
        self.memory: dict[int, torch.Tensor] = {}
        self.last_update: dict[int, int] = {}
        self._step = 0
        self._n_new = 0

    def get(self, node_ids: Iterable[int], device) -> torch.Tensor:
        mems = []
        for nid in node_ids:
            nid = int(nid)
            if nid not in self.memory:
                self.memory[nid] = torch.zeros(self.memory_dim)
                self.last_update[nid] = self._step
                self._n_new += 1
            gap = self._step - self.last_update[nid]
            mems.append(self.memory[nid] * (self.decay ** gap))
        return torch.stack(mems).to(device)

    @torch.no_grad()
    def update_with_gru(self, node_ids, new_messages: torch.Tensor,
                        gru_cell: nn.GRUCell, device) -> None:
        old_mems = self.get(node_ids, device)
        updated = gru_cell(new_messages.detach().to(device), old_mems).cpu()
        for i, nid in enumerate(node_ids):
            nid = int(nid)
            if nid in self.memory:
                self.memory[nid] = self.alpha * self.memory[nid] + (1 - self.alpha) * updated[i]
            else:
                self.memory[nid] = updated[i]
            self.last_update[nid] = self._step
        self._step += 1

    def reset(self) -> None:
        self.memory.clear()
        self.last_update.clear()
        self._step = 0
        self._n_new = 0

    def __len__(self) -> int:
        return len(self.memory)

    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self.memory.values())

    def stats(self) -> str:
        return f"Nodes={len(self.memory)} | New={self._n_new} | Step={self._step}"
