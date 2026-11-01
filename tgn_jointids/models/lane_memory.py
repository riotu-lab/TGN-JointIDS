"""Differentiable, lane-local host memory of TGN-JointIDS (``memory.mode: tbptt``).

The memory is short-term: every lane's memory is reset when the lane starts a new unit (a 60 s
block), so the state never carries information across blocks, partitions or sessions.

Read and update (with gradients):

    read:    m~_v = gamma^(s - s_v) * m_v        (unseen in this unit -> 0)
    update:  m_v <- alpha * m_v + (1 - alpha) * GRU(h_v, m~_v)   (seen before)
             m_v <- GRU(h_v, m~_v)                               (first visit)

``s`` counts windows processed by the lane (gamma = 1 in the paper, i.e. no
decay). Keys are (lane, node_id), so one IP's state never mixes across lanes.
Tensors are never modified in place: every update stores a new tensor, which
keeps autograd valid across the K steps of truncated BPTT.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class LaneMemory:
    def __init__(self, memory_dim: int, alpha: float = 0.7, decay: float = 1.0):
        self.memory_dim = memory_dim
        self.alpha = float(alpha)
        self.decay = float(decay)
        self.state: dict[int, dict[int, torch.Tensor]] = {}
        self.last: dict[int, dict[int, int]] = {}
        self.lane_step: dict[int, int] = {}

    # ------------------------------------------------------------------ state
    def reset_lane(self, lane: int) -> None:
        self.state[lane] = {}
        self.last[lane] = {}
        self.lane_step[lane] = 0

    def reset(self) -> None:
        self.state.clear()
        self.last.clear()
        self.lane_step.clear()

    def detach(self) -> None:
        """Truncate BPTT: keep values, drop the autograd history."""
        for lane, mem in self.state.items():
            self.state[lane] = {k: v.detach() for k, v in mem.items()}

    def n_keys(self) -> int:
        return sum(len(m) for m in self.state.values())

    def requires_grad_any(self) -> bool:
        return any(v.requires_grad for m in self.state.values() for v in m.values())

    def _lane(self, lane: int) -> dict[int, torch.Tensor]:
        if lane not in self.state:
            self.reset_lane(lane)
        return self.state[lane]

    # ------------------------------------------------------------------ read / write
    def read(self, lanes: Sequence[int], nids: Sequence[int], device) -> torch.Tensor:
        """Decayed memory for each (lane, node) pair; zeros for unseen pairs.
        Pure: never inserts keys (prediction happens before any write)."""
        if len(lanes) != len(nids):
            raise ValueError("lanes and nids must align")
        zero = torch.zeros(self.memory_dim, device=device)
        out = []
        for lane, nid in zip(lanes, nids):
            lane, nid = int(lane), int(nid)
            m = self._lane(lane).get(nid)
            if m is None:
                out.append(zero)
                continue
            if self.decay != 1.0:
                m = m * (self.decay ** (self.lane_step[lane] - self.last[lane][nid]))
            out.append(m.to(device))
        if not out:
            return torch.zeros(0, self.memory_dim, device=device)
        return torch.stack(out)

    def write(self, lanes: Sequence[int], nids: Sequence[int], messages: torch.Tensor,
              gru: nn.GRUCell) -> None:
        """Eq. 7 for every (lane, node) of the current step. ``messages`` keeps
        its graph when gradients are wanted; pass a detached tensor otherwise."""
        keys = [(int(l), int(n)) for l, n in zip(lanes, nids)]
        if len(set(keys)) != len(keys):
            raise ValueError("A (lane, node) pair occurs twice in one step; "
                             "a lane must contribute at most one graph per step.")
        if not keys:
            return
        prev = self.read(lanes, nids, messages.device)
        updated = gru(messages, prev)
        for i, (lane, nid) in enumerate(keys):
            mem = self._lane(lane)
            old = mem.get(nid)
            mem[nid] = updated[i] if old is None else self.alpha * old + (1 - self.alpha) * updated[i]
            self.last[lane][nid] = self.lane_step[lane]
        for lane in {k[0] for k in keys}:
            self.lane_step[lane] += 1

    def view(self, node_lanes: Sequence[int]) -> "LaneMemoryView":
        return LaneMemoryView(self, node_lanes)


class LaneMemoryView:
    """Adapter with the ``get(node_ids, device)`` interface that
    ``TGNJointIDS.encode`` already uses; it supplies each node's lane."""

    def __init__(self, memory: LaneMemory, node_lanes: Sequence[int]):
        self.memory = memory
        self.node_lanes = list(node_lanes)

    def get(self, node_ids, device) -> torch.Tensor:
        return self.memory.read(self.node_lanes, node_ids, device)
