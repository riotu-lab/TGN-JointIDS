"""Chronological lane scheduling over evaluation units (60 s blocks).

A unit is a temporal block: unit(g) = (window_start(g) - t0) // unit_ms.
Units are dealt to ``num_lanes`` lanes; each lane plays its units back to back,
windows inside a unit in chronological order. Step t batches the t-th pending
window of every active lane, so a batch mixes classes (different units) while
each lane moves forward in time. A lane contributes at most one graph per step.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np


def unit_ids(graphs, unit_ms: int, t0: int) -> np.ndarray:
    return np.array([(int(g.window_id.item()) - int(t0)) // int(unit_ms) for g in graphs],
                    dtype=np.int64)


@dataclass
class Step:
    idx: list = field(default_factory=list)      # graph indices, one per active lane
    lanes: list = field(default_factory=list)    # lane of each graph
    reset: list = field(default_factory=list)    # True when the graph starts a unit in its lane
    units: list = field(default_factory=list)    # unit id of each graph


class UnitLaneScheduler:
    """Deterministic given (seed, epoch). ``shuffle=False`` keeps units in time
    order (used for evaluation)."""

    def __init__(self, graphs, num_lanes: int, unit_ms: int, t0: int, seed: int = 0,
                 shuffle: bool = True):
        if num_lanes < 1:
            raise ValueError("num_lanes must be >= 1")
        self.num_lanes, self.seed, self.shuffle = int(num_lanes), int(seed), shuffle
        u = unit_ids(graphs, unit_ms, t0)
        t = np.array([int(g.window_id.item()) for g in graphs], dtype=np.int64)
        units: dict[int, list[int]] = {}
        for i in np.lexsort((np.arange(len(t)), t)):          # stable time order
            units.setdefault(int(u[i]), []).append(int(i))
        self.units = units
        self.unit_order = sorted(units)
        self.n_graphs = len(graphs)

    def lane_plan(self, epoch: int = 0) -> list[list[int]]:
        """Units per lane. Shuffled units are placed on the currently shortest
        lane (balanced lengths, still random)."""
        order = list(self.unit_order)
        if self.shuffle:
            rng = np.random.default_rng([self.seed, int(epoch)])
            order = [order[i] for i in rng.permutation(len(order))]
        heap = [(0, lane) for lane in range(self.num_lanes)]
        plan: list[list[int]] = [[] for _ in range(self.num_lanes)]
        for unit in order:
            length, lane = heapq.heappop(heap)
            plan[lane].append(unit)
            heapq.heappush(heap, (length + len(self.units[unit]), lane))
        return plan

    def steps(self, epoch: int = 0):
        plan = self.lane_plan(epoch)
        seqs = []
        for lane, lane_units in enumerate(plan):
            seq = []
            for unit in lane_units:
                for k, i in enumerate(self.units[unit]):
                    seq.append((i, k == 0, unit))
            seqs.append(seq)
        n_steps = max((len(s) for s in seqs), default=0)
        for t in range(n_steps):
            step = Step()
            for lane, seq in enumerate(seqs):
                if t < len(seq):
                    i, start, unit = seq[t]
                    step.idx.append(i)
                    step.lanes.append(lane)
                    step.reset.append(start)
                    step.units.append(unit)
            yield step

    def n_steps(self, epoch: int = 0) -> int:
        return max((sum(len(self.units[u]) for u in lane) for lane in self.lane_plan(epoch)),
                   default=0)
