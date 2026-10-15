"""Continual-adaptation components (Section 3.6): EWC and replay buffers."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


class EWC:
    """Diagonal-Fisher Elastic Weight Consolidation (Eq. 17).

    The Fisher is estimated from the binary-head log-likelihood on a subset
    of the training graphs, after offline training.
    """

    def __init__(self, model, dataloader: DataLoader, device, lambda_ewc: float = 150.0):
        self.model, self.device, self.lambda_ewc = model, device, lambda_ewc
        self.params, self.fisher = {}, {}
        self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        self.model.eval()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.params[name] = p.data.clone()
                self.fisher[name] = torch.zeros_like(p)
        n = 0
        for data in dataloader:
            data = data.to(self.device)
            self.model.zero_grad()
            bin_logits = self.model(data)[0]
            loss = F.nll_loss(F.log_softmax(bin_logits, dim=1), data.y_bin.view(-1))
            loss.backward()
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[name] += p.grad.data ** 2
            n += data.y_bin.size(0)
        for name in self.fisher:
            self.fisher[name] /= max(n, 1)
        self.model.zero_grad()

    def penalty(self) -> torch.Tensor:
        # NOTE: the reported runs use λ·Σ F (θ-θ*)², i.e. without the 1/2 of Eq. 17;
        # the factor is absorbed into λ_EWC.
        loss = 0.0
        for name, p in self.model.named_parameters():
            if name in self.fisher:
                loss = loss + (self.fisher[name] * (p - self.params[name]) ** 2).sum()
        return self.lambda_ewc * loss


class FIFOReplayBuffer:
    """Bounded FIFO buffer with uniform sampling (``replay.mode: fifo``).

    This is the buffer that was active in the runs reported in the paper
    (``collections.deque(maxlen=5000)``); see README, "Reproducibility notes".
    """

    def __init__(self, capacity: int = 5000):
        self.buf = deque(maxlen=capacity)

    def add(self, graph) -> None:
        self.buf.append(graph)

    def sample(self, n: int):
        return random.sample(self.buf, k=min(n, len(self.buf)))

    def __len__(self) -> int:
        return len(self.buf)


class ClassSpecificReplayBuffer:
    """Per-class reservoirs with a minimum quota k_min plus an overflow
    reservoir of capacity B_ov (``replay.mode: class_specific``)."""

    def __init__(self, num_classes: int, min_per_class: int = 10, extra_capacity: int = 2000):
        self.num_classes = num_classes
        self.min_per_class = min_per_class
        self.extra_capacity = extra_capacity
        self.class_bufs = {c: [] for c in range(num_classes)}
        self.class_seen = {c: 0 for c in range(num_classes)}
        self.extra_buf, self.extra_seen = [], 0

    def add(self, graph) -> None:
        c = int(graph.y_multi.item())
        self.class_seen[c] += 1
        if len(self.class_bufs[c]) < self.min_per_class:
            self.class_bufs[c].append(graph)
        else:
            idx = np.random.randint(0, self.class_seen[c])
            if idx < self.min_per_class:
                self.class_bufs[c][idx] = graph
        self.extra_seen += 1
        if len(self.extra_buf) < self.extra_capacity:
            self.extra_buf.append(graph)
        else:
            idx = np.random.randint(0, self.extra_seen)
            if idx < self.extra_capacity:
                self.extra_buf[idx] = graph

    def get_all(self):
        out = []
        for c in range(self.num_classes):
            out.extend(self.class_bufs[c])
        out.extend(self.extra_buf)
        return out

    def sample(self, n: int):
        pool = self.get_all()
        if not pool:
            return []
        idxs = np.random.choice(len(pool), min(n, len(pool)), replace=False)
        return [pool[i] for i in idxs]

    def per_class_counts(self) -> dict:
        return {c: len(b) for c, b in self.class_bufs.items()}

    def __len__(self) -> int:
        return sum(len(b) for b in self.class_bufs.values()) + len(self.extra_buf)

    def __bool__(self) -> bool:  # an empty buffer must not be falsy
        return True


def make_replay_buffer(cfg_replay, num_classes: int):
    if cfg_replay.mode == "fifo":
        return FIFOReplayBuffer(capacity=cfg_replay.fifo_capacity)
    if cfg_replay.mode == "class_specific":
        return ClassSpecificReplayBuffer(num_classes, cfg_replay.min_per_class,
                                         cfg_replay.extra_capacity)
    raise ValueError(f"Unknown replay mode: {cfg_replay.mode}")
