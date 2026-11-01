"""Train-only feature scaling for node and edge attributes.

The stored graphs keep the raw NetFlow values. ``FeatureScaler`` is fitted on
the TRAINING partition only and applied when a graph bundle is loaded
(``features.normalization`` in the config), so the same ``.pt`` file serves
both the legacy (unscaled) and the corrected experiments.

Transform per selected dimension j:

    z_j = (sign(x_j) · log1p(|x_j|) - mu_j) / sd_j        (mode "log1p_zscore")
    z_j = (x_j - mu_j) / sd_j                             (mode "zscore")

mu_j, sd_j are computed on the transformed training values. Dimensions with
sd_j < eps (e.g. the zero-filled typo column) are centred but not divided.
Dimensions listed in ``skip`` (the sinusoidal encoding, the intra-window decay
weight, which already lie in [-1, 1]) are left unchanged.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

MODES = ("none", "zscore", "log1p_zscore")


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


class _DimScaler:
    def __init__(self, dim: int, mode: str, skip: Sequence[int] = (), eps: float = 1e-6):
        if mode not in MODES:
            raise ValueError(f"Unknown normalization mode: {mode}")
        self.dim, self.mode, self.eps = dim, mode, eps
        self.skip = sorted({int(s) % dim for s in skip}) if dim else []
        self.active = [j for j in range(dim) if j not in self.skip]
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.std = torch.ones(dim, dtype=torch.float64)

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        return _signed_log1p(x) if self.mode == "log1p_zscore" else x

    def fit(self, tensors: Iterable[torch.Tensor]) -> "_DimScaler":
        if self.mode == "none":
            return self
        n, s1, s2 = 0, torch.zeros(self.dim, dtype=torch.float64), torch.zeros(self.dim, dtype=torch.float64)
        for t in tensors:
            v = self._pre(t.double())
            n += v.size(0)
            s1 += v.sum(0)
            s2 += (v * v).sum(0)
        if n == 0:
            raise ValueError("Cannot fit a scaler on zero rows.")
        mean = s1 / n
        var = (s2 / n - mean * mean).clamp_min(0.0)
        std = var.sqrt()
        std = torch.where(std < self.eps, torch.ones_like(std), std)
        for j in self.skip:
            mean[j], std[j] = 0.0, 1.0
        self.mean, self.std = mean, std
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        out = x.clone().double()
        if self.active:
            a = self.active
            out[:, a] = (self._pre(out[:, a]) - self.mean[a]) / self.std[a]
        return out.to(x.dtype)

    def state(self) -> dict:
        return {"dim": self.dim, "mode": self.mode, "skip": self.skip,
                "mean": self.mean.tolist(), "std": self.std.tolist()}


class FeatureScaler:
    """Scales ``data.x`` and ``data.edge_attr``; statistics fitted on training graphs only."""

    def __init__(self, node_dim: int, edge_dim: int, mode: str = "log1p_zscore",
                 node_skip: Sequence[int] = (), edge_skip: Sequence[int] = ()):
        self.mode = mode
        self.node = _DimScaler(node_dim, mode, node_skip)
        self.edge = _DimScaler(edge_dim, mode, edge_skip)
        self.fitted = False

    def fit(self, train_graphs) -> "FeatureScaler":
        self.node.fit(g.x for g in train_graphs)
        self.edge.fit(g.edge_attr for g in train_graphs if g.edge_attr is not None)
        self.fitted = True
        return self

    def transform_graph(self, g):
        g.x = self.node.transform(g.x)
        if g.edge_attr is not None:
            g.edge_attr = self.edge.transform(g.edge_attr)
        return g

    def state(self) -> dict:
        return {"mode": self.mode, "node": self.node.state(), "edge": self.edge.state()}


def scaler_from_config(cfg, node_dim: int, edge_dim: int) -> FeatureScaler | None:
    """Build an (unfitted) scaler from ``features.normalization`` or return None.

    Default skips: the last ``graph.temporal_dim`` node dims (sinusoidal
    encoding) and the last edge dim (decay weight of Eq. 1).
    """
    norm = cfg.features.get("normalization", None)
    if not norm or norm.get("mode", "none") == "none":
        return None
    temporal_dim = int(cfg.graph.get("temporal_dim", 0) or 0)
    default_node_skip = list(range(node_dim - temporal_dim, node_dim)) if temporal_dim > 0 else []
    node_skip = norm.get("node_skip", None)
    edge_skip = norm.get("edge_skip", None)
    return FeatureScaler(node_dim, edge_dim, mode=norm.mode,
                         node_skip=default_node_skip if node_skip is None else node_skip,
                         edge_skip=[edge_dim - 1] if edge_skip is None else edge_skip)


def assert_finite_and_bounded(graphs, bound: float = 50.0, sample: int = 5000, seed: int = 0) -> dict:
    """Validation check: scaled features are finite and |z| stays below ``bound``
    on (a sample of) the given graphs. Returns the observed maxima."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), min(sample, len(graphs)), replace=False)
    xmax = max(float(graphs[i].x.abs().max()) for i in idx)
    emax = max(float(graphs[i].edge_attr.abs().max()) for i in idx)
    finite = all(torch.isfinite(graphs[i].x).all() and torch.isfinite(graphs[i].edge_attr).all()
                 for i in idx)
    if not finite:
        raise ValueError("Non-finite values after feature scaling.")
    return {"node_abs_max": xmax, "edge_abs_max": emax, "bound": bound,
            "within_bound": xmax < bound and emax < bound}
