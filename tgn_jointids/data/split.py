"""Class-stratified train / validation / streaming split (Section 4.3)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_chronological_split(graphs, train_ratio: float = 0.60, val_ratio: float = 0.15,
                                   seed: int = 42, sort_by: Optional[str] = None):
    """Stratified 60/15/25 split on the window attack-type label.

    sort_by:
        None        -> keep the order returned by ``train_test_split``.
                       This is the behaviour of the code that produced the
                       reported results (see README, "Reproducibility notes").
        "window_id" -> re-sort each subset chronologically by window index.

    Returns (train_graphs, val_graphs, stream_graphs, indices_dict).
    """
    labels = [int(g.y_multi.item()) for g in graphs]
    idx = np.arange(len(graphs))
    train_idx, temp_idx = train_test_split(idx, train_size=train_ratio,
                                           stratify=labels, random_state=seed)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, stream_idx = train_test_split(temp_idx, train_size=val_ratio / (1 - train_ratio),
                                           stratify=temp_labels, random_state=seed)

    if sort_by == "window_id":
        key = lambda i: int(graphs[i].window_id.item())  # noqa: E731
        train_idx, val_idx, stream_idx = (sorted(s, key=key) for s in (train_idx, val_idx, stream_idx))
    elif sort_by is not None:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    indices = {"train": list(map(int, train_idx)), "val": list(map(int, val_idx)),
               "stream": list(map(int, stream_idx))}
    return ([graphs[i] for i in indices["train"]], [graphs[i] for i in indices["val"]],
            [graphs[i] for i in indices["stream"]], indices)
