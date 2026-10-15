"""Flow-level loading and class balancing (Section 4)."""

from __future__ import annotations

import pandas as pd

TIME_COL = "FLOW_START_MILLISECONDS"
LABEL_COL = "Attack"


def load_flows(path: str, time_col: str = TIME_COL, label_col: str = LABEL_COL) -> pd.DataFrame:
    """Load a NetFlow-v3 CSV, normalise labels and sort chronologically."""
    df = pd.read_csv(path)
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    return df.sort_values(time_col).reset_index(drop=True)


def balance_classes(df: pd.DataFrame, target_samples: int, seed: int = 42,
                    time_col: str = TIME_COL, label_col: str = LABEL_COL) -> pd.DataFrame:
    """Resample every class to ``target_samples`` flows.

    Classes above the target are under-sampled without replacement; classes
    below it are over-sampled with replacement. The result is re-sorted
    chronologically. Because balancing erases the true class frequencies,
    the ENS weights are computed from the ORIGINAL counts (see losses.py).
    """
    parts = []
    for attack_type, group in df.groupby(label_col):
        replace = len(group) < target_samples
        sampled = group.sample(n=target_samples, replace=replace, random_state=seed).copy()
        sampled[label_col] = attack_type
        parts.append(sampled)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(time_col).reset_index(drop=True)
