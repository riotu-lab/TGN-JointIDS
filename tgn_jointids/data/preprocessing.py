"""Flow-level loading."""

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
