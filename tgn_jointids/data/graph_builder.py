"""Sliding-window communication-graph construction (Section 3.1).

Each window becomes a directed graph G_t = (V_t, E_t): nodes are IPv4
addresses, edges are flows. Every piece of metadata is stored as a tensor
so that it survives PyTorch Geometric mini-batching.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .features import LabelSpace, ip_to_int

SRC_COL, DST_COL = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
TIME_COL, LABEL_COL = "FLOW_START_MILLISECONDS", "Attack"


def iter_time_windows(df: pd.DataFrame, time_col: str = TIME_COL,
                      window_ms: int = 5000, stride_ms: int = 2500
                      ) -> Iterator[tuple[int, pd.DataFrame]]:
    """Two-pointer sliding window over a chronologically sorted frame.

    Yields (window_id, snapshot); empty windows are skipped but still
    consume a window_id, so window_id is monotone in time.
    """
    t = df[time_col].values
    t0, t1 = t.min(), t.max()
    start, wid, left, right, n = t0, 0, 0, 0, len(df)
    while start <= t1:
        end = start + window_ms
        while left < n and t[left] < start:
            left += 1
        while right < n and t[right] < end:
            right += 1
        if right > left:
            yield wid, df.iloc[left:right]
        start += stride_ms
        wid += 1


class EnhancedGraphBuilder:
    """Builds PyG graphs from flow snapshots.

    Node features (35-d in the paper): mean of the raw flow descriptors over
    flows where the node is source and destination (averaged), 6 structural
    descriptors, and an ``temporal_dim``-d sinusoidal positional encoding.
    Edge features: the raw edge attributes + intra-window decay weight (Eq. 1).
    """

    def __init__(self, node_features: Sequence[str], edge_features: Sequence[str],
                 label_space: LabelSpace, priority_classes: Sequence[str] = (),
                 use_edge_types: bool = True, use_temporal_encoding: bool = True,
                 structural_features: bool = True, temporal_dim: int = 8,
                 edge_decay_factor: float = 0.1):
        self.node_features = list(node_features)
        self.edge_features = list(edge_features)
        self.labels = label_space
        self.benign_label = label_space.benign_label
        self.priority_classes = tuple(c.lower() for c in priority_classes)
        self.use_edge_types = use_edge_types
        self.use_temporal_enc = use_temporal_encoding
        self.use_structural = structural_features
        self.temporal_dim = int(temporal_dim)
        self.edge_decay = float(edge_decay_factor)
        self.protocol_to_type = {6: 0, 17: 1}  # TCP=0, UDP=1, other=2

    # ------------------------------------------------------------------
    def _edge_type(self, protocol) -> int:
        try:
            return self.protocol_to_type.get(int(protocol), 2)
        except Exception:
            return 2

    def _structural_features(self, node_map: dict, df: pd.DataFrame) -> np.ndarray:
        """out/in degree, #unique dst/src IPs, #unique dst/src ports."""
        S = np.zeros((len(node_map), 6), dtype=np.float32)
        og, ig = df.groupby(SRC_COL), df.groupby(DST_COL)
        out_deg, in_deg = og.size(), ig.size()
        udst = og[DST_COL].nunique()
        usrc = ig[SRC_COL].nunique()
        udport = og["L4_DST_PORT"].nunique() if "L4_DST_PORT" in df.columns else og.size() * 0
        usport = ig["L4_SRC_PORT"].nunique() if "L4_SRC_PORT" in df.columns else ig.size() * 0
        for ip, idx in node_map.items():
            S[idx] = [out_deg.get(ip, 0), in_deg.get(ip, 0), udst.get(ip, 0),
                      usrc.get(ip, 0), udport.get(ip, 0), usport.get(ip, 0)]
        return S

    def _temporal_encoding(self, n_nodes: int) -> np.ndarray:
        dim = self.temporal_dim
        if dim <= 0:
            return np.zeros((n_nodes, 0), dtype=np.float32)
        pos = np.arange(n_nodes, dtype=np.float32)[:, None]
        div = np.exp(np.arange(0, dim, 2, dtype=np.float32) * -(np.log(10000.0) / dim))
        pe = np.zeros((n_nodes, dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe

    def _edge_weights(self, df: pd.DataFrame) -> np.ndarray:
        """w_uv = exp(-λ_e (1 - t̃_uv))  (Eq. 1)."""
        if TIME_COL not in df.columns:
            return np.ones(len(df), dtype=np.float32)
        ts = df[TIME_COL].values.astype(np.float64)
        if ts.max() == ts.min():
            return np.ones(len(ts), dtype=np.float32)
        norm = (ts - ts.min()) / (ts.max() - ts.min())
        return np.exp(-self.edge_decay * (1.0 - norm)).astype(np.float32)

    def label_window(self, df: pd.DataFrame):
        """Priority-aware window labelling (Eqs. 3-4).

        y_bin = 1 iff any malicious flow. y_multi = first priority class
        present (in priority order), else majority vote among MALICIOUS
        flows only — benign flows never compete for the attack-type label.
        """
        labels = df[LABEL_COL].astype(str).str.lower()
        attack_mask = labels != self.benign_label
        y_bin = int(attack_mask.any())
        if y_bin == 0:
            return 0, self.benign_label, attack_mask
        attack_counts = labels[attack_mask].value_counts()
        for cls in self.priority_classes:
            if cls in attack_counts.index:
                return 1, cls, attack_mask
        return 1, attack_counts.idxmax(), attack_mask

    # ------------------------------------------------------------------
    def build_graph(self, df_snap: pd.DataFrame, window_id: int = 0) -> Data | None:
        if len(df_snap) == 0:
            return None

        ips = pd.concat([df_snap[SRC_COL], df_snap[DST_COL]]).astype(str).unique()
        node_map = {ip: i for i, ip in enumerate(ips)}
        n_nodes = len(node_map)
        node_ids_np = np.array([ip_to_int(ip) for ip in ips], dtype=np.int64)

        src_idx = df_snap[SRC_COL].astype(str).map(node_map).values
        dst_idx = df_snap[DST_COL].astype(str).map(node_map).values
        edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

        if self.use_edge_types and "PROTOCOL" in df_snap.columns:
            edge_type = torch.tensor(df_snap["PROTOCOL"].apply(self._edge_type).values,
                                     dtype=torch.long)
        else:
            edge_type = torch.zeros(len(src_idx), dtype=torch.long)

        # ---- node features ----
        tmp = df_snap.copy()
        for c in self.node_features:
            if c not in tmp.columns:
                tmp[c] = 0.0  # missing columns are zero-filled
        tmp[self.node_features] = tmp[self.node_features].fillna(0)
        X_stat = np.zeros((n_nodes, len(self.node_features)), dtype=np.float32)
        for grp_col in (SRC_COL, DST_COL):
            agg = tmp.groupby(grp_col)[self.node_features].mean()
            for ip in agg.index.astype(str):
                if ip in node_map:
                    X_stat[node_map[ip]] += agg.loc[ip].values.astype(np.float32)
        X_stat /= 2.0

        parts = [X_stat]
        if self.use_structural:
            parts.append(self._structural_features(node_map, df_snap))
        if self.use_temporal_enc and self.temporal_dim > 0:
            parts.append(self._temporal_encoding(n_nodes))
        X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

        # ---- edge features ----
        for c in self.edge_features:
            if c not in df_snap.columns:
                df_snap = df_snap.copy()
                df_snap[c] = 0.0
        ea_base = df_snap[self.edge_features].fillna(0).values.astype(np.float32)
        ew = self._edge_weights(df_snap).reshape(-1, 1)
        edge_attr = np.concatenate([ea_base, ew], axis=1)

        # ---- labels ----
        y_bin, y_multi_str, attack_mask = self.label_window(df_snap)
        y_node_np = np.zeros(n_nodes, dtype=np.int64)
        if y_bin == 1:
            atk = df_snap[attack_mask.values]
            for s in atk[SRC_COL].astype(str):
                y_node_np[node_map[s]] = 1
            for d in atk[DST_COL].astype(str):
                y_node_np[node_map[d]] = 1

        dist = np.zeros(self.labels.num_classes, dtype=np.float32)
        for a, cnt in df_snap[LABEL_COL].astype(str).str.lower().value_counts().items():
            if a in self.labels.attack2id:
                dist[self.labels.attack2id[a]] = cnt / len(df_snap)

        return Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            edge_type=edge_type,
            y_bin=torch.tensor([y_bin], dtype=torch.long),
            y_multi=torch.tensor([self.labels.to_id(y_multi_str)], dtype=torch.long),
            y_node=torch.tensor(y_node_np, dtype=torch.long),
            node_ids=torch.tensor(node_ids_np, dtype=torch.long),
            window_id=torch.tensor([int(window_id)], dtype=torch.long),
            attack_dist=torch.tensor(dist, dtype=torch.float32),
            attack_ratio=torch.tensor([float(attack_mask.mean())], dtype=torch.float32),
            n_flows=torch.tensor([len(df_snap)], dtype=torch.long),
            num_nodes=n_nodes,
        )


KEEP_KEYS = {"x", "edge_index", "edge_attr", "edge_type", "y_bin", "y_multi", "y_node",
             "node_ids", "window_id", "attack_dist", "attack_ratio", "n_flows"}


def clean_graphs(graphs: list[Data]) -> list[Data]:
    """Keep only tensor attributes so DataLoader batches correctly."""
    cleaned = []
    for g in graphs:
        new_g = Data()
        for key in KEEP_KEYS:
            val = getattr(g, key, None)
            if torch.is_tensor(val):
                setattr(new_g, key, val)
        new_g.num_nodes = int(g.x.size(0))
        cleaned.append(new_g)
    return cleaned


def build_graphs(df: pd.DataFrame, builder: EnhancedGraphBuilder,
                 window_ms: int, stride_ms: int, progress: bool = True) -> list[Data]:
    it = iter_time_windows(df, window_ms=window_ms, stride_ms=stride_ms)
    if progress:
        it = tqdm(it, desc="Building graphs")
    raw = [g for wid, snap in it if (g := builder.build_graph(snap, window_id=wid)) is not None]
    return clean_graphs(raw)
