"""Prequential test-then-train streaming loop (Section 3.6, Algorithm 1)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


def run_stream(model, tgn_memory, trainer, stream_graphs, device, update_interval: int = 32,
               conf_threshold: float = 0.85, id2attack: dict | None = None,
               log_every: int = 5000) -> tuple[pd.DataFrame, list[dict]]:
    """Process each graph once, in order.

    1. inference with the parameters/memory available BEFORE the label;
    2. record the prediction;
    3. write back detached node embeddings to the TGN memory;
    4. reveal the label -> accumulate; every ``update_interval`` graphs,
       insert into replay and perform one EWC-regularised update.

    Latency covers steps 1 and 3 (graph encoding, memory read/write,
    prediction). Returns a per-window DataFrame and the alert list.
    """
    records, alerts, accumulated = [], [], []
    model.eval()
    for i, graph in enumerate(stream_graphs):
        t0 = time.perf_counter()
        with torch.no_grad():
            g = Batch.from_data_list([graph]).to(device)
            bin_l, multi_l, _, node_emb, nids = model(g, tgn_memory)
            bin_p = F.softmax(bin_l, dim=1)[0]
            multi_p = F.softmax(multi_l, dim=1)[0]
        if nids is not None:
            tgn_memory.update_with_gru(nids, node_emb.detach(), model.memory_updater, device)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        bin_pred, multi_pred = int(bin_p.argmax()), int(multi_p.argmax())
        rec = {
            "window_id": int(graph.window_id.item()),
            "bin_true": int(graph.y_bin.item()),
            "bin_pred": bin_pred,
            "attack_prob": float(bin_p[1]),
            "bin_conf": float(bin_p.max()),
            "multi_true": int(graph.y_multi.item()),
            "multi_pred": multi_pred,
            "multi_conf": float(multi_p.max()),
            "latency_ms": lat_ms,
        }
        records.append(rec)
        if bin_pred == 1 and rec["bin_conf"] >= conf_threshold:
            alerts.append({"step": i, "window_id": rec["window_id"], "conf": rec["bin_conf"],
                           "type": (id2attack or {}).get(multi_pred, multi_pred)})

        accumulated.append(graph)                         # label revealed after prediction
        if len(accumulated) >= update_interval:
            trainer.incremental_update(accumulated, n_steps=1)
            model.eval()
            accumulated = []

        if log_every and (i + 1) % log_every == 0:
            recent = records[-log_every:]
            acc = np.mean([r["bin_pred"] == r["bin_true"] for r in recent])
            print(f"  {i + 1:,}/{len(stream_graphs):,} | acc(last {log_every})={acc:.4f} | "
                  f"latency={np.mean([r['latency_ms'] for r in recent]):.2f} ms | "
                  f"memory={len(tgn_memory):,}")

    if accumulated:
        trainer.incremental_update(accumulated, n_steps=1)
    return pd.DataFrame(records), alerts
