"""Per-window feature vectors for the non-graph baselines (XGBoost, Random Forest, Window-GRU).

    python scripts/baseline_features.py --config configs/nf_cicids2018_v3.yaml

Loads the same graph bundle, split and train-only log1p z-score scaling as TGN-JointIDS
(pipeline.load_graph_bundle) and summarises every window graph as one vector:
mean / std / min / max of the 35-d host features and of the 5-d flow (edge) features,
plus the number of hosts, flows (edges) and raw flows. No identifiers (IPs) are used.
Writes data/processed/baselines/<experiment_name>_winfeat.npz with, per partition
(train / val / stream): X, y_multi, y_bin, window_id, sorted by window_id.
"""

from pathlib import Path

import numpy as np
import torch

from _common import base_parser, load
from tgn_jointids.pipeline import load_graph_bundle

STATS = ("mean", "std", "min", "max")


def summarise(t: torch.Tensor) -> np.ndarray:
    t = t.float()
    if t.numel() == 0:
        return np.zeros(4 * t.shape[1], np.float32)
    std = t.std(0, unbiased=False)
    return torch.cat([t.mean(0), std, t.min(0).values, t.max(0).values]).numpy()


def window_vector(g) -> np.ndarray:
    counts = np.array([g.num_nodes, g.edge_index.shape[1], float(g.n_flows)], np.float32)
    return np.concatenate([summarise(g.x), summarise(g.edge_attr), np.log1p(counts)])


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg, _ = load(args)
    bundle = load_graph_bundle(cfg)
    nd, ed = bundle["node_dim"], bundle["edge_dim"]
    names = ([f"node{i}_{s}" for s in STATS for i in range(nd)] +
             [f"edge{i}_{s}" for s in STATS for i in range(ed)] + ["log_hosts", "log_edges", "log_flows"])
    out = {"feature_names": np.array(names), "classes": np.array(list(cfg.data.classes)),
           "t0": np.array(bundle["t0"])}
    for part in ("train", "val", "stream"):
        gs = bundle[part]
        wid = np.array([int(g.window_id) for g in gs])
        order = np.argsort(wid, kind="stable")
        out[f"{part}_X"] = np.stack([window_vector(gs[i]) for i in order]).astype(np.float32)
        out[f"{part}_y_multi"] = np.array([int(gs[i].y_multi) for i in order])
        out[f"{part}_y_bin"] = np.array([int(gs[i].y_bin) for i in order])
        out[f"{part}_window_id"] = wid[order]
        print(part, out[f"{part}_X"].shape, flush=True)
    path = Path("data/processed/baselines") / f"{cfg.experiment_name}_winfeat.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print("wrote", path)


if __name__ == "__main__":
    main()
