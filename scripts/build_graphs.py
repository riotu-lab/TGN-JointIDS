"""Step 1 — flows CSV -> balanced flows -> sliding-window graphs -> fixed split.

    python scripts/build_graphs.py --config configs/nf_ton_iot_v3.yaml
"""

from collections import Counter
from pathlib import Path

import torch

from _common import base_parser, load
from tgn_jointids.data import balance_classes, build_graphs, load_flows, stratified_chronological_split
from tgn_jointids.pipeline import make_builder, make_label_space
from tgn_jointids.utils import ensure_dir, env_path


def main():
    args = base_parser(__doc__).parse_args()
    cfg, _ = load(args)
    labels = make_label_space(cfg)

    df = load_flows(env_path(cfg.paths.flows_csv))
    print(f"Loaded {len(df):,} flows\n{df['Attack'].value_counts()}")
    if cfg.data.balance_target:
        df = balance_classes(df, cfg.data.balance_target, seed=cfg.data.balance_seed)
        print(f"Balanced to {len(df):,} flows ({cfg.data.balance_target:,} per class)")

    graphs = build_graphs(df, make_builder(cfg, labels), cfg.graph.window_ms, cfg.graph.stride_ms)
    g0 = graphs[0]
    print(f"Graphs: {len(graphs):,} | node dim {g0.x.shape[1]} | edge dim {g0.edge_attr.shape[1]}")

    train, val, stream, split = stratified_chronological_split(
        graphs, cfg.split.train_ratio, cfg.split.val_ratio, seed=cfg.split.seed,
        sort_by=cfg.split.sort_by)
    for name, part in (("train", train), ("val", val), ("stream", stream)):
        dist = Counter(labels.id2attack[int(g.y_multi.item())] for g in part)
        print(f"{name:<7} {len(part):>7,}  {dict(dist)}")

    out = Path(env_path(cfg.paths.graphs))
    ensure_dir(out.parent)
    torch.save({"graphs": graphs, "split": split, "node_dim": int(g0.x.shape[1]),
                "edge_dim": int(g0.edge_attr.shape[1]), "classes": labels.classes,
                "config": cfg.to_dict()}, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
