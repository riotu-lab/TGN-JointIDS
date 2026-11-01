
"""Build leakage-controlled temporal-block graph partitions."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tgn_jointids.config import load_config
from tgn_jointids.data.graph_builder import build_graphs
from tgn_jointids.data.preprocessing import load_flows
from tgn_jointids.pipeline import make_builder, make_label_space


# Import the configuration using the project's existing loader.
# This script accepts a Config object through build_block_dataset().


def assign_blocks(df, block_ms=60000, seed=42,
                  train_ratio=0.60, val_ratio=0.15):
    """Assign disjoint time blocks using multilabel coverage heuristics.

    This is a class-covered randomized-block protocol, NOT a
    forward-in-time evaluation.
    """
    df = df.copy()
    t0 = df["FLOW_START_MILLISECONDS"].min()

    df["_block_id"] = (
        (df["FLOW_START_MILLISECONDS"] - t0) // block_ms
    ).astype(np.int64)

    counts = pd.crosstab(df["_block_id"], df["Attack"])
    classes = list(counts.columns)

    rng = np.random.default_rng(seed)
    block_ids = counts.index.to_numpy()
    rng.shuffle(block_ids)

    # Scarce-class blocks are assigned first.
    class_frequency = (counts > 0).sum(axis=0)
    rarity = 1.0 / class_frequency.clip(lower=1)

    block_ids = sorted(
        block_ids,
        key=lambda b: float(
            ((counts.loc[b] > 0) * rarity).sum()
        ),
        reverse=True
    )

    names = ["train", "val", "stream"]
    ratios = np.array([
        train_ratio,
        val_ratio,
        1 - train_ratio - val_ratio
    ])

    assignment = {}
    totals = {s: 0 for s in names}
    coverage = {s: set() for s in names}
    target = ratios * len(df)

    for block_id in block_ids:
        row = counts.loc[block_id]
        present = set(row.index[row > 0])
        n = int(row.sum())

        def score(name):
            missing = len(present - coverage[name])
            capacity = (target[names.index(name)] - totals[name])
            return (missing * 1000, capacity / max(target[names.index(name)], 1))

        chosen = max(names, key=score)
        assignment[int(block_id)] = chosen
        totals[chosen] += n
        coverage[chosen].update(present)

    partitions = {
        name: df.loc[df["_block_id"].map(assignment) == name].copy()
        for name in names
    }

    for name in names:
        partitions[name] = partitions[name].sort_values(
            "FLOW_START_MILLISECONDS"
        )

    return partitions, assignment




def rebalance_assignments(df, assignment, train_ratio=0.60,
                          val_ratio=0.15, seed=42,
                          max_iterations=10000):
    """Rebalance whole blocks while preserving class coverage."""
    import numpy as np
    import pandas as pd

    names = ("train", "val", "stream")
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "stream": 1 - train_ratio - val_ratio
    }

    counts = pd.crosstab(df["_block_id"], df["Attack"])
    sizes = counts.sum(axis=1)
    presence = counts.gt(0)

    assignment = dict(assignment)
    total = len(df)
    targets = {s: ratios[s] * total for s in names}
    totals = {s: 0 for s in names}
    coverage = {
        s: {c: 0 for c in counts.columns}
        for s in names
    }

    block_classes = {
        int(b): list(counts.columns[presence.loc[b]])
        for b in counts.index
    }

    for block_id, split in assignment.items():
        totals[split] += int(sizes.loc[block_id])
        for c in block_classes[block_id]:
            coverage[split][c] += 1

    rng = np.random.default_rng(seed)
    block_ids = list(assignment)
    rng.shuffle(block_ids)

    for _ in range(max_iterations):
        improved = False

        for block_id in block_ids:
            src = assignment[block_id]
            size = int(sizes.loc[block_id])
            classes = block_classes[block_id]

            if any(coverage[src][c] <= 1 for c in classes):
                continue

            current = sum(
                abs(totals[s] - targets[s])
                for s in names
            )

            best_dst = None
            best_score = current

            for dst in names:
                if dst == src:
                    continue

                score = current
                score -= abs(totals[src] - targets[src])
                score -= abs(totals[dst] - targets[dst])
                score += abs(totals[src] - size - targets[src])
                score += abs(totals[dst] + size - targets[dst])

                if score < best_score - 1e-8:
                    best_score = score
                    best_dst = dst

            if best_dst is not None:
                assignment[block_id] = best_dst
                totals[src] -= size
                totals[best_dst] += size

                for c in classes:
                    coverage[src][c] -= 1
                    coverage[best_dst][c] += 1

                improved = True
                break

        if not improved:
            break

    partitions = {
        s: df.loc[
            df["_block_id"].map(assignment) == s
        ].sort_values("FLOW_START_MILLISECONDS").copy()
        for s in names
    }

    assert sum(len(p) for p in partitions.values()) == len(df)
    assert all(
        len(coverage[s]) == len(counts.columns)
        and all(v > 0 for v in coverage[s].values())
        for s in names
    )

    print("\nREBALANCED BLOCK SPLIT")
    for s in names:
        print(
            f"{s}: {len(partitions[s]):,} flows "
            f"({len(partitions[s])/total:.2%}), "
            f"{len(partitions[s]['_block_id'].unique())} blocks"
        )

    return partitions, assignment


def build_block_dataset(cfg):
    """Build leakage-controlled, class-covered temporal-block graphs."""
    labels = make_label_space(cfg)
    df = load_flows(cfg.paths.flows_csv)

    if cfg.data.balance_target is not None:
        raise ValueError(
            "Set data.balance_target: null for block-split evaluation."
        )

    time_col = "FLOW_START_MILLISECONDS"
    global_t0 = int(df[time_col].min())

    partitions, assignment = assign_blocks(
        df,
        block_ms=cfg.split.block_ms,
        seed=cfg.split.seed,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio
    )

    df["_block_id"] = (
        (df[time_col] - global_t0) // cfg.split.block_ms
    ).astype("int64")

    partitions, assignment = rebalance_assignments(
        df, assignment,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio,
        seed=cfg.split.seed
    )

    names = ("train", "val", "stream")
    assert len(assignment) == df["_block_id"].nunique()
    assert sum(len(p) for p in partitions.values()) == len(df)

    block_sets = {
        name: set(partitions[name]["_block_id"].unique())
        for name in names
    }
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert block_sets[a].isdisjoint(block_sets[b])

    graphs = []
    indices = {}
    graph_stats = {}
    builder = make_builder(cfg, labels)

    for name in names:
        part = partitions[name]
        part_graphs = []

        for block_id, block in part.groupby("_block_id", sort=True):
            print(
                f"[BUILD] split={name}, "
                f"block={block_id}, "
                f"flows={len(block):,}",
                flush=True
            )
        
            clean_block = block.drop(columns=["_block_id"])

            block_graphs = build_graphs(
                clean_block,
                builder,
                cfg.graph.window_ms,
                cfg.graph.stride_ms,
                progress=False
            )

            block_t0 = int(clean_block[time_col].min())

            for g in block_graphs:
                local_id = int(g.window_id.item())
                actual_start = (
                    block_t0 + local_id * cfg.graph.stride_ms
                )
                g.window_id = torch.tensor(
                    [actual_start], dtype=torch.long
                )

            part_graphs.extend(block_graphs)

        if not part_graphs:
            raise ValueError(f"No graphs generated for {name}")

        start_idx = len(graphs)
        graphs.extend(part_graphs)
        indices[name] = list(range(start_idx, len(graphs)))

        window_times = [
            int(g.window_id.item()) for g in part_graphs
        ]

        violations = sum(
            window_times[i] < window_times[i - 1]
            for i in range(1, len(window_times))
        )

        assert violations == 0, (
            f"{name}: {violations} temporal-order violations"
        )

        counts = Counter(int(g.y_multi.item()) for g in part_graphs)
        class_counts = {
            labels.id2attack[i]: counts.get(i, 0)
            for i in range(labels.num_classes)
        }

        missing = [
            cls for cls, count in class_counts.items()
            if count == 0
        ]
        if missing:
            raise ValueError(
                f"{name}: missing graph-level classes {missing}"
            )

        graph_stats[name] = {
            "flows": len(part),
            "flow_ratio": len(part) / len(df),
            "blocks": len(block_sets[name]),
            "graphs": len(part_graphs),
            "ordering_violations": violations,
            "class_counts": class_counts
        }

        print(
            f"{name}: {len(part):,} flows, "
            f"{len(part_graphs):,} graphs, "
            f"{len(block_sets[name]):,} blocks, "
            f"ordering violations={violations}",
            flush=True
        )

    out = Path(cfg.paths.graphs)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        raise FileExistsError(
            f"Output already exists: {out}. "
            "Choose another path to avoid overwriting."
        )

    torch.save({
        "graphs": graphs,
        "split": indices,
        "node_dim": int(graphs[0].x.shape[1]),
        "edge_dim": int(graphs[0].edge_attr.shape[1]),
        "classes": labels.classes,
        "config": cfg.to_dict()
    }, out)

    manifest = {
        "protocol": "class_covered_temporal_blocks",
        "forward_in_time": False,
        "block_ms": cfg.split.block_ms,
        "window_id_units": "unix_milliseconds_window_start",
        "block_assignment": {
            str(k): v for k, v in assignment.items()
        },
        "graph_stats": graph_stats
    }

    with open(out.with_suffix(".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved graphs: {out}")
    print(f"Saved audit: {out.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    import argparse
    import sys

    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from tgn_jointids.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_block_dataset(cfg)
