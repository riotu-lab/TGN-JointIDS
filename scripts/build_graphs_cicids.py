"""Build the NF-CICIDS2018-v3 graph bundles (block-stratified split and later-day test).

    python scripts/build_graphs_cicids.py --config configs/nf_cicids2018_v3.yaml
    python scripts/build_graphs_cicids.py --config configs/nf_cicids2018_v3_later_day.yaml

split.protocol
  block_stratified  whole 60 s blocks -> train/val/stream 60/15/25 with every class in
                    every partition. Uses assign_blocks + rebalance_assignments from
                    scripts/build_graphs.py unchanged (the NF-ToN-IoT-v3 protocol).
  forward_in_time   later-day test: flows starting before split.test_start_utc -> train/val (whole
                    blocks, class-covered, val share split.val_ratio_within_train_period);
                    flows from the cut on -> stream (test). data.heldout_classes must not
                    occur before the cut and are not learned (binary label only).

Graphs are built block by block (memory resets per block; a window never mixes blocks)
in parallel worker processes; the result is identical to a serial build. Strict mode:
unknown attack labels and missing / NaN features raise. Writes the bundle (.pt, refuses
to overwrite), <bundle>.manifest.json and, with --stats-dir, dataset/split statistics.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import base_parser, load  # noqa: F401  (adds the repo root to sys.path)
from tgn_jointids.data.features import LabelSpace
from tgn_jointids.data.graph_builder import build_graphs
from tgn_jointids.pipeline import make_builder

TIME, END, LABEL = "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Attack"
META = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL",
        TIME, END, LABEL, "Label"]
PARTS = ("train", "val", "stream")


# ---------------------------------------------------------------------------- loading
def feature_columns(cfg) -> list[str]:
    f = cfg.features
    return list(dict.fromkeys(list(f.statistical) + list(f.behavioral) + list(f.edge)))


def full_label_space(cfg) -> LabelSpace:
    """Learned classes followed by held-out classes (strict)."""
    held = [c for c in (cfg.data.get("heldout_classes") or [])]
    return LabelSpace(list(cfg.data.classes) + held, benign_label=cfg.data.benign_label,
                      strict=True)


def validate_flows(df: pd.DataFrame, cfg) -> dict:
    """Strict checks on the flow table; raises on any violation."""
    feats = feature_columns(cfg)
    missing = [c for c in feats + META if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    nan = df[feats + [TIME, END]].isna().sum()
    if int(nan.sum()):
        raise ValueError(f"NaN values: {nan[nan > 0].to_dict()}")
    if not np.isfinite(df[feats].to_numpy(np.float64)).all():
        raise ValueError("Non-finite feature values")
    known = set(full_label_space(cfg).classes)
    seen = set(df[LABEL].unique())
    if seen - known:
        raise KeyError(f"Unknown attack labels {sorted(seen - known)}; known: {sorted(known)}")
    unused = sorted(known - seen)
    if unused:
        raise ValueError(f"Configured classes absent from the data: {unused}")
    benign = cfg.data.benign_label
    bad = int(((df[LABEL] == benign) != (df["Label"] == 0)).sum())
    if bad:
        raise ValueError(f"{bad} rows where Label and Attack disagree")
    if int((df[END] < df[TIME]).sum()):
        raise ValueError("Flows ending before they start")
    return {"rows": int(len(df)), "labels": {k: int(v) for k, v in
                                             df[LABEL].value_counts().items()}}


def load_official_flows(cfg) -> pd.DataFrame:
    path = cfg.paths.flows_csv
    df = pd.read_csv(path, usecols=lambda c: c in set(feature_columns(cfg) + META))
    df[LABEL] = df[LABEL].astype(str).str.strip().str.lower()
    # stable sort: ties in start time keep file order (deterministic node order)
    return df.sort_values(TIME, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------- splitting
def utc_ms(s: str) -> int:
    return int(pd.Timestamp(s).tz_convert("UTC").value // 10**6)


def assign_blocks_covered(counts: pd.DataFrame, ratios: dict[str, float], seed: int) -> dict:
    """Whole blocks -> partitions so that every class (column of ``counts``, flows per
    block) is present in every partition; then move blocks to approach the flow ratios
    without breaking coverage. Generic version of the block-split heuristic for any partitions."""
    names = list(ratios)
    sizes = counts.sum(axis=1).to_numpy(np.int64)
    pres = counts.to_numpy() > 0
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))
    rarity = 1.0 / np.maximum(pres.sum(axis=0), 1)
    score = (pres * rarity).sum(axis=1)
    order = order[np.argsort(-score[order], kind="stable")]
    target = {n: ratios[n] * sizes.sum() for n in names}
    tot = {n: 0 for n in names}
    cov = {n: np.zeros(pres.shape[1], np.int64) for n in names}
    assign = np.empty(len(counts), dtype=object)
    for b in order:
        def key(n):
            return (int((pres[b] & (cov[n] == 0)).sum()) * 1000,
                    (target[n] - tot[n]) / max(target[n], 1))
        n = max(names, key=key)
        assign[b] = n
        tot[n] += sizes[b]
        cov[n] += pres[b]
    for _ in range(100000):                       # greedy rebalancing, coverage-preserving
        moved = False
        for b in rng.permutation(len(counts)):
            src = assign[b]
            if (cov[src][pres[b]] <= 1).any():
                continue
            cur = sum(abs(tot[n] - target[n]) for n in names)
            best, best_dst = cur, None
            for dst in names:
                if dst == src:
                    continue
                new = (cur - abs(tot[src] - target[src]) - abs(tot[dst] - target[dst])
                       + abs(tot[src] - sizes[b] - target[src]) + abs(tot[dst] + sizes[b] - target[dst]))
                if new < best - 1e-8:
                    best, best_dst = new, dst
            if best_dst is not None:
                assign[b] = best_dst
                tot[src] -= sizes[b]
                tot[best_dst] += sizes[b]
                cov[src] -= pres[b]
                cov[best_dst] += pres[b]
                moved = True
        if not moved:
            break
    for n in names:
        if (cov[n] == 0).any():
            raise ValueError(f"{n}: classes without any block "
                             f"{list(counts.columns[cov[n] == 0])}")
    return {int(blk): str(a) for blk, a in zip(counts.index, assign)}


def split_block_stratified(df: pd.DataFrame, cfg) -> tuple[dict, dict]:
    from build_graphs import assign_blocks, rebalance_assignments
    s = cfg.split
    cols = [TIME, LABEL]
    _, assignment = assign_blocks(df[cols], block_ms=s.block_ms, seed=s.seed,
                                  train_ratio=s.train_ratio, val_ratio=s.val_ratio)
    sub = df[cols].assign(_block_id=df["_block"])
    _, assignment = rebalance_assignments(sub, assignment, train_ratio=s.train_ratio,
                                          val_ratio=s.val_ratio, seed=s.seed)
    return assignment, {"protocol": "block_stratified", "forward_in_time": False}


def split_forward_in_time(df: pd.DataFrame, cfg, t0: int) -> tuple[dict, dict]:
    s = cfg.split
    cut = utc_ms(s.test_start_utc)
    gap = int(s.get("min_gap_minutes", 60)) * 60000
    t = df[TIME].to_numpy()
    before, after = t[t < cut], t[t >= cut]
    if not len(before) or not len(after):
        raise ValueError("Cut leaves an empty side")
    info = {"protocol": "forward_in_time", "forward_in_time": True, "test_start_utc":
            s.test_start_utc, "cut_ms": cut, "last_train_period_flow_ms": int(before.max()),
            "first_test_flow_ms": int(after.min()),
            "idle_gap_minutes": (after.min() - before.max()) / 60000}
    if cut - before.max() < gap or after.min() - cut < gap:
        raise ValueError(f"Cut is within {gap // 60000} min of traffic: {info}")
    crossing = int((df.loc[df[TIME] < cut, END] >= cut).sum())
    info["flows_crossing_cut"] = crossing
    if crossing:
        raise ValueError(f"{crossing} flows start before and end after the cut")
    if (cut - t0) % s.block_ms:
        info["cut_block_aligned"] = False   # harmless: no flow lies near the cut
    held = list(cfg.data.get("heldout_classes") or [])
    pre = df[df[TIME] < cut]
    leak = sorted(set(pre[LABEL].unique()) & set(held))
    if leak:
        raise ValueError(f"Held-out classes before the cut: {leak}")
    counts = pd.crosstab(pre["_block"], pre[LABEL])
    vr = float(s.val_ratio_within_train_period)
    assignment = assign_blocks_covered(counts, {"train": 1 - vr, "val": vr}, seed=int(s.seed))
    for b in df.loc[df[TIME] >= cut, "_block"].unique():
        assignment[int(b)] = "stream"
    return assignment, info


# ---------------------------------------------------------------------------- building
_DF = None
_BUILDER = None
_WIN = None


def _init_worker(cfg_dict):
    global _BUILDER, _WIN
    from tgn_jointids.config import Config
    cfg = Config(cfg_dict)
    _BUILDER = make_builder(cfg, full_label_space(cfg))
    _WIN = (int(cfg.graph.window_ms), int(cfg.graph.stride_ms))


def _build_block(task):
    lo, hi = task
    block = _DF.iloc[lo:hi].drop(columns=["_block"])
    gs = build_graphs(block, _BUILDER, _WIN[0], _WIN[1], progress=False)
    b_t0 = int(block[TIME].min())
    for g in gs:
        g.window_id = torch.tensor([b_t0 + int(g.window_id.item()) * _WIN[1]], dtype=torch.long)
    # plain pickle bytes: torch's multiprocessing reducer would pass every tensor through
    # its own shared-memory mapping and exhaust the mapping limit on ~10^5 graphs
    return pickle.dumps(gs, protocol=pickle.HIGHEST_PROTOCOL)


def build_all(df, cfg, assignment, workers: int):
    """Graphs for every partition, blocks in time order. Returns graphs, split indices."""
    global _DF
    _DF = df
    blk = df["_block"].to_numpy()
    starts = np.flatnonzero(np.r_[True, blk[1:] != blk[:-1]])
    ends = np.r_[starts[1:], len(df)]
    if len(np.unique(blk[starts])) != len(starts):
        raise AssertionError("Blocks are not contiguous after sorting")
    tasks = {p: [] for p in PARTS}
    for lo, hi in zip(starts, ends):
        tasks[assignment[int(blk[lo])]].append((int(lo), int(hi)))
    graphs, split = [], {}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg.to_dict(),)) as pool:
        for p in PARTS:
            t = time.time()
            part = []
            for i, blob in enumerate(pool.imap(_build_block, tasks[p], chunksize=4)):
                part.extend(pickle.loads(blob))
                if (i + 1) % 500 == 0:
                    print(f"[build] {p}: {i + 1}/{len(tasks[p])} blocks, {len(part):,} graphs "
                          f"({time.time() - t:.0f}s)", flush=True)
            split[p] = list(range(len(graphs), len(graphs) + len(part)))
            graphs.extend(part)
            print(f"[build] {p}: {len(tasks[p])} blocks -> {len(part):,} graphs "
                  f"in {time.time() - t:.0f}s", flush=True)
    return graphs, split


# ---------------------------------------------------------------------------- labels
def remap_heldout(graphs, cfg) -> tuple[np.ndarray, dict]:
    """Full-space labels -> learned label space. Held-out (unlearned) classes keep
    y_bin = 1, get the placeholder y_multi = 0 and are recorded in y_multi_full."""
    full = full_label_space(cfg)
    learned = len(cfg.data.classes)
    y_full = np.array([int(g.y_multi.item()) for g in graphs], np.int64)
    n_held = 0
    for g in graphs:
        if int(g.y_multi.item()) >= learned:
            g.y_multi = torch.tensor([0], dtype=torch.long)
            n_held += 1
        g.attack_dist = g.attack_dist[:learned].contiguous()
    return y_full, {"classes_full": full.classes, "heldout_windows": n_held}


def mixed_window_stats(graphs, split, y_full, cfg) -> dict:
    """Windows with >= 2 attack classes and how often the priority rule, rather than the
    attack-majority vote, decides the label (computed on full-space flow shares)."""
    full = full_label_space(cfg)
    out = {}
    for p, idx in split.items():
        mixed, decided = 0, Counter()
        for i in idx:
            d = graphs[i].attack_dist_full
            att = d[1:]
            if int((att > 0).sum()) >= 2:
                mixed += 1
                maj = 1 + int(att.argmax())
                if maj != y_full[i]:
                    decided[f"{full.classes[maj]} -> {full.classes[y_full[i]]}"] += 1
        out[p] = {"windows_2plus_attack_classes": mixed,
                  "label_changed_by_priority": dict(decided)}
    return out


# ---------------------------------------------------------------------------- checks
def leakage_checks(df, assignment, split, graphs, cfg, t0, extra) -> dict:
    part = df["_block"].map(assignment)
    if part.isna().any():
        raise AssertionError("Blocks without assignment")
    res = {"flows_per_partition": {p: int((part == p).sum()) for p in PARTS},
           "blocks_per_partition": {p: int(sum(v == p for v in assignment.values())) for p in PARTS}}
    if sum(res["flows_per_partition"].values()) != len(df):
        raise AssertionError("Flows lost or duplicated by the split")
    # every graph's window start lies inside a block of its own partition
    for p, idx in split.items():
        for i in idx:
            b = (int(graphs[i].window_id.item()) - t0) // int(cfg.split.block_ms)
            if assignment.get(int(b)) != p:
                raise AssertionError(f"{p} graph {i} window starts in block {b} "
                                     f"assigned to {assignment.get(int(b))}")
        wid = [int(graphs[i].window_id.item()) for i in idx]
        if any(b < a for a, b in zip(wid, wid[1:])):
            raise AssertionError(f"{p}: windows not in chronological order")
    # exact duplicate flows (all loaded columns incl. time and IPs) across partitions
    key = pd.util.hash_pandas_object(df.drop(columns=["_block"]), index=False)
    dup = pd.DataFrame({"k": key.to_numpy(), "p": part.to_numpy()})
    multi = dup.groupby("k")["p"].nunique()
    res["exact_duplicate_flows_across_partitions"] = int((multi > 1).sum())
    if res["exact_duplicate_flows_across_partitions"]:
        raise AssertionError("Identical flow records in more than one partition")
    res["exact_duplicate_rows_within_partitions"] = int(key.duplicated().sum())
    # feature-only duplicates (same descriptors, different time/hosts): informative
    fkey = pd.util.hash_pandas_object(df[feature_columns(cfg) + [LABEL]], index=False)
    fd = pd.DataFrame({"k": fkey.to_numpy(), "p": part.to_numpy()})
    tr_keys = set(fd.loc[fd.p == "train", "k"])
    res["stream_flows_with_feature_identical_train_flow_pct"] = float(
        100 * fd.loc[fd.p == "stream", "k"].isin(tr_keys).mean())
    if extra.get("forward_in_time"):
        cut = extra["cut_ms"]
        tv = df.loc[part.isin(["train", "val"])]
        st = df.loc[part == "stream"]
        res["max_train_val_flow_end_ms"] = int(tv[END].max())
        res["min_test_flow_start_ms"] = int(st[TIME].min())
        if not (tv[TIME].max() < cut <= st[TIME].min() and tv[END].max() < cut):
            raise AssertionError("Temporal boundary violated")
        for p in ("train", "val"):
            if max(int(graphs[i].window_id.item()) for i in split[p]) + int(cfg.graph.window_ms) > cut:
                raise AssertionError(f"{p} window reaches past the cut")
    return res


def coverage(graphs, split, y_full, classes_full, learned) -> dict:
    out = {}
    for p, idx in split.items():
        c = Counter(int(y_full[i]) for i in idx)
        out[p] = {classes_full[k]: int(c.get(k, 0)) for k in range(len(classes_full))}
    for p in ("train", "val"):
        miss = [classes_full[k] for k in range(learned) if out[p][classes_full[k]] == 0]
        if miss:
            raise ValueError(f"{p}: learned classes without graphs: {miss}")
        held = [classes_full[k] for k in range(learned, len(classes_full)) if out[p][classes_full[k]]]
        if held:
            raise ValueError(f"{p}: held-out classes present: {held}")
    return out


def identity_stats(graphs, split) -> dict:
    tr = set()
    for i in split["train"]:
        tr.update(graphs[i].node_ids.tolist())
    out = {}
    for p in ("val", "stream"):
        fr = np.array([np.mean([n in tr for n in graphs[i].node_ids.tolist()]) for i in split[p]])
        out[p] = {"windows_all_ips_seen_in_train_pct": float(100 * (fr == 1).mean()),
                  "windows_no_ip_seen_in_train_pct": float(100 * (fr == 0).mean())}
    return out


# ---------------------------------------------------------------------------- main
def main():
    p = base_parser(__doc__)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 8))
    p.add_argument("--stats-dir", default=None)
    p.add_argument("--max-rows", type=int, default=0, help="debug only: first N rows")
    args = p.parse_args()
    cfg, _ = load(args)
    out = Path(cfg.paths.graphs)
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    if cfg.data.get("balance_target") is not None:
        raise ValueError("data.balance_target must be null (no over/under-sampling)")

    t_start = time.time()
    csv_sha = hashlib.sha256()
    with open(cfg.paths.flows_csv, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            csv_sha.update(chunk)
    df = load_official_flows(cfg)
    if args.max_rows:
        df = df.iloc[: args.max_rows].copy()
    data_info = validate_flows(df, cfg)
    data_info["csv_sha256"] = csv_sha.hexdigest()
    print(f"loaded {len(df):,} flows in {time.time() - t_start:.0f}s", flush=True)

    t0 = int(df[TIME].min())
    bms = int(cfg.split.block_ms)
    df["_block"] = ((df[TIME] - t0) // bms).astype(np.int64)
    proto = cfg.split.protocol
    if proto == "block_stratified":
        assignment, extra = split_block_stratified(df, cfg)
    elif proto == "forward_in_time":
        assignment, extra = split_forward_in_time(df, cfg, t0)
    else:
        raise ValueError(f"Unknown split.protocol {proto}")
    part_flows = df["_block"].map(assignment).value_counts()
    print(f"split ({proto}): " + ", ".join(f"{k} {int(v):,} flows" for k, v in part_flows.items())
          + f" | {json.dumps(extra, default=str)}", flush=True)

    # priority order (rarest first) must hold on the training flows as well
    part = df["_block"].map(assignment)
    tr_counts = df.loc[part == "train", LABEL].value_counts()
    pri = list(cfg.graph.priority_classes)
    pri_counts = [int(tr_counts.get(c, 0)) for c in pri]
    extra["priority_classes"] = dict(zip(pri, pri_counts))
    extra["priority_order_is_rarest_first_on_train"] = pri_counts == sorted(pri_counts)

    graphs, split = build_all(df, cfg, assignment, args.workers)
    for g in graphs:
        g.attack_dist_full = g.attack_dist.clone()
    y_full, lab = remap_heldout(graphs, cfg)
    mixed = mixed_window_stats(graphs, split, y_full, cfg)
    for g in graphs:
        del g.attack_dist_full
    checks = leakage_checks(df, assignment, split, graphs, cfg, t0, extra)
    learned = len(cfg.data.classes)
    cov = coverage(graphs, split, y_full, lab["classes_full"], learned)
    ident = identity_stats(graphs, split)

    split_sha = hashlib.sha256(json.dumps({k: list(map(int, v)) for k, v in split.items()},
                                          sort_keys=True).encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graphs": graphs, "split": split, "node_dim": int(graphs[0].x.shape[1]),
                "edge_dim": int(graphs[0].edge_attr.shape[1]), "classes": list(cfg.data.classes),
                "classes_full": lab["classes_full"], "y_multi_full": y_full.tolist(),
                "protocol": proto, "config": cfg.to_dict()}, out)
    manifest = {"bundle": str(out), "protocol": proto, **extra, "data": data_info,
                "t0_ms": t0, "block_ms": bms, "window_id_units": "unix_ms_window_start",
                "split_sha256": split_sha, "graphs_per_partition": {k: len(v) for k, v in split.items()},
                "graph_label_counts": cov, "heldout": lab, "mixed_windows": mixed,
                "leakage_checks": checks, "identity_overlap": ident,
                "block_assignment": {str(k): v for k, v in sorted(assignment.items())},
                "build_seconds": time.time() - t_start}
    with open(out.with_suffix(".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if args.stats_dir:
        sd = Path(args.stats_dir)
        sd.mkdir(parents=True, exist_ok=True)
        with open(sd / "split_statistics.json", "w") as f:
            json.dump({k: v for k, v in manifest.items() if k != "block_assignment"}, f, indent=1)
        flows = (df.assign(_p=part).groupby(["_p", LABEL]).size().unstack(0).fillna(0)
                 .astype(int).reindex(columns=list(PARTS)))
        flows.to_csv(sd / "flow_counts_by_partition.csv")
        pd.DataFrame(cov).reindex(columns=list(PARTS)).to_csv(sd / "graph_counts_by_partition.csv")
    print(json.dumps({k: manifest[k] for k in ("graphs_per_partition", "split_sha256",
                                               "leakage_checks", "mixed_windows")}, indent=1))
    print(f"saved {out} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
