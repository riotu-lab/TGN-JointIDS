"""NF-CICIDS2018-v3 graph building: strict data checks, block-stratified and later-day
(forward-in-time) splitting, held-out classes, mixed-window labelling and leakage checks
(scripts/build_graphs_cicids.py).
Run:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_graphs_cicids as bgp  # noqa: E402
from tgn_jointids.config import Config  # noqa: E402
from tgn_jointids.data.features import LabelSpace  # noqa: E402
from tgn_jointids.data.graph_builder import build_graphs  # noqa: E402
from tgn_jointids.pipeline import make_builder  # noqa: E402

FEATS = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS"]
DAY = 86_400_000
T0 = bgp.utc_ms("2018-02-28T12:00:00Z")


def cfg(protocol="forward_in_time", classes=("benign", "a", "b"), held=("h",), pri=()):
    return Config({
        "paths": {"flows_csv": "unused.csv", "graphs": "unused.pt", "output_dir": "unused"},
        "data": {"benign_label": "benign", "strict_labels": True, "classes": list(classes),
                 "heldout_classes": list(held), "balance_target": None},
        "features": {"statistical": FEATS[:2], "behavioral": FEATS[2:], "edge": FEATS},
        "graph": {"window_ms": 5000, "stride_ms": 2500, "temporal_dim": 8,
                  "structural_features": True, "use_edge_types": True, "edge_decay": 0.1,
                  "strict_features": True, "priority_classes": list(pri)},
        "split": {"protocol": protocol, "test_start_utc": "2018-03-01T04:00:00Z",
                  "min_gap_minutes": 60, "val_ratio_within_train_period": 0.25, "seed": 0,
                  "block_ms": 60000, "train_ratio": 0.6, "val_ratio": 0.15}})


def flows(times, labels, seed=0, src=None):
    rng = np.random.default_rng(seed)
    n = len(times)
    df = pd.DataFrame({f: rng.integers(1, 1000, n).astype(float) for f in FEATS})
    df["IPV4_SRC_ADDR"] = src if src is not None else [f"10.0.0.{i % 7}" for i in range(n)]
    df["IPV4_DST_ADDR"] = [f"10.0.1.{i % 3}" for i in range(n)]
    df["L4_SRC_PORT"], df["L4_DST_PORT"], df["PROTOCOL"] = 1234, 80, 6
    df[bgp.TIME] = np.asarray(times, np.int64)
    df[bgp.END] = df[bgp.TIME] + 10
    df[bgp.LABEL] = labels
    df["Label"] = (df[bgp.LABEL] != "benign").astype(int)
    return df.sort_values(bgp.TIME, kind="mergesort").reset_index(drop=True)


def two_day_flows():
    """Pre-cut: 40 blocks with benign, a, b; post-cut: benign, a and held-out h."""
    t, y = [], []
    for k in range(40):
        base = T0 + k * 60000
        for j in range(6):
            t.append(base + j * 7000)
            y.append(["benign", "a", "b"][(k + j) % 3] if j < 3 else "benign")
    test0 = T0 + DAY                                           # 1 Mar 12:00 UTC
    for k in range(10):
        for j in range(4):
            t.append(test0 + k * 60000 + j * 9000)
            y.append(["benign", "a", "h", "benign"][j])
    return flows(t, y)


# ------------------------------------------------------------------ strict labels/features
def test_label_space_strict_raises_and_legacy_maps_to_benign():
    assert LabelSpace(["benign", "a"]).to_id("zzz") == 0
    with pytest.raises(KeyError):
        LabelSpace(["benign", "a"], strict=True).to_id("zzz")


def test_validate_flows_rejects_unknown_label_and_nan():
    c = cfg()
    df = two_day_flows()
    bgp.validate_flows(df, c)
    bad = df.copy()
    bad.loc[0, bgp.LABEL] = "unknown_attack"
    with pytest.raises(KeyError):
        bgp.validate_flows(bad, c)
    bad = df.copy()
    bad.loc[0, "IN_BYTES"] = np.nan
    with pytest.raises(ValueError):
        bgp.validate_flows(bad, c)
    with pytest.raises(KeyError):
        bgp.validate_flows(df.drop(columns=["TCP_FLAGS"]), c)


def test_builder_strict_features_raise_legacy_zero_fills():
    c = cfg()
    df = two_day_flows().iloc[:6].drop(columns=["TCP_FLAGS"])
    with pytest.raises(KeyError):
        make_builder(c, bgp.full_label_space(c)).build_graph(df)
    c.graph.strict_features = False
    lax = Config({**c.to_dict(), "data": {**c.data.to_dict(), "strict_labels": False}})
    g = make_builder(lax, LabelSpace(list(c.data.classes) + ["h"])).build_graph(df)
    assert g is not None


def test_builder_strict_labels_raise_inside_window():
    c = cfg()
    df = two_day_flows().iloc[:6].copy()
    df.loc[2, bgp.LABEL] = "not_a_class"
    with pytest.raises(KeyError):
        make_builder(c, bgp.full_label_space(c)).build_graph(df)


# ------------------------------------------------------------------ splitting
def test_forward_in_time_split_boundaries_and_coverage():
    c = cfg()
    df = two_day_flows()
    t0 = int(df[bgp.TIME].min())
    df["_block"] = (df[bgp.TIME] - t0) // 60000
    assignment, info = bgp.split_forward_in_time(df, c, t0)
    cut = info["cut_ms"]
    part = df["_block"].map(assignment)
    assert (df.loc[part == "stream", bgp.TIME] >= cut).all()
    assert (df.loc[part != "stream", bgp.TIME] < cut).all()
    assert info["flows_crossing_cut"] == 0
    for p in ("train", "val"):
        assert set(df.loc[part == p, bgp.LABEL]) == {"benign", "a", "b"}
    assert abs((part == "val").sum() / (part != "stream").sum() - 0.25) < 0.1


def test_forward_in_time_rejects_heldout_before_cut_crossing_flow_and_close_traffic():
    c = cfg()
    for mutate in ("held", "cross", "close"):
        df = two_day_flows()
        if mutate == "held":
            df.loc[0, bgp.LABEL] = "h"
        elif mutate == "cross":
            df.loc[df[bgp.TIME] < T0 + DAY, bgp.END] = T0 + 2 * DAY
        else:
            df.loc[len(df) - 1, bgp.TIME] = bgp.utc_ms("2018-03-01T04:10:00Z")
            df = df.sort_values(bgp.TIME, kind="mergesort").reset_index(drop=True)
            df[bgp.END] = df[bgp.TIME] + 10
        t0 = int(df[bgp.TIME].min())
        df["_block"] = (df[bgp.TIME] - t0) // 60000
        with pytest.raises(ValueError):
            bgp.split_forward_in_time(df, c, t0)


def test_assign_blocks_covered_is_deterministic_and_covers_rare_classes():
    rng = np.random.default_rng(1)
    counts = pd.DataFrame(rng.integers(0, 3, (60, 4)) * rng.integers(0, 2, (60, 4)),
                          columns=["benign", "x", "y", "z"])
    counts["benign"] += 5
    counts.loc[[3, 17], "z"] = [4, 2]
    counts.loc[[i for i in range(60) if i not in (3, 17)], "z"] = 0
    r = {"train": 0.8, "val": 0.2}
    a1 = bgp.assign_blocks_covered(counts, r, seed=3)
    assert a1 == bgp.assign_blocks_covered(counts, r, seed=3)
    for p in r:
        blocks = [b for b, v in a1.items() if v == p]
        assert (counts.loc[blocks].sum() > 0).all()


# ------------------------------------------------------------------ labels
def test_priority_rule_rarest_first_and_majority_otherwise():
    c = cfg(classes=("benign", "a", "b", "c"), held=(), pri=("b",))
    b = make_builder(c, bgp.full_label_space(c))
    df = flows([T0 + i for i in range(6)], ["a", "a", "a", "b", "benign", "benign"])
    assert b.label_window(df)[1] == "b"                 # priority beats majority
    df = flows([T0 + i for i in range(6)], ["a", "a", "c", "benign", "benign", "benign"])
    assert b.label_window(df)[1] == "a"                 # attack majority, benign ignored


def test_remap_heldout_keeps_binary_label_and_records_true_class():
    c = cfg()
    full = bgp.full_label_space(c)
    gs = build_graphs(flows([T0, T0 + 1, T0 + 2], ["benign", "h", "benign"]),
                      make_builder(c, full), 5000, 2500, progress=False)
    gs += build_graphs(flows([T0, T0 + 1], ["a", "benign"]), make_builder(c, full),
                       5000, 2500, progress=False)
    assert [int(g.y_multi) for g in gs] == [3, 1]
    y_full, info = bgp.remap_heldout(gs, c)
    assert y_full.tolist() == [3, 1] and info["heldout_windows"] == 1
    assert int(gs[0].y_multi) == 0 and int(gs[0].y_bin) == 1
    assert int(gs[1].y_multi) == 1
    assert all(g.attack_dist.numel() == 3 for g in gs)


# ------------------------------------------------------------------ building + leakage
def test_parallel_block_build_matches_serial_and_passes_leakage_checks():
    c = cfg()
    df = two_day_flows()
    t0 = int(df[bgp.TIME].min())
    df["_block"] = (df[bgp.TIME] - t0) // 60000
    assignment, info = bgp.split_forward_in_time(df, c, t0)
    graphs, split = bgp.build_all(df, c, assignment, workers=2)
    serial = []
    for p in bgp.PARTS:
        for b in sorted(k for k, v in assignment.items() if v == p):
            blk = df[df["_block"] == b].drop(columns=["_block"])
            gs = build_graphs(blk, make_builder(c, bgp.full_label_space(c)), 5000, 2500,
                              progress=False)
            for g in gs:
                g.window_id = torch.tensor([int(blk[bgp.TIME].min()) + int(g.window_id) * 2500])
            serial.extend(gs)
    assert len(serial) == len(graphs)
    for a, b in zip(serial, graphs):
        assert torch.equal(a.x, b.x) and torch.equal(a.edge_attr, b.edge_attr)
        assert int(a.window_id) == int(b.window_id) and int(a.y_multi) == int(b.y_multi)
    y_full, lab = bgp.remap_heldout(graphs, c)
    res = bgp.leakage_checks(df, assignment, split, graphs, c, t0, info)
    assert res["exact_duplicate_flows_across_partitions"] == 0
    cov = bgp.coverage(graphs, split, y_full, lab["classes_full"], 3)
    assert cov["stream"]["h"] > 0 and cov["train"]["h"] == 0


def test_leakage_check_detects_identical_flow_in_two_partitions():
    c = cfg()
    df = two_day_flows()
    t0 = int(df[bgp.TIME].min())
    df["_block"] = (df[bgp.TIME] - t0) // 60000
    assignment, info = bgp.split_forward_in_time(df, c, t0)
    graphs, split = bgp.build_all(df, c, assignment, workers=1)
    bgp.remap_heldout(graphs, c)
    tr_block = next(b for b, v in assignment.items() if v == "train")
    va_block = next(b for b, v in assignment.items() if v == "val")
    i = df.index[df["_block"] == tr_block][0]
    j = df.index[df["_block"] == va_block][0]
    df2 = df.copy()
    cols = [col for col in df.columns if col != "_block"]
    df2.loc[j, cols] = df.loc[i, cols].values
    with pytest.raises(AssertionError):
        bgp.leakage_checks(df2, assignment, split, graphs, c, t0, info)


# ------------------------------------------------------------------ flow reduction rule
def test_prepare_flows_keeps_small_classes_and_samples_large_ones():
    import prepare_flows
    labels = pd.Series(["benign"] * 5000 + ["big"] * 1000 + ["small"] * 999 + ["tiny"] * 7)
    keep = prepare_flows.select_rows(labels, frac=0.2, min_flows=1000, seed=42)
    kept = labels[keep].value_counts()
    assert kept["benign"] == 1000 and kept["big"] == 200
    assert kept["small"] == 999 and kept["tiny"] == 7
    assert np.array_equal(keep, prepare_flows.select_rows(labels, 0.2, 1000, 42))
    assert not np.array_equal(keep, prepare_flows.select_rows(labels, 0.2, 1000, 7))
