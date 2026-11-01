"""Audit-only analysis of a NetFlow-v3 CSV before any graph building or training.

    python scripts/audit_netflow_v3.py --csv data/raw/<file>.csv \
        --config configs/nf_cicids2018_v3.yaml --out results/nf_cicids2018_v3/dataset_audit/<name>

Reads the CSV in chunks (works for ~20M rows); never writes to data/ and never
changes any dataset. Reports:
  * row count, columns, class distribution, binary Label vs Attack consistency;
  * missing / non-finite / negative values, exact-duplicate rows, invalid IPv4;
  * timestamp coverage (FLOW_START/END_MILLISECONDS), ordering, span;
  * presence and type of every node / edge feature named in the config
    (so that nothing would be silently zero-filled by the graph builder);
  * per-minute class timeline -> attack episodes (contiguous activity separated by
    gaps > --episode-gap-min) and overlap of benign / attack traffic;
  * 5 s / 2.5 s window composition (mixed windows, windows with >= 2 attack
    classes, suppression rates as in the manuscript's Eq. 2) inside 60 s blocks;
  * feasibility of three protocols: forward-in-time, 60 s block-disjoint
    class-covered, and episode-disjoint (per-class episode counts);
  * attacker-IP identity: unique source IPs per attack class and cross-class reuse.
Outputs: summary.json, class_counts.csv, episodes.csv, timeline_minutes.csv,
window_composition.csv, protocol_feasibility.json, fig_timeline.png/pdf.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQUIRED = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL",
            "FLOW_START_MILLISECONDS", "Attack"]
WIN_MS, STRIDE_MS, BLOCK_MS = 5000, 2500, 60000


def is_ipv4(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    shape_ok = s.str.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}").fillna(False)
    parts = s.where(shape_ok, "0.0.0.0").str.split(".", expand=True).astype(int)
    return shape_ok & (parts <= 255).all(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--episode-gap-min", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=0, help="debug only")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    feats = {"statistical": cfg["features"]["statistical"], "behavioral": cfg["features"]["behavioral"],
             "edge": cfg["features"]["edge"]}
    want = sorted(set(sum(feats.values(), [])) | set(REQUIRED) | {"FLOW_END_MILLISECONDS", "Label"})

    header = pd.read_csv(args.csv, nrows=0).columns.tolist()
    present = {c: (c in header) for c in want}
    usecols = [c for c in want if c in header]
    n = 0
    cls = Counter(); lbl_mismatch = 0; label_seen = "Label" in header
    missing = Counter(); nonfinite = Counter(); negative = Counter(); zero = Counter()
    nonnumeric = set(); bad_ip = 0; t_bad = 0; end_before_start = 0
    row_hashes = []; times = []; codes = []; srcs = []; classes = {}
    tmin, tmax, unsorted_pairs, last_t = np.inf, -np.inf, 0, None
    for chunk in pd.read_csv(args.csv, usecols=usecols, chunksize=args.chunksize, low_memory=False):
        if args.max_rows and n >= args.max_rows:
            break
        n += len(chunk)
        att = chunk["Attack"].astype(str).str.strip().str.lower()
        cls.update(att.value_counts().to_dict())
        if label_seen:
            lab = pd.to_numeric(chunk["Label"], errors="coerce")
            lbl_mismatch += int(((att != "benign") != (lab == 1)).sum())
        for c in usecols:
            if c in ("Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"):
                missing[c] += int(chunk[c].isna().sum()); continue
            v = pd.to_numeric(chunk[c], errors="coerce")
            if v.isna().sum() > chunk[c].isna().sum():
                nonnumeric.add(c)
            missing[c] += int(chunk[c].isna().sum())
            vv = v.to_numpy(dtype=float)
            nonfinite[c] += int((~np.isfinite(vv) & ~np.isnan(vv)).sum())
            negative[c] += int((vv < 0).sum()); zero[c] += int((vv == 0).sum())
        bad_ip += int((~is_ipv4(chunk["IPV4_SRC_ADDR"])).sum() + (~is_ipv4(chunk["IPV4_DST_ADDR"])).sum())
        t = pd.to_numeric(chunk["FLOW_START_MILLISECONDS"], errors="coerce").to_numpy(dtype=float)
        t_bad += int((~np.isfinite(t) | (t <= 0)).sum())
        if "FLOW_END_MILLISECONDS" in chunk:
            te = pd.to_numeric(chunk["FLOW_END_MILLISECONDS"], errors="coerce").to_numpy(dtype=float)
            end_before_start += int((te < t).sum())
        if last_t is not None and np.isfinite(t[0]):
            unsorted_pairs += int(t[0] < last_t)
        unsorted_pairs += int(np.sum(np.diff(t[np.isfinite(t)]) < 0)); last_t = t[np.isfinite(t)][-1] if np.isfinite(t).any() else last_t
        ok = np.isfinite(t)
        tmin, tmax = min(tmin, np.nanmin(t)), max(tmax, np.nanmax(t))
        for a in att.unique():
            classes.setdefault(a, len(classes))
        times.append(t[ok].astype(np.int64)); codes.append(att[ok].map(classes).to_numpy(np.int16))
        srcs.append(chunk.loc[ok, "IPV4_SRC_ADDR"].astype(str).to_numpy())
        row_hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
    times = np.concatenate(times); codes = np.concatenate(codes); srcs = np.concatenate(srcs)
    hashes = np.concatenate(row_hashes)
    inv = {v: k for k, v in classes.items()}
    benign = classes.get("benign", -1)
    dup_rows = int(len(hashes) - len(np.unique(hashes)))

    # ---------------- timeline: per-minute class counts and episodes
    t0 = int(times.min())
    minute = (times - t0) // 60000
    tl = pd.DataFrame({"minute": minute, "cls": codes}).value_counts().unstack(fill_value=0)
    tl.columns = [inv[c] for c in tl.columns]
    tl.to_csv(out / "timeline_minutes.csv")
    gap = args.episode_gap_min
    eps = []
    for c in tl.columns:
        act = tl.index[tl[c] > 0].to_numpy()
        if len(act) == 0:
            continue
        cuts = np.flatnonzero(np.diff(act) > gap)
        starts = np.r_[act[0], act[cuts + 1]]; ends = np.r_[act[cuts], act[-1]]
        for k, (s, e) in enumerate(zip(starts, ends)):
            flows = int(tl.loc[s:e, c].sum())
            eps.append({"class": c, "episode": k + 1, "start_hour": s / 60, "end_hour": (e + 1) / 60,
                        "active_minutes": int(((tl.loc[s:e, c]) > 0).sum()), "flows": flows})
    eps = pd.DataFrame(eps); eps.to_csv(out / "episodes.csv", index=False)
    att_cols = [c for c in tl.columns if c != "benign"]
    multi_att_minutes = int(((tl[att_cols] > 0).sum(1) >= 2).sum()) if att_cols else 0
    benign_overlap = {c: float(((tl[c] > 0) & (tl.get("benign", 0) > 0)).sum() / max((tl[c] > 0).sum(), 1))
                      for c in att_cols}

    # ---------------- 5 s / 2.5 s windows inside 60 s blocks
    order = np.argsort(times, kind="stable")
    ts, cs = times[order] - t0, codes[order]
    blk = ts // BLOCK_MS
    rows = []
    # windows start at each block's first flow (as scripts/build_graphs.py does)
    df = pd.DataFrame({"t": ts, "c": cs, "b": blk})
    first = df.groupby("b")["t"].transform("min")
    rel = df["t"] - first
    k1 = rel // STRIDE_MS                          # window starting at k1*stride covers t
    parts_ = []
    for k in (k1, k1 - 1):
        valid = (k >= 0) & ((rel - k * STRIDE_MS) < WIN_MS)
        parts_.append(pd.DataFrame({"b": df["b"][valid], "w": k[valid], "c": df["c"][valid]}))
    wc = pd.concat(parts_).groupby(["b", "w", "c"]).size().rename("n").reset_index()
    piv = wc.pivot_table(index=["b", "w"], columns="c", values="n", fill_value=0)
    piv.columns = [inv[c] for c in piv.columns]
    n_windows = len(piv)
    attack_cols = [c for c in piv.columns if c != "benign"]
    n_att_present = (piv[attack_cols] > 0).sum(1)
    mixed = (piv > 0).sum(1) > 1
    supp = []
    for c in attack_cols:
        m = piv[c] > 0
        if not m.any():
            continue
        others = piv.drop(columns=[c]).max(1); others_att = piv[[x for x in attack_cols if x != c]].max(1) if len(attack_cols) > 1 else 0
        tot = piv.sum(1)
        supp.append({"class": c, "windows": int(m.sum()), "mixed_pct": 100 * float(mixed[m].mean()),
                     "suppressed_pct_incl_benign": 100 * float((piv.loc[m, c] < others[m]).mean()),
                     "suppressed_pct_attack_only": 100 * float((piv.loc[m, c] < (others_att[m] if len(attack_cols) > 1 else 0)).mean()),
                     "mean_share_pct": 100 * float((piv.loc[m, c] / tot[m]).mean())})
    pd.DataFrame(supp).to_csv(out / "window_composition.csv", index=False)
    # window label as in the graph builder: majority among attack classes, else benign
    lab = np.where(n_att_present > 0, piv[attack_cols].idxmax(1), "benign") if attack_cols else np.array(["benign"] * n_windows)
    lab = pd.Series(lab, index=piv.index)

    # ---------------- protocol feasibility (60 s blocks)
    blocks = lab.groupby(level=0).agg(lambda s: Counter(s))
    block_ids = np.array(sorted(blocks.index))
    win_per_block = lab.groupby(level=0).size().reindex(block_ids).to_numpy()
    cum = np.cumsum(win_per_block) / win_per_block.sum()
    cut1, cut2 = block_ids[np.searchsorted(cum, 0.60)], block_ids[np.searchsorted(cum, 0.75)]
    part = np.where(block_ids <= cut1, "train", np.where(block_ids <= cut2, "val", "test"))
    pmap = dict(zip(block_ids, part))
    lab_part = pd.Series([pmap[b] for b in lab.index.get_level_values(0)], index=lab.index)
    fwd = pd.crosstab(lab, lab_part).reindex(columns=["train", "val", "test"], fill_value=0)
    fwd.to_csv(out / "forward_in_time_window_counts.csv")
    blocks_per_class = {c: int(len(set(lab.index.get_level_values(0)[lab.values == c]))) for c in sorted(set(lab))}
    ep_counts = eps.groupby("class")["episode"].max().to_dict() if len(eps) else {}
    feas = {
        "forward_in_time_60_15_25": {"cut_hours": [float(cut1 / 60), float(cut2 / 60)],
                                     "classes_missing_from_train": [c for c in fwd.index if fwd.loc[c, "train"] == 0],
                                     "classes_missing_from_test": [c for c in fwd.index if fwd.loc[c, "test"] == 0],
                                     "classes_unseen_in_train_but_in_test": [c for c in fwd.index if fwd.loc[c, "train"] == 0 and fwd.loc[c, "test"] > 0]},
        "block_disjoint_class_covered": {"blocks_per_class": blocks_per_class,
                                         "classes_with_fewer_than_3_blocks": [c for c, v in blocks_per_class.items() if v < 3]},
        "episode_disjoint": {"episodes_per_class": {k: int(v) for k, v in ep_counts.items()},
                             "classes_with_2plus_episodes": [k for k, v in ep_counts.items() if v >= 2],
                             "classes_with_single_episode": [k for k, v in ep_counts.items() if v < 2]},
    }
    json.dump(feas, open(out / "protocol_feasibility.json", "w"), indent=1, default=float)

    # ---------------- identity
    att_mask = codes != benign
    src_by_cls = defaultdict(set)
    for s, c in zip(srcs[att_mask], codes[att_mask]):
        src_by_cls[inv[c]].add(s)
    ip_cls = defaultdict(set)
    for c, ss in src_by_cls.items():
        for s in ss:
            ip_cls[s].add(c)
    identity = {"unique_attack_src_ips_per_class": {c: len(s) for c, s in src_by_cls.items()},
                "attack_src_ips_used_by_2plus_classes": int(sum(len(v) > 1 for v in ip_cls.values())),
                "attack_src_ips_total": len(ip_cls)}

    summary = {
        "csv": args.csv, "rows": int(n), "columns_in_file": len(header),
        "class_counts": dict(cls.most_common()),
        "binary_label_column": label_seen, "label_attack_mismatches": int(lbl_mismatch),
        "exact_duplicate_rows_in_used_columns": dup_rows,
        "invalid_ipv4_values": int(bad_ip), "bad_or_missing_timestamps": int(t_bad),
        "flow_end_before_start": int(end_before_start), "timestamp_unsorted_adjacent_pairs": int(unsorted_pairs),
        "time_span_hours": float((tmax - tmin) / 3.6e6), "time_start_ms": float(tmin), "time_end_ms": float(tmax),
        "features_present": present, "features_missing": [c for c, p in present.items() if not p],
        "non_numeric_features": sorted(nonnumeric), "missing_values": {k: v for k, v in missing.items() if v},
        "nonfinite_values": {k: v for k, v in nonfinite.items() if v},
        "negative_values": {k: v for k, v in negative.items() if v},
        "zero_fraction": {k: round(v / max(n, 1), 4) for k, v in zero.items()},
        "minutes_with_2plus_attack_classes": multi_att_minutes,
        "attack_minutes_with_benign_present_fraction": benign_overlap,
        "windows_estimated": int(n_windows), "windows_with_2plus_attack_classes": int((n_att_present >= 2).sum()),
        "windows_mixed_pct": 100 * float(mixed.mean()), "blocks_60s_nonempty": int(len(block_ids)),
        "episode_gap_minutes": gap, "identity": identity,
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    pd.Series(cls).sort_values(ascending=False).to_csv(out / "class_counts.csv", header=["flows"])
    timeline_figure(tl, t0, out)
    print(json.dumps({k: summary[k] for k in ("rows", "time_span_hours", "windows_estimated",
                                              "windows_with_2plus_attack_classes", "features_missing")}, default=float))
    print(json.dumps(feas, indent=1, default=float))


def timeline_figure(tl, t0, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = tl.sum().sort_values(ascending=False).index.tolist()
    if "benign" in order:
        order.remove("benign"); order = ["benign"] + order
    hours = tl.index.to_numpy() / 60
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(order) + 1.4))
    for i, c in enumerate(order):
        v = tl[c].to_numpy()
        m = v > 0
        ax.scatter(hours[m], np.full(m.sum(), i), c=np.log10(v[m]), cmap="viridis", s=4, marker="|", vmin=0,
                   vmax=np.log10(tl.to_numpy().max()))
    ax.set_yticks(range(len(order)), order, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel(f"hours since first flow (t0 = {pd.to_datetime(t0, unit='ms')} UTC)")
    ax.set_title("Per-minute flow activity by class (colour = log10 flows per minute)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_timeline.png", dpi=200); fig.savefig(out / "fig_timeline.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
