"""NF-CICIDS2018-v3 (60 s block-stratified split, every class in every partition): five-seed
evaluation of TGN-JointIDS and, with --ablations, of its ablations (w/o ENS, Graph-only, frozen GRU).

    python scripts/analyze_cicids.py --ablations

Inputs: results/nf_cicids2018_v3/tgn_jointids/seed_<s>/ and
results/nf_cicids2018_v3/ablations/{w_o_ens, graph_only, frozen_gru}/seed_<s>/, the graph bundle and
its manifest. Outputs: results/nf_cicids2018_v3/report/ (CSV + LaTeX tables, PNG + PDF figures,
statistics.json). "stream" = the test partition.

Predictions are the frozen causal predictions written at the end of training (checkpoint =
best validation macro-F1; the test partition is never used for any choice). "Raw" = argmax of
the 15-class attack-type head; "gated" = benign whenever the binary head (threshold 0.5) says
benign, otherwise the raw prediction. Raw and gated results are never mixed. Threshold
sensitivity is descriptive only.

The split shares attack sessions (and, for several classes, the single attacker IP) between
training and test blocks: it measures recognition of known attack campaigns, NOT forward-in-time
generalisation.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             precision_recall_curve, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tgn_jointids.config import load_config  # noqa: E402
from tgn_jointids.stats import macro_f1_from_cm  # noqa: E402

SEEDS = [13, 42, 123, 777, 2026]
CFG = ROOT / "configs/nf_cicids2018_v3.yaml"
BASE = ROOT / "results/nf_cicids2018_v3"
OUT = BASE / "report"
NAME = "TGN-JointIDS"   # internal key "A"
MODELS = {"A": (NAME, BASE / "tgn_jointids"),
          "B": ("w/o ENS", BASE / "ablations/w_o_ens"),
          "graph_only": ("Graph-only (no memory)", BASE / "ablations/graph_only"),
          "frozen_gru": ("Frozen GRU", BASE / "ablations/frozen_gru")}
SHORTM = {"A": "TGN-JointIDS", "B": "w/o ENS", "graph_only": "Graph-only", "frozen_gru": "Frozen GRU"}
COMPARISONS = [("A", "graph_only"), ("frozen_gru", "graph_only"), ("A", "frozen_gru"), ("A", "B")]
COLORS = {"graph_only": "#8c8c8c", "frozen_gru": "#5e8ec4", "A": "#1f3a68", "B": "#c8762b"}
SHORT = {"benign": "Benign", "ftp-bruteforce": "FTP-BF", "ssh-bruteforce": "SSH-BF",
         "dos_attacks-goldeneye": "DoS-GoldenEye", "dos_attacks-slowloris": "DoS-Slowloris",
         "dos_attacks-slowhttptest": "DoS-SlowHTTP", "dos_attacks-hulk": "DoS-Hulk",
         "ddos_attacks-loic-http": "DDoS-LOIC-HTTP", "ddos_attack-loic-udp": "DDoS-LOIC-UDP",
         "ddos_attack-hoic": "DDoS-HOIC", "brute_force_-web": "Web-BF", "brute_force_-xss": "XSS",
         "sql_injection": "SQL-Inj", "infilteration": "Infiltration", "bot": "Bot"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def ms(v, d=4):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.{d}f} ± {np.nanstd(v, ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_table(df, name, caption):
    df.to_csv(OUT / f"{name}.csv")
    tex = df.to_latex(escape=True, caption=caption, label=f"tab:{name}")
    (OUT / f"{name}.tex").write_text(tex)


def gated(pred):
    p = pred["pred_multi"].copy()
    p[pred["pred_bin"] == 0] = 0
    return p


# ---------------------------------------------------------------------------- metrics
def binary_metrics(y, p, score):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"accuracy": (tp + tn) / len(y), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
            "average_precision": average_precision_score(y, score),
            "cm": np.array([[tn, fp], [fn, tp]])}


def multi_metrics(y, p, C):
    labels = list(range(C))
    P, R, F, S = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    cm = confusion_matrix(y, p, labels=labels)
    w = S / S.sum()
    return {"accuracy": float(np.mean(y == p)), "macro_f1": float(F.mean()),
            "weighted_f1": float((F * w).sum()), "macro_precision": float(P.mean()),
            "macro_recall": float(R.mean()), "P": P, "R": R, "F": F, "S": S, "cm": cm,
            "fn": S - np.diag(cm), "fp": cm.sum(0) - np.diag(cm)}


# ---------------------------------------------------------------------------- bundle side
def bundle_side(cfg):
    """Graph statistics, partition class counts and the source-IP identity baseline."""
    b = torch.load(ROOT / cfg.paths.graphs, weights_only=False)
    g, sp = b["graphs"], b["split"]
    stats = []
    for part in ("train", "val", "stream"):
        for i in sp[part]:
            stats.append({"partition": part, "label": int(g[i].y_multi), "nodes": int(g[i].num_nodes),
                          "edges": int(g[i].edge_index.shape[1]), "attack_ratio": float(g[i].attack_ratio)})
    stats = pd.DataFrame(stats)
    # source-IP attack-history baseline: an IP that initiated flows in training windows gets that
    # window history; a test window is assigned the majority attack class of the source IP with the
    # strongest attack history (share >= 0.5), else benign. No traffic features are used.
    hist = defaultdict(Counter)

    def sources(gr):
        return set(gr.node_ids[gr.edge_index[0]].tolist())
    for i in sp["train"]:
        y = int(g[i].y_multi)
        for n in sources(g[i]):
            hist[n][y] += 1
    best = {}
    for n, c in hist.items():
        tot = sum(c.values())
        att = {k: v for k, v in c.items() if k != 0}
        if att:
            k, v = max(att.items(), key=lambda kv: kv[1])
            if sum(att.values()) / tot >= 0.5:
                best[n] = (sum(att.values()) / tot, k)
    pred = []
    for i in sp["stream"]:
        cands = [best[n] for n in sources(g[i]) if n in best]
        pred.append(max(cands)[1] if cands else 0)
    wid = np.array([int(g[i].window_id) for i in sp["stream"]])
    t0 = min(int(x.window_id) for x in g)
    del b, g
    return stats, np.array(pred), wid, t0


# ---------------------------------------------------------------------------- cluster bootstrap
def cluster_stats(y, p, ybin, pbin, clusters, C):
    keys, inv = np.unique(clusters, return_inverse=True)
    cm = np.zeros((len(keys), C, C), np.int64)
    np.add.at(cm, (inv, y, p), 1)
    b = np.zeros((len(keys), 2), np.int64)
    ben = ybin == 0
    np.add.at(b, (inv[ben], 0), 1)
    np.add.at(b, (inv[ben], 1), (pbin[ben] == 1).astype(np.int64))
    return cm, b


def paired_bootstrap(sa, sb, fn, n_boot=2000, seed=0):
    S, K = len(sa), sa[0][0].shape[0]

    def delta(w):
        return np.array([fn(np.tensordot(w, sa[s][0], 1), w @ sa[s][1])
                         - fn(np.tensordot(w, sb[s][0], 1), w @ sb[s][1]) for s in range(S)], float)
    full = delta(np.ones(K))
    rng = np.random.default_rng(seed)
    boots = np.array([np.nanmean(delta(np.bincount(rng.integers(0, K, K), minlength=K).astype(float)))
                      for _ in range(n_boot)])
    ok = np.isfinite(boots)
    lo, hi = np.percentile(boots[ok], [2.5, 97.5]) if ok.mean() > 0.95 else (np.nan, np.nan)
    return {"mean_delta": float(np.nanmean(full)), "per_seed": full.round(4).tolist(),
            "seeds_positive": int(np.sum(full > 0)), "ci95": [float(lo), float(hi)],
            "valid_boot_fraction": float(ok.mean())}


# ---------------------------------------------------------------------------- main
def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--latency", default=str(BASE / "latency_idle.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="testing only")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    SEEDS[:] = args.seeds
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(CFG)
    classes = list(cfg.data.classes)
    names = [SHORT[c] for c in classes]
    C = len(classes)
    man = json.load(open((ROOT / cfg.paths.graphs).with_suffix(".manifest.json")))
    models = list(MODELS) if args.ablations else ["A"]

    R = {}
    for m in models:
        for s in SEEDS:
            run = MODELS[m][1] / f"seed_{s}"
            if not (run / "predictions_stream.npz").exists():
                raise SystemExit(f"missing run: {m} seed {s} ({run})")
            st, va = dict(np.load(run / "predictions_stream.npz")), dict(np.load(run / "predictions_val.npz"))
            info = json.load(open(run / "run_info.json"))
            R[(m, s)] = {"st": st, "va": va, "info": info, "hist": json.load(open(run / "history.json")),
                         "raw": multi_metrics(st["y_multi"], st["pred_multi"], C),
                         "gated": multi_metrics(st["y_multi"], gated(st), C),
                         "val": multi_metrics(va["y_multi"], va["pred_multi"], C),
                         "bin": binary_metrics(st["y_bin"], st["pred_bin"], st["prob_attack"]),
                         "val_bin": binary_metrics(va["y_bin"], va["pred_bin"], va["prob_attack"])}
    ref = R[("A", 13)]
    for k, r in R.items():                      # identical samples and protocol for every run
        assert np.array_equal(r["st"]["window_id"], ref["st"]["window_id"]), k
        assert np.array_equal(r["st"]["y_multi"], ref["st"]["y_multi"]), k
        assert np.array_equal(r["va"]["window_id"], ref["va"]["window_id"]), k
        assert r["info"]["split_sha256"] == man["split_sha256"], k
        assert r["info"]["total_updates"] == 7000, k
        t = r["info"]["config"]["training"]
        assert (t["max_updates"], t["eval_every_updates"], t["lr_schedule"]) == (7000, 100, "cosine"), k
    checks = {"split_sha256": man["split_sha256"], "identical_stream_and_val_windows": True,
              "all_runs_7000_updates": True, "runs": len(R),
              "data_csv_sha256": man["data"]["csv_sha256"]}
    lat = json.load(open(args.latency)) if Path(args.latency).exists() else {}

    stats_df, src_base, wid, t0 = bundle_side(cfg)
    y_test = ref["st"]["y_multi"]
    base = multi_metrics(y_test, src_base, C)
    A = [R[("A", s)] for s in SEEDS]

    # ---------------- Table 1: overall (A, five seeds)
    rows = []
    for key, name in (("accuracy", "Binary accuracy"), ("precision", "Binary precision"),
                      ("recall", "Binary recall"), ("f1", "Binary F1"), ("fpr", "Binary FPR"),
                      ("fnr", "Binary FNR"), ("average_precision", "Binary average precision")):
        v = [r["bin"][key] for r in A]
        rows.append({"metric": name, "mean ± sd": ms(v), **{f"seed {s}": round(x, 4) for s, x in zip(SEEDS, v)}})
    for mode in ("raw", "gated"):
        for key, name in (("accuracy", "accuracy"), ("macro_f1", "macro-F1"), ("weighted_f1", "weighted F1"),
                          ("macro_precision", "macro precision"), ("macro_recall", "macro recall")):
            v = [r[mode][key] for r in A]
            rows.append({"metric": f"Multiclass {name} ({mode})", "mean ± sd": ms(v),
                         **{f"seed {s}": round(x, 4) for s, x in zip(SEEDS, v)}})
    v = [r["val"]["macro_f1"] for r in A]
    rows.append({"metric": "Validation macro-F1 (selection)", "mean ± sd": ms(v),
                 **{f"seed {s}": round(x, 4) for s, x in zip(SEEDS, v)}})
    rows.append({"metric": "Source-IP identity baseline macro-F1 (no features)", "mean ± sd": f"{base['macro_f1']:.4f}"})
    rows.append({"metric": "Source-IP identity baseline accuracy", "mean ± sd": f"{base['accuracy']:.4f}"})
    t1 = pd.DataFrame(rows).set_index("metric")
    write_table(t1, "table1_overall", "NF-CICIDS2018-v3 (block-stratified, class-covered split): TGN-JointIDS, "
                "five seeds, test partition.")

    # ---------------- Table 2: per class, raw and gated
    for mode in ("raw", "gated"):
        rows = []
        for i, c in enumerate(classes):
            r = {"class": names[i], "support": int(A[0][mode]["S"][i])}
            for k, nm in (("P", "precision"), ("R", "recall"), ("F", "F1")):
                r[nm] = ms([x[mode][k][i] for x in A], 3)
            r["FN"] = ms([x[mode]["fn"][i] for x in A], 1)
            r["FP"] = ms([x[mode]["fp"][i] for x in A], 1)
            if mode == "raw":
                r["source-IP baseline F1"] = round(float(base["F"][i]), 3)
            rows.append(r)
        write_table(pd.DataFrame(rows).set_index("class"), f"table2_per_class_{mode}",
                    f"NF-CICIDS2018-v3 per-class test results ({mode} predictions), mean ± sd over five seeds.")

    # ---------------- Table 3: minority classes (fewest training windows)
    tr_counts = man["graph_label_counts"]["train"]
    minority = sorted(classes[1:], key=lambda c: tr_counts[c])[:6]
    rows = []
    for c in minority:
        i = classes.index(c)
        cmi = [x["raw"]["cm"] for x in A]
        rows.append({"class": SHORT[c], "train windows": tr_counts[c], "test windows": int(A[0]["raw"]["S"][i]),
                     "recall": ms([x["raw"]["R"][i] for x in A], 3), "precision": ms([x["raw"]["P"][i] for x in A], 3),
                     "F1": ms([x["raw"]["F"][i] for x in A], 3),
                     "FN": ms([x["raw"]["fn"][i] for x in A], 1), "FP": ms([x["raw"]["fp"][i] for x in A], 1),
                     "benign → class": ms([cm[0, i] for cm in cmi], 1),
                     "class → benign": ms([cm[i, 0] for cm in cmi], 1),
                     "main confusion": (lambda row: SHORT[classes[int(np.argmax(row))]] if row.max() > 0 else "—")(
                         np.where(np.arange(C) == i, 0, np.mean(cmi, 0)[i])),
                     "gated recall": ms([x["gated"]["R"][i] for x in A], 3)})
    write_table(pd.DataFrame(rows).set_index("class"), "table3_minority",
                "NF-CICIDS2018-v3 minority classes (fewest training windows): errors, mean ± sd over five seeds.")

    # ---------------- Table 4: compute
    rows = []
    for s, r in zip(SEEDS, A):
        i = r["info"]
        rows.append({"seed": s, "parameters": i["parameters"], "best update": i["best_update"],
                     "training time (min)": round(i["train_seconds_total"] / 60, 1),
                     "peak GPU memory, training (MB)": round(i["peak_gpu_mb_training"], 0),
                     **({"latency mean (ms)": round(lat[str(s)]["mean_ms"], 3),
                         "latency median (ms)": round(lat[str(s)]["median_ms"], 3),
                         "latency P95 (ms)": round(lat[str(s)]["p95_ms"], 3),
                         "latency P99 (ms)": round(lat[str(s)]["p99_ms"], 3),
                         "peak GPU memory, inference (MB)": round(lat[str(s)]["peak_gpu_mb"], 1)}
                        if str(s) in lat else {})})
    t4 = pd.DataFrame(rows).set_index("seed")
    t4.loc["mean ± sd"] = [ms(t4[c].astype(float), 2) for c in t4.columns]
    write_table(t4, "table4_compute", "NF-CICIDS2018-v3 compute: training on one RTX 4090 (4 runs in parallel for "
                "seeds 42-2026, 2 for seed 13); latency per 5 s window, causal, one window per pass, idle GPU.")

    # ---------------- Table 5: dataset and graph statistics
    flows = pd.read_csv(BASE / "dataset/flow_counts_by_partition.csv", index_col=0)
    graphs = pd.DataFrame(man["graph_label_counts"])[["train", "val", "stream"]]
    dm = json.load(open(ROOT / "data/raw/NF-CICIDS2018-v3_common20_rare100.csv.manifest.json"))
    t5 = pd.DataFrame({"official flows": {c: dm["per_class"][c]["before"] for c in classes},
                       "used flows": {c: dm["per_class"][c]["after"] for c in classes}})
    t5["kept %"] = (100 * t5["used flows"] / t5["official flows"]).round(1)
    for p in ("train", "val", "stream"):
        t5[f"flows {p}"] = flows[p].reindex(classes).fillna(0).astype(int)
    for p in ("train", "val", "stream"):
        t5[f"windows {p}"] = graphs[p].reindex(classes).astype(int)
    t5.index = names
    t5.loc["Total"] = t5.sum(numeric_only=True)
    t5.loc["Total", "kept %"] = round(100 * t5.loc["Total", "used flows"] / t5.loc["Total", "official flows"], 1)
    write_table(t5, "table5_dataset", "NF-CICIDS2018-v3 flows (official vs used, NF-ToN-IoT-v3 reduction rule) "
                "and window graphs per partition (block-stratified split).")
    gs = stats_df.groupby("partition")[["nodes", "edges"]].describe(percentiles=[0.5, 0.95]).round(1)
    gs.columns = [f"{a} {b}" for a, b in gs.columns]
    write_table(gs.reindex(["train", "val", "stream"]), "table6_graph_statistics",
                "NF-CICIDS2018-v3 window-graph size (nodes = IPs, edges = flows) per partition.")

    # ---------------- ablations
    stats = {"checks": checks, "source_ip_baseline": {"macro_f1": base["macro_f1"], "accuracy": base["accuracy"],
                                                      "per_class_f1": dict(zip(classes, base["F"].round(4).tolist()))},
             "minority_classes": minority}
    if args.ablations:
        rows = []
        for m in MODELS:
            rr = [R[(m, s)] for s in SEEDS]
            row = {"model": MODELS[m][0]}
            for key, nm in (("macro_f1", "macro-F1 (raw)"), ("accuracy", "accuracy (raw)"),
                            ("weighted_f1", "weighted F1 (raw)")):
                row[nm] = ms([x["raw"][key] for x in rr])
            row["macro-F1 (gated)"] = ms([x["gated"]["macro_f1"] for x in rr])
            for key, nm in (("f1", "binary F1"), ("fpr", "binary FPR"), ("fnr", "binary FNR")):
                row[nm] = ms([x["bin"][key] for x in rr])
            for c in minority[:4]:
                row[f"{SHORT[c]} F1"] = ms([x["raw"]["F"][classes.index(c)] for x in rr], 3)
            row["val macro-F1"] = ms([x["val"]["macro_f1"] for x in rr])
            row["training time (min)"] = ms([x["info"]["train_seconds_total"] / 60 for x in rr], 1)
            rows.append(row)
        write_table(pd.DataFrame(rows).set_index("model"), "table7_ablations",
                    "NF-CICIDS2018-v3 ablations of TGN-JointIDS, five seeds, identical protocol (only the named switch differs).")
        fns = {"macro_f1": lambda cm, b: macro_f1_from_cm(cm, labels=np.arange(C)),
               "binary_fpr": lambda cm, b: b[1] / b[0] if b[0] else np.nan}
        for c in minority[:4]:
            i = classes.index(c)
            fns[f"{c}_f1"] = (lambda i: lambda cm, b: 2 * cm[i, i] / (cm[i].sum() + cm[:, i].sum())
                              if (cm[i].sum() + cm[:, i].sum()) else np.nan)(i)
        paired, rows = {}, []
        for cmin in (10, 60):
            cl = (wid - t0) // (cmin * 60000)
            S = {k: cluster_stats(r["st"]["y_multi"], r["st"]["pred_multi"], r["st"]["y_bin"],
                                  r["st"]["pred_bin"], cl, C) for k, r in R.items()}
            res = {}
            for a, b in COMPARISONS:
                res[f"{a} - {b}"] = {k: paired_bootstrap([S[(a, s)] for s in SEEDS], [S[(b, s)] for s in SEEDS],
                                                         fn, n_boot=args.n_boot) for k, fn in fns.items()}
                for k, v in res[f"{a} - {b}"].items():
                    rows.append({"clusters": f"{cmin} min", "comparison": f"{SHORTM[a]} − {SHORTM[b]}", "metric": k,
                                 "mean Δ": round(v["mean_delta"], 4),
                                 "95% CI": f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]",
                                 "seeds > 0": f"{v['seeds_positive']}/{len(SEEDS)}"})
            paired[f"{cmin}min"] = res
        write_table(pd.DataFrame(rows).set_index(["clusters", "comparison", "metric"]), "table8_paired",
                    "Paired cluster bootstrap of seed-mean differences (2,000 resamples of time clusters).")
        stats["paired"] = paired

    # ---------------- figures
    def cm_plot(cm, labels, name, title, norm=True):
        v = cm / np.maximum(cm.sum(1, keepdims=True), 1) if norm else cm
        n = len(labels)
        fig, ax = plt.subplots(figsize=(0.5 * n + 2.5, 0.45 * n + 2) if n > 2 else (3.2, 2.8))
        im = ax.imshow(v, cmap="Blues", vmin=0, vmax=1 if norm else None)
        ax.set_xticks(range(n), labels, rotation=60 if n > 2 else 0, ha="right" if n > 2 else "center", fontsize=7)
        ax.set_yticks(range(n), labels, fontsize=7)
        for i in range(n):
            for j in range(n):
                if v[i, j] >= 0.005 or (n == 2):
                    ax.text(j, i, f"{v[i, j]:.2f}" if norm else f"{v[i, j]:.0f}", ha="center", va="center",
                            fontsize=5.5 if n > 2 else 8, color="white" if v[i, j] > 0.5 * v.max() else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, fraction=0.045)
        save(fig, name)

    cm_plot(np.mean([r["raw"]["cm"] for r in A], 0), names, "fig1_confusion_15class_raw",
            f"TGN-JointIDS, attack-type head (raw), mean of {len(SEEDS)} seed(s), row-normalised")
    cm_plot(np.mean([r["gated"]["cm"] for r in A], 0), names, "fig1b_confusion_15class_gated",
            f"TGN-JointIDS, binary-gated predictions, mean of {len(SEEDS)} seed(s), row-normalised")
    cm_plot(np.mean([r["bin"]["cm"] for r in A], 0), ["Benign", "Attack"], "fig2_confusion_binary",
            f"TGN-JointIDS binary head ({len(SEEDS)}-seed mean)")
    cm_plot(np.mean([r["bin"]["cm"] for r in A], 0), ["Benign", "Attack"], "fig2b_confusion_binary_counts",
            f"TGN-JointIDS binary head, counts ({len(SEEDS)}-seed mean)", norm=False)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    x = np.arange(C)
    for ax, k, lab in ((axes[0], "F", "F1"), (axes[1], "R", "Recall")):
        vals = np.array([r["raw"][k] for r in A])
        ax.bar(x, vals.mean(0), 0.6, yerr=vals.std(0, ddof=1), color=COLORS["A"], capsize=2,
               error_kw={"lw": 0.8}, label=f"TGN-JointIDS (raw; mean ± sd, dots = {len(SEEDS)} seeds)")
        for row in vals:
            ax.plot(x, row, ".", color="#e0a030", ms=3)
        if k == "F":
            ax.plot(x, base["F"], "_", color="#c0392b", ms=10, mew=1.5, label="Source-IP identity baseline")
            ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        ax.set_ylabel(f"{lab} (test)")
        ax.set_ylim(0, 1.05)
    axes[1].set_xticks(x, names, rotation=60, ha="right", fontsize=7)
    save(fig, "fig3_per_class_f1_recall")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for s, r in zip(SEEDS, A):
        pr, rc, _ = precision_recall_curve(r["st"]["y_bin"], r["st"]["prob_attack"])
        axes[0].plot(rc, pr, lw=1, label=f"seed {s} (AP {r['bin']['average_precision']:.3f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Binary head, precision-recall (test)", fontsize=9)
    axes[0].legend(fontsize=6.5)
    th = np.linspace(0.05, 0.95, 19)
    curves = {k: [] for k in ("precision", "recall", "f1", "fpr")}
    for r in A:
        y, sc = r["st"]["y_bin"], r["st"]["prob_attack"]
        for k in curves:
            curves[k].append([binary_metrics(y, (sc >= t).astype(int), sc)[k] for t in th])
    for k, col in zip(curves, ("#1f3a68", "#2e8b57", "#c8762b", "#c0392b")):
        a = np.array(curves[k])
        axes[1].plot(th, a.mean(0), color=col, label=k.upper() if k == "fpr" else k.capitalize())
        axes[1].fill_between(th, a.min(0), a.max(0), color=col, alpha=0.2)
    axes[1].axvline(0.5, ls="--", color="k", lw=0.8)
    axes[1].set_xlabel("Decision threshold (0.5 used; others descriptive only)")
    axes[1].set_title(f"Threshold sensitivity (test, {len(SEEDS)} seed(s): mean, min-max)", fontsize=9)
    axes[1].legend(fontsize=7)
    save(fig, "fig4_pr_and_threshold")
    thr = pd.DataFrame({f"{k} (mean)": np.array(v).mean(0).round(4) for k, v in curves.items()}, index=th.round(2))
    thr.index.name = "threshold"
    write_table(thr, "table9_threshold_sensitivity", "TGN-JointIDS binary head: threshold sensitivity on the test "
                "partition (descriptive only; 0.5 fixed a priori).")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for s, r in zip(SEEDS, A):
        h = r["hist"]
        axes[0].plot(h["update"], h["val_macro_f1"], lw=1, label=f"seed {s}")
        axes[1].plot(h["update"], h["val_loss"], lw=1)
        axes[0].plot(r["info"]["best_update"], max(h["val_macro_f1"]), "k*", ms=5)
    axes[0].set_ylabel("Validation macro-F1")
    axes[1].set_ylabel("Validation loss")
    for ax in axes:
        ax.set_xlabel("Optimizer update")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Convergence (★ = selected checkpoint)", fontsize=9)
    save(fig, "fig5_convergence")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    t = t5.drop(index="Total")
    axes[0].barh(t.index, t["official flows"], color="#bbbbbb", label="official")
    axes[0].barh(t.index, t["used flows"], color=COLORS["A"], label="used")
    axes[0].set_xscale("log")
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].set_xlabel("Flows (log)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Flows per class", fontsize=9)
    w = t[["windows train", "windows val", "windows stream"]]
    yy = np.arange(len(w))
    for k, (col, cc) in enumerate(zip(w.columns, ("#1f3a68", "#5e8ec4", "#c8762b"))):
        axes[1].barh(yy + (k - 1) * 0.27, w[col], 0.27, color=cc, label=col.split()[1])
    axes[1].set_xscale("log")
    axes[1].set_ylim(len(w) - 0.5, -0.5)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Window graphs (log)")
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    axes[1].set_title("Windows per class and partition", fontsize=9)
    for part, cc in zip(("train", "val", "stream"), ("#1f3a68", "#5e8ec4", "#c8762b")):
        d = stats_df[stats_df.partition == part]
        axes[2].hist(np.log10(d["edges"]), bins=60, histtype="step", color=cc, label=part, density=True)
    axes[2].set_xlabel("log10 edges (flows) per window graph")
    axes[2].set_ylabel("Density")
    axes[2].legend(fontsize=7)
    axes[2].set_title("Graph size", fontsize=9)
    save(fig, "fig6_dataset_and_graphs")

    if args.ablations:
        order = ["graph_only", "frozen_gru", "A", "B"]
        lab = {"graph_only": "Graph-only", "frozen_gru": "Frozen GRU", "A": "TGN-JointIDS", "B": "w/o ENS"}
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 2.2, 1.3], "wspace": 0.45})
        for k, m in enumerate(order):
            v = [R[(m, s)]["raw"]["macro_f1"] for s in SEEDS]
            axes[0].bar(k, np.mean(v), 0.65, color=COLORS[m], yerr=np.std(v, ddof=1), capsize=3)
            axes[0].plot([k] * len(v), v, "k.", ms=3)
        axes[0].set_xticks(range(4), [lab[m] for m in order], rotation=30, ha="right")
        lo = min(R[(m, s)]["raw"]["macro_f1"] for m in order for s in SEEDS)
        axes[0].set_ylim(max(0, lo - 0.03), min(1, lo + 0.07))
        axes[0].set_ylabel("Test macro-F1 (15 classes)")
        axes[0].set_title("(a) Overall", fontsize=9)
        hard = ["brute_force_-web", "brute_force_-xss", "sql_injection", "ddos_attacks-loic-http",
                "ddos_attack-loic-udp", "infilteration"]
        x = np.arange(len(hard))
        for k, m in enumerate(order):
            v = np.array([[R[(m, s)]["raw"]["F"][classes.index(c)] for c in hard] for s in SEEDS])
            axes[1].bar(x + (k - 1.5) * 0.2, v.mean(0), 0.2, yerr=v.std(0, ddof=1), color=COLORS[m],
                        capsize=1.5, error_kw={"lw": 0.6}, label=lab[m])
        axes[1].set_xticks(x, [SHORT[c] for c in hard], rotation=30, ha="right")
        axes[1].set_ylabel("Test F1")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
        axes[1].set_title(f"(b) Difficult classes (mean ± sd, {len(SEEDS)} seeds)", fontsize=9)
        pr = paired["60min"]
        rows = [(f"{lab[a]}\nvs {lab[b]}", pr[f"{a} - {b}"]["macro_f1"]) for a, b in COMPARISONS]
        for k, (nm, v) in enumerate(rows):
            lo_, hi_ = v["ci95"]
            axes[2].errorbar(v["mean_delta"], k, xerr=[[v["mean_delta"] - lo_], [hi_ - v["mean_delta"]]],
                             fmt="o", color="#1f3a68", capsize=3)
            axes[2].text(hi_ + 0.002, k, f"{v['seeds_positive']}/{len(SEEDS)}", va="center", fontsize=7)
        axes[2].axvline(0, color="k", lw=0.8, ls="--")
        axes[2].set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Δ macro-F1, 95% CI\n(60-min cluster bootstrap)")
        axes[2].set_title("(c) Paired differences (seeds > 0)", fontsize=9)
        save(fig, "fig7_ablations")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    stats["per_seed_A"] = {s: {"binary": {k: v for k, v in R[("A", s)]["bin"].items() if k != "cm"},
                               "raw": {k: R[("A", s)]["raw"][k] for k in ("accuracy", "macro_f1", "weighted_f1")},
                               "gated": {k: R[("A", s)]["gated"][k] for k in ("accuracy", "macro_f1", "weighted_f1")},
                               "val_macro_f1": R[("A", s)]["val"]["macro_f1"]} for s in SEEDS}
    json.dump(clean(stats), open(OUT / "statistics.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t1.to_string())
    for f in sorted(OUT.glob("table*.csv")):
        print("wrote", f.name)


if __name__ == "__main__":
    main()
