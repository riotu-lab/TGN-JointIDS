"""NF-ToN-IoT-v3 ablations of TGN-JointIDS: tables, statistics and figures.

    python scripts/analyze_ton_ablations.py

Models (identical protocol: 7,000 updates, cosine LR, causal validation every 100 updates,
selection on validation macro-F1, frozen causal test evaluation; seeds 13, 42, 123, 777, 2026):
  A           TGN-JointIDS: trainable GRU + ENS + focal  (results/nf_ton_iot_v3/tgn_jointids)
  B           w/o ENS: focal loss without ENS weights    (results/nf_ton_iot_v3/ablations/w_o_ens)
  graph_only  Graph-only: no memory                      (results/nf_ton_iot_v3/ablations/graph_only)
  frozen_gru  memory with the GRU fixed at init          (results/nf_ton_iot_v3/ablations/frozen_gru)
Raw 10-class predictions (attack-type head argmax) and binary-gated predictions (benign
whenever the binary head says benign) are reported separately and never mixed. "stream" = test.
Writes results/nf_ton_iot_v3/ablations/report/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _ton_metrics import CLASSES, low_purity_mask, metrics  # noqa: E402
from tgn_jointids.stats import macro_f1_from_cm  # noqa: E402

SEEDS = [13, 42, 123, 777, 2026]
CC = ROOT / "results/nf_ton_iot_v3/ablations"
OUT = CC / "report"
MODELS = {
    "graph_only": ("Graph-only (no memory)", CC / "graph_only"),
    "frozen_gru": ("Frozen GRU", CC / "frozen_gru"),
    "A": ("TGN-JointIDS", ROOT / "results/nf_ton_iot_v3/tgn_jointids"),
    "B": ("w/o ENS", CC / "w_o_ens"),
}
COLORS = {"graph_only": "#8c8c8c", "frozen_gru": "#5e8ec4", "A": "#1f3a68", "B": "#c8762b"}
LABELS = ["Benign", "DDoS", "XSS", "Password", "Scanning", "Backdoor", "Injection", "DoS", "MITM", "Ransomware"]
CRIT = ["mitm", "ransomware", "scanning"]
COMPARISONS = [("A", "B"), ("A", "graph_only"), ("A", "frozen_gru"), ("frozen_gru", "graph_only")]
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.{d}f} ± {np.nanstd(v, ddof=1):.{d}f}"


def gated(pred):
    p = pred["pred_multi"].copy()
    p[pred["pred_bin"] == 0] = 0
    return {**pred, "pred_multi": p}


# ------------------------------------------------------------------ cluster bootstrap (generic)
def cluster_stats(pred, clusters):
    """Per-cluster sufficient statistics: 10x10 confusion + binary counts."""
    keys, inv = np.unique(clusters, return_inverse=True)
    K, C = len(keys), len(CLASSES)
    cm = np.zeros((K, C, C), np.int64)
    np.add.at(cm, (inv, pred["y_multi"], pred["pred_multi"]), 1)
    benign = pred["y_bin"] == 0
    b = np.zeros((K, 2), np.int64)       # benign windows, benign flagged by binary head
    np.add.at(b, (inv[benign], 0), 1)
    np.add.at(b, (inv[benign], 1), (pred["pred_bin"][benign] == 1).astype(np.int64))
    return cm, b


def stat_fns():
    def rec(c):
        i = CLASSES.index(c)
        return lambda cm, b: cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan

    def f1(c):
        i = CLASSES.index(c)
        return lambda cm, b: (2 * cm[i, i] / (cm[i].sum() + cm[:, i].sum())
                              if (cm[i].sum() + cm[:, i].sum()) else np.nan)
    fns = {"macro_f1": lambda cm, b: macro_f1_from_cm(cm),
           "binary_benign_fpr": lambda cm, b: b[1] / b[0] if b[0] else np.nan}
    for c in CRIT:
        fns[f"{c}_recall"], fns[f"{c}_f1"] = rec(c), f1(c)
    return fns


def paired_bootstrap(sa, sb, fn, n_boot=2000, seed=0):
    """sa, sb: lists over seeds of (cm[K,C,C], b[K,2]) on the same clusters."""
    S, K = len(sa), sa[0][0].shape[0]

    def delta(w):
        d = []
        for s in range(S):
            ca, ba = np.tensordot(w, sa[s][0], 1), w @ sa[s][1]
            cb, bb = np.tensordot(w, sb[s][0], 1), w @ sb[s][1]
            d.append(fn(ca, ba) - fn(cb, bb))
        return np.array(d, float)
    full = delta(np.ones(K))
    rng = np.random.default_rng(seed)
    boots = np.array([np.nanmean(delta(np.bincount(rng.integers(0, K, K), minlength=K).astype(float)))
                      for _ in range(n_boot)])
    ok = np.isfinite(boots)
    lo, hi = (np.percentile(boots[ok], [2.5, 97.5]) if ok.mean() > 0.95 else (np.nan, np.nan))
    return {"mean_delta": float(np.nanmean(full)), "per_seed": full.round(4).tolist(),
            "seeds_positive": int(np.sum(full > 0)), "ci95": [float(lo), float(hi)],
            "valid_boot_fraction": float(ok.mean())}


# ------------------------------------------------------------------ main
def dataset_table():
    """T1: flows per class (derived file, from the reduction-rule record) and window graphs per
    partition (graph-bundle manifest)."""
    rule = json.load(open(ROOT / "results/nf_cicids2018_v3/downsampling_rule.json"))
    man = json.load(open(ROOT / "data/processed/nf_ton_iot_v3.manifest.json"))["graph_stats"]
    flows = {c: v[1] for c, v in rule["per_class_official_derived_ratio"].items()}
    t = pd.DataFrame({"flows": [flows[c] for c in CLASSES]}, index=pd.Index(CLASSES, name="class"))
    t["flow_share_pct"] = (100 * t["flows"] / t["flows"].sum()).round(2)
    for p in ("train", "val", "stream"):
        t[f"{p}_windows"] = [man[p]["class_counts"][c] for c in CLASSES]
    t.loc["total"] = t.sum()
    t.loc["total", "flow_share_pct"] = 100.0
    return t.astype({c: int for c in t.columns if c != "flow_share_pct"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_table().to_csv(OUT / "table1_dataset.csv")
    low_mask, wid_ref, t0 = low_purity_mask("data/processed/nf_ton_iot_v3.pt", "stream")
    R, missing = {}, []
    for m, (_, path) in MODELS.items():
        for s in SEEDS:
            run = path / f"seed_{s}"
            if not (run / "predictions_stream.npz").exists():
                missing.append(f"{m} seed {s}")
                continue
            st, va = dict(np.load(run / "predictions_stream.npz")), dict(np.load(run / "predictions_val.npz"))
            info = json.load(open(run / "run_info.json"))
            R[(m, s)] = {"st": st, "va": va, "info": info, "hist": json.load(open(run / "history.json")),
                         "raw": metrics(st, low_mask), "gated": metrics(gated(st), low_mask),
                         "val": metrics(va)}
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    # identical evaluation samples + protocol checks
    ref = R[("A", 13)]
    for k, r in R.items():
        assert np.array_equal(r["st"]["window_id"], ref["st"]["window_id"]), k
        assert np.array_equal(r["st"]["y_multi"], ref["st"]["y_multi"]), k
        assert np.array_equal(r["va"]["window_id"], ref["va"]["window_id"]), k
        assert r["info"]["split_sha256"] == ref["info"]["split_sha256"], k
        assert r["info"]["total_updates"] == 7000, k
        c = r["info"]["config"]["training"]
        assert (c["max_updates"], c["eval_every_updates"], c["lr_schedule"]) == (7000, 100, "cosine"), k
    checks = {"identical_stream_windows": True, "identical_val_windows": True, "identical_split_sha": True,
              "all_7000_updates": True, "n_runs": len(R)}
    lat = json.load(open(CC / "latency_idle.json")) if (CC / "latency_idle.json").exists() else {}
    tables = build_tables(R, lat)
    stats = statistics(R, wid_ref, t0)
    for name, t in tables.items():
        t.to_csv(OUT / f"{name}.csv")
    with open(OUT / "tables.md", "w") as f:
        for name, t in tables.items():
            f.write(f"## {name}\n\n{t.to_markdown()}\n\n")
    json.dump({"checks": checks, "paired": stats}, open(OUT / "statistics.json", "w"), indent=1, default=float)
    figures(R)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    for name, t in tables.items():
        print(f"== {name}\n{t.to_string()}\n")
    print(json.dumps(stats, indent=1, default=float))


def build_tables(R, lat):
    T = {}
    # T2 overall (A): raw and binary-gated attack-type metrics + binary head
    rows = []
    for key, name in [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
                      ("macro_recall", "Macro recall"), ("macro_f1", "Macro-F1"), ("weighted_f1", "Weighted F1"),
                      ("benign_misclass_multiclass_head", "Benign misclassified (10-class)"),
                      ("low_identity_macro_f1", "Low-identity macro-F1")]:
        rows.append({"metric": name,
                     "validation (raw)": ms([R[("A", s)]["val"][key] for s in SEEDS]) if key != "low_identity_macro_f1" else "",
                     "stream raw 10-class": ms([R[("A", s)]["raw"][key] for s in SEEDS]),
                     "stream binary-gated": ms([R[("A", s)]["gated"][key] for s in SEEDS]),
                     **{f"raw s{s}": round(R[("A", s)]["raw"][key], 4) for s in SEEDS}})
    for key, name in [("binary_f1", "Binary F1 (binary head)"), ("benign_fpr_binary_head", "Benign FPR (binary head)")]:
        rows.append({"metric": name, "validation (raw)": ms([R[("A", s)]["val"][key] for s in SEEDS]),
                     "stream raw 10-class": ms([R[("A", s)]["raw"][key] for s in SEEDS]), "stream binary-gated": "",
                     **{f"raw s{s}": round(R[("A", s)]["raw"][key], 4) for s in SEEDS}})
    aps, fnrs, accs = [], [], []
    for s in SEEDS:
        st = R[("A", s)]["st"]
        aps.append(average_precision_score(st["y_bin"], st["prob_attack"]))
        att = st["y_bin"] == 1
        fnrs.append(np.mean(st["pred_bin"][att] == 0)); accs.append(np.mean(st["pred_bin"] == st["y_bin"]))
    for name, v in (("Binary accuracy (binary head)", accs), ("Attack FNR (binary head)", fnrs),
                    ("Binary average precision", aps)):
        rows.append({"metric": name, "validation (raw)": "", "stream raw 10-class": ms(v, 4),
                     "stream binary-gated": "", **{f"raw s{s}": round(x, 4) for s, x in zip(SEEDS, v)}})
    T["table2_overall_A"] = pd.DataFrame(rows).set_index("metric")
    # T3 10-class P/R/F1 (A, raw stream)
    rows = []
    for c, lab in zip(CLASSES, LABELS):
        pc = [R[("A", s)]["raw"]["per_class"][c] for s in SEEDS]
        rows.append({"class": lab, "precision": ms([p["precision"] for p in pc]),
                     "recall": ms([p["recall"] for p in pc]), "F1": ms([p["f1"] for p in pc]),
                     "support": pc[0]["support"],
                     "F1 per seed": " / ".join(f"{p['f1']:.3f}" for p in pc)})
    T["table3_per_class_A"] = pd.DataFrame(rows).set_index("class")
    # T4 minority + false alarms (A, raw and gated)
    rows = []
    for c in ["mitm", "ransomware", "scanning", "injection"]:
        for mode in ("raw", "gated"):
            pc = [R[("A", s)][mode]["per_class"][c] for s in SEEDS]
            rows.append({"class": c, "predictions": "raw 10-class" if mode == "raw" else "binary-gated",
                         "precision": ms([p["precision"] for p in pc]), "recall": ms([p["recall"] for p in pc]),
                         "F1": ms([p["f1"] for p in pc]), "FN": ms([p["fn"] for p in pc], 1),
                         "FP": ms([p["fp"] for p in pc], 1), "benign→class": ms([p["benign_as"] for p in pc], 1),
                         "support": pc[0]["support"]})
    T["table4_minority_A"] = pd.DataFrame(rows).set_index(["class", "predictions"])
    # T5 memory ablation and T6 ENS ablation (raw stream, mean ± sd)
    def model_rows(models):
        rows = []
        for m in models:
            r = [R[(m, s)] for s in SEEDS]
            row = {"model": MODELS[m][0], "val macro-F1": ms([x["val"]["macro_f1"] for x in r]),
                   "stream macro-F1": ms([x["raw"]["macro_f1"] for x in r]),
                   "macro precision": ms([x["raw"]["macro_precision"] for x in r]),
                   "macro recall": ms([x["raw"]["macro_recall"] for x in r]),
                   "binary FPR": ms([x["raw"]["benign_fpr_binary_head"] for x in r]),
                   "benign misclassified (10-class)": ms([x["raw"]["benign_misclass_multiclass_head"] for x in r]),
                   "low-identity macro-F1": ms([x["raw"]["low_identity_macro_f1"] for x in r])}
            for c in CRIT:
                row[f"{c} P"] = ms([x["raw"]["per_class"][c]["precision"] for x in r])
                row[f"{c} R"] = ms([x["raw"]["per_class"][c]["recall"] for x in r])
                row[f"{c} F1"] = ms([x["raw"]["per_class"][c]["f1"] for x in r])
            row["stream macro-F1 per seed"] = " / ".join(f"{x['raw']['macro_f1']:.3f}" for x in r)
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")
    T["table5_memory_ablation"] = model_rows(["graph_only", "frozen_gru", "A"])
    T["table6_ens_ablation"] = model_rows(["B", "A"])
    # T7 efficiency
    rows = []
    for m in MODELS:
        infos = [R[(m, s)]["info"] for s in SEEDS]
        L = [lat.get(f"{m}_{s}", {}) for s in SEEDS]
        rows.append({"model": MODELS[m][0], "total params": infos[0]["parameters"] if m != "frozen_gru" else 302224,
                     "trainable params": infos[0]["parameters"],
                     "train time (min)": ms([i["train_seconds_total"] / 60 for i in infos], 1),
                     "peak GPU train (MB)": ms([i["peak_gpu_mb_training"] for i in infos], 0),
                     "latency mean (ms)": ms([l.get("mean_ms", np.nan) for l in L], 2),
                     "latency median (ms)": ms([l.get("median_ms", np.nan) for l in L], 2),
                     "latency P95 (ms)": ms([l.get("p95_ms", np.nan) for l in L], 2),
                     "peak GPU inference (MB)": ms([l.get("peak_gpu_mb", np.nan) for l in L], 1),
                     "best update": ms([i["best_update"] for i in infos], 0)})
    T["table7_efficiency"] = pd.DataFrame(rows).set_index("model")
    # T8 per-class one-vs-rest TPR/FPR (A, raw 10-class, all stream windows)
    rows = []
    for c, lab in zip(CLASSES, LABELS):
        i = CLASSES.index(c)
        tpr, fpr = [], []
        for s in SEEDS:
            y, p = R[("A", s)]["st"]["y_multi"], R[("A", s)]["st"]["pred_multi"]
            tpr.append(np.mean(p[y == i] == i)); fpr.append(np.mean(p[y != i] == i))
        rows.append({"class": lab, "TPR": ms(tpr), "FPR (%)": ms(np.array(fpr) * 100, 3)})
    T["table8_ovr_tpr_fpr_A"] = pd.DataFrame(rows).set_index("class")
    return T


def statistics(R, wid, t0):
    fns = stat_fns()
    out = {}
    for cm_min in (10, 60):
        clusters = (wid - t0) // (cm_min * 60000)
        S = {(m, s): cluster_stats(R[(m, s)]["st"], clusters) for m in MODELS for s in SEEDS}
        res = {}
        for a, b in COMPARISONS:
            res[f"{a} - {b}"] = {k: paired_bootstrap([S[(a, s)] for s in SEEDS], [S[(b, s)] for s in SEEDS], fn)
                                 for k, fn in fns.items()}
        out[f"{cm_min}min_clusters"] = res
    return out


# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figures(R):
    A = [R[("A", s)] for s in SEEDS]
    # F1 binary confusion matrix (pooled over seeds, row-normalised)
    cm = np.zeros((2, 2))
    for r in A:
        for t in (0, 1):
            for p in (0, 1):
                cm[t, p] += np.sum((r["st"]["y_bin"] == t) & (r["st"]["pred_bin"] == p))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for t in range(2):
        for p in range(2):
            ax.text(p, t, f"{cmn[t, p]:.3f}\n({cm[t, p] / 5:.0f})", ha="center", va="center",
                    color="white" if cmn[t, p] > 0.5 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["Benign", "Attack"]); ax.set_yticks([0, 1], ["Benign", "Attack"])
    ax.set_xlabel("Predicted (binary head)"); ax.set_ylabel("True")
    ax.set_title("Binary confusion matrix, TGN-JointIDS\n(row-normalised; mean count per seed)", fontsize=9)
    save(fig, "fig1_binary_confusion")
    # F2 10-class confusion (mean over seeds, row-normalised)
    cm = np.mean([np.array(r["raw"]["cm"]) for r in A], axis=0)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cmn[i, j] >= 0.005:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(10), LABELS, rotation=45, ha="right"); ax.set_yticks(range(10), LABELS)
    ax.set_xlabel("Predicted (raw 10-class head)"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised rate")
    ax.set_title("10-class confusion matrix, TGN-JointIDS (mean over five seeds)", fontsize=9)
    save(fig, "fig2_multiclass_confusion")
    # F3 memory ablation
    models = ["graph_only", "frozen_gru", "A"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for j, m in enumerate(models):
        v = [R[(m, s)]["raw"]["macro_f1"] for s in SEEDS]
        axes[0].scatter(np.full(5, j) + np.linspace(-0.1, 0.1, 5), v, color=COLORS[m], s=18, zorder=3)
        axes[0].errorbar(j + 0.25, np.mean(v), yerr=np.std(v, ddof=1), fmt="o", color="black", capsize=3, ms=4)
    for s in SEEDS:
        axes[0].plot(range(3), [R[(m, s)]["raw"]["macro_f1"] for m in models], color="#cccccc", lw=0.6, zorder=1)
    axes[0].set_xticks(range(3), ["Graph-only", "Frozen GRU", "TGN-JointIDS"])
    axes[0].set_ylabel("Stream macro-F1 (raw 10-class)"); axes[0].set_title("Per seed (lines join paired seeds)", fontsize=9)
    for k, c in enumerate(CRIT):
        for j, m in enumerate(models):
            v = [R[(m, s)]["raw"]["per_class"][c]["recall"] for s in SEEDS]
            axes[1].errorbar(k + (j - 1) * 0.22, np.mean(v), yerr=np.std(v, ddof=1), fmt="o", color=COLORS[m],
                             capsize=3, ms=4, label=["Graph-only", "Frozen GRU", "TGN-JointIDS"][j] if k == 0 else None)
    axes[1].set_xticks(range(3), ["MITM", "Ransomware", "Scanning"]); axes[1].set_ylabel("Stream recall (mean ± SD)")
    axes[1].legend(fontsize=7, frameon=False); axes[1].set_title("Critical-class recall", fontsize=9)
    fig.suptitle("Memory ablation under the matched 7,000-update protocol", fontsize=9)
    fig.tight_layout(); save(fig, "fig3_memory_ablation")
    # F4 ENS precision-recall trade-off
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), sharex=False)
    for ax, c in zip(axes, CRIT):
        for s in SEEDS:
            pa, pb = R[("A", s)]["raw"]["per_class"][c], R[("B", s)]["raw"]["per_class"][c]
            ax.annotate("", xy=(pa["recall"], pa["precision"]), xytext=(pb["recall"], pb["precision"]),
                        arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
            ax.scatter(pb["recall"], pb["precision"], color=COLORS["B"], s=16, zorder=3)
            ax.scatter(pa["recall"], pa["precision"], color=COLORS["A"], s=16, zorder=3)
        ax.set_title({"mitm": "MITM", "ransomware": "Ransomware", "scanning": "Scanning"}[c], fontsize=9)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    axes[0].scatter([], [], color=COLORS["B"], label="w/o ENS"); axes[0].scatter([], [], color=COLORS["A"], label="TGN-JointIDS")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("ENS precision–recall trade-off (arrows: w/o ENS → TGN-JointIDS, same seed)", fontsize=9)
    fig.tight_layout(); save(fig, "fig4_ens_tradeoff")
    # F5 minority recall and F1, all four models
    order = ["graph_only", "frozen_gru", "B", "A"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for ax, met in zip(axes, ("recall", "f1")):
        for k, c in enumerate(["mitm", "ransomware", "scanning", "injection"]):
            for j, m in enumerate(order):
                v = [R[(m, s)]["raw"]["per_class"][c][met] for s in SEEDS]
                ax.bar(k + (j - 1.5) * 0.2, np.mean(v), 0.2, yerr=np.std(v, ddof=1), color=COLORS[m],
                       capsize=2, error_kw={"lw": 0.7}, label=MODELS[m][0] if k == 0 else None)
        ax.set_xticks(range(4), ["MITM", "Ransomware", "Scanning", "Injection"]); ax.set_ylim(0.5, 1.0)
        ax.set_ylabel(f"Stream {'recall' if met == 'recall' else 'F1'} (mean ± SD)")
    fig.suptitle("Minority-class recall and F1, five seeds", fontsize=9)
    fig.tight_layout()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    save(fig, "fig5_minority_comparison")
    # F6 binary PR curve + threshold sensitivity (A)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    th = np.linspace(0.05, 0.95, 37)
    curves = {k: [] for k in ("precision", "recall", "f1", "benign FPR", "attack FNR")}
    for r in A:
        st = r["st"]
        p, rc, _ = precision_recall_curve(st["y_bin"], st["prob_attack"])
        axes[0].plot(rc, p, lw=0.8, color=COLORS["A"], alpha=0.6)
        for t in th:
            pred = st["prob_attack"] >= t
            tp = np.sum(pred & (st["y_bin"] == 1)); fp = np.sum(pred & (st["y_bin"] == 0))
            fn = np.sum(~pred & (st["y_bin"] == 1))
            pr, re = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            curves["precision"].append(pr); curves["recall"].append(re)
            curves["f1"].append(2 * pr * re / max(pr + re, 1e-12))
            curves["benign FPR"].append(fp / max(np.sum(st["y_bin"] == 0), 1))
            curves["attack FNR"].append(fn / max(np.sum(st["y_bin"] == 1), 1))
    ap = np.mean([average_precision_score(r["st"]["y_bin"], r["st"]["prob_attack"]) for r in A])
    axes[0].set_xlabel("Recall (attack)"); axes[0].set_ylabel("Precision (attack)")
    axes[0].set_title(f"Binary precision–recall, 5 seeds (mean AP {ap:.4f})", fontsize=9); axes[0].set_ylim(0.9, 1.0)
    for k, v in curves.items():
        v = np.array(v).reshape(5, len(th))
        axes[1].plot(th, v.mean(0), lw=1.2, label=k)
        axes[1].fill_between(th, v.min(0), v.max(0), alpha=0.15)
    axes[1].axvline(0.5, color="black", lw=0.6, ls=":")
    axes[1].set_xlabel("Attack-probability threshold"); axes[1].set_ylabel("Value")
    axes[1].set_title("Threshold sensitivity (band = seed range)", fontsize=9); axes[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); save(fig, "fig6_binary_pr_threshold")
    # F7 confidence distributions
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8))
    st = np.concatenate([r["st"]["prob_attack"] for r in A]); yb = np.concatenate([r["st"]["y_bin"] for r in A])
    pb = np.concatenate([r["st"]["pred_bin"] for r in A])
    conf = np.where(pb == 1, st, 1 - st)
    for lab, mask, col in (("correct attack", (yb == 1) & (pb == 1), "#1f3a68"),
                           ("correct benign", (yb == 0) & (pb == 0), "#3a9a5b"),
                           ("false alarm", (yb == 0) & (pb == 1), "#c8762b"),
                           ("missed attack", (yb == 1) & (pb == 0), "#b2443a")):
        axes[0].hist(conf[mask], bins=40, range=(0.5, 1.0), histtype="step", density=True, color=col, label=lab, lw=1.1)
    axes[0].set_xlabel("Binary-head confidence"); axes[0].set_ylabel("Density"); axes[0].legend(fontsize=6.5, frameon=False)
    axes[0].set_title("Binary head", fontsize=9)
    pm = np.concatenate([r["st"]["prob_multi"].max(1) for r in A])
    ok = np.concatenate([r["st"]["pred_multi"] == r["st"]["y_multi"] for r in A])
    axes[1].hist(pm[ok], bins=40, range=(0.1, 1.0), histtype="step", density=True, color="#1f3a68", label="correct", lw=1.1)
    axes[1].hist(pm[~ok], bins=40, range=(0.1, 1.0), histtype="step", density=True, color="#b2443a", label="incorrect", lw=1.1)
    axes[1].set_xlabel("10-class head max probability"); axes[1].legend(fontsize=6.5, frameon=False)
    axes[1].set_title("Attack-type head", fontsize=9)
    fig.suptitle("Stream prediction confidence, TGN-JointIDS (five seeds pooled)", fontsize=9)
    fig.tight_layout(); save(fig, "fig7_confidence")
    # F8 rolling performance in stream order, with documented block structure
    st = A[0]["st"]
    order = np.argsort(st["window_id"], kind="stable")
    y, yb = st["y_multi"][order], st["y_bin"][order]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 3.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    lows = []
    for k, r in enumerate(A):
        s_ = r["st"]
        correct = (s_["pred_bin"][order] == s_["y_bin"][order]).astype(float)
        roll = pd.Series(correct).rolling(500, min_periods=500).mean()
        lows.append(np.nanmin(roll.values))
        axes[0].plot(roll.values, lw=0.7, alpha=0.85, color=plt.get_cmap("Greys")(0.45 + 0.11 * k),
                     label=f"seed {SEEDS[k]}")
    axes[0].set_ylabel("Rolling binary accuracy\n(500 windows)")
    axes[0].set_ylim(max(0.0, np.floor(min(lows) * 20) / 20), 1.0)
    axes[0].legend(fontsize=6, frameon=False, ncol=5, loc="lower right")
    axes[0].set_title("Stream in time order (1,322 non-contiguous 60 s blocks; memory reset at every block start); one line per seed",
                      fontsize=8.5)
    cmap = plt.get_cmap("tab10")
    axes[1].scatter(np.arange(len(y)), np.zeros(len(y)), c=[cmap(int(k)) for k in y], marker="|", s=40)
    axes[1].set_yticks([]); axes[1].set_xlabel("Stream window index (time order)")
    for k, lab in enumerate(LABELS):
        axes[1].scatter([], [], color=cmap(k), label=lab, marker="s", s=12)
    axes[1].legend(ncol=5, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.9))
    axes[1].set_title("True attack-type label (attack phases)", fontsize=8)
    fig.tight_layout(); save(fig, "fig8_rolling_stream")
    # F9 convergence + GRU gradients
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for m in ["graph_only", "frozen_gru", "A", "B"]:
        for s in SEEDS:
            h = R[(m, s)]["hist"]
            axes[0].plot(h["update"], h["val_macro_f1"], color=COLORS[m], lw=0.6, alpha=0.7,
                         label=MODELS[m][0] if s == 13 else None)
            if m in ("A", "B"):
                axes[1].plot(h["update"], h["gru_grad_norm"], color=COLORS[m], lw=0.6, alpha=0.7,
                             label=MODELS[m][0] if s == 13 else None)
        axes[2].plot(R[(m, 13)]["hist"]["update"], R[(m, 13)]["hist"]["train_loss_window"], color=COLORS[m], lw=0.9)
    axes[0].set_ylim(0.85, 0.94); axes[0].set_ylabel("Validation macro-F1"); axes[0].legend(fontsize=6, frameon=False)
    axes[1].set_ylabel("GRU gradient norm"); axes[1].legend(fontsize=6, frameon=False)
    axes[2].set_ylabel("Training loss (seed 13)")
    for ax in axes:
        ax.set_xlabel("Optimiser update")
    fig.suptitle("Convergence under the 7,000-update cosine schedule; frozen-GRU and graph-only GRU gradients are zero by design",
                 fontsize=8.5)
    fig.tight_layout(); save(fig, "fig9_convergence")


if __name__ == "__main__":
    main()
