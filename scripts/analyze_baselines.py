"""TGN-JointIDS vs external baselines, both datasets, five paired seeds.

    python scripts/analyze_baselines.py [--n-boot 1000]

Baselines (results/baselines/<dataset>/<model>/seed_<s>/, scripts/train_baseline_*.py): XGBoost
and Random Forest on per-window feature vectors, E-GraphSAGE (static edge-feature GNN) and
Window-GRU (temporal, no graph). Graph-only (TGN-JointIDS without memory) is shown as the internal
reference. All models share the windows, split, labels and train-only scaling, and are selected on
validation only. Metrics and the paired cluster bootstrap (10- and 60-min time clusters, same
resampled clusters for every model and seed) are those of scripts/analyze_ablation_statistics.py;
Δ = TGN-JointIDS minus the baseline. Latency: idle GPU, one window per call
(results/baselines/latency_idle.json, scripts/latency_baselines.py).
Writes results/baselines/report/:
  table_main_<dataset>.csv / .tex      mean ± SD of every model (test)
  table_per_class_f1.csv               per-class F1 of every model
  all_metrics_paired.csv               Δ, CIs and verdict for every metric and baseline
  where_tgn_improves.csv/.tex, where_tgn_is_worse.csv
  fig_baselines_main.pdf/png           key metrics, all models, both datasets
  fig_baselines_per_class.pdf/png      per-class F1 heatmap
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_ablation_statistics import LOWER_BETTER, SEEDS, SHORT, datasets, metrics, suff  # noqa: E402

B = ROOT / "results/baselines"
OUT = B / "report"
DSDIR = {"NF-ToN-IoT-v3": "nf_ton_iot_v3", "NF-CICIDS2018-v3": "nf_cicids2018_v3"}
BASE = {"xgboost": "XGBoost", "random_forest": "Random Forest", "egraphsage": "E-GraphSAGE",
        "window_gru": "Window-GRU"}
NAMES = {"A": "TGN-JointIDS", "graph_only": "Graph-only", **BASE}
ORDER = ["random_forest", "xgboost", "window_gru", "egraphsage", "graph_only", "A"]
COL = {"random_forest": "#a6bddb", "xgboost": "#74a9cf", "window_gru": "#c8762b", "egraphsage": "#7b5ea7",
       "graph_only": "#8c8c8c", "A": "#1f3a68"}
TGN_LAT = {"NF-ToN-IoT-v3": ROOT / "results/nf_ton_iot_v3/latency_idle.json",
           "NF-CICIDS2018-v3": ROOT / "results/nf_cicids2018_v3/latency_idle.json"}
KEY = ["multiclass macro-F1", "multiclass accuracy", "multiclass weighted F1", "binary F1", "binary precision",
       "binary recall", "binary FPR", "binary FNR", "binary average precision", "rare F1", "rare recall"]
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def ms(v, d=4):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.{d}f} ± {np.nanstd(v, ddof=1):.{d}f}"


def load_runs(d, ds):
    runs = {"A": d["runs"]["A"], "graph_only": d["runs"]["graph_only"]}
    runs.update({m: B / DSDIR[ds] / m for m in BASE})
    return runs


def info_of(m, path, ds):
    """Parameters and idle latency (mean ms over seeds)."""
    lat = json.load(open(B / "latency_idle.json")).get(ds, {}) if (B / "latency_idle.json").exists() else {}
    if m == "A":
        L = json.load(open(TGN_LAT[ds]))
        mean_ms = np.mean([L[str(s)]["mean_ms"] for s in SEEDS])
    elif m in lat:
        mean_ms = np.mean([lat[m][str(s)]["mean_ms"] for s in SEEDS])
    else:
        mean_ms = np.nan
    ri = json.load(open(path / f"seed_{SEEDS[0]}/run_info.json"))
    return {"parameters": ri.get("parameters"), "latency_ms": mean_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_class, mains = [], [], {}
    for ds, d in datasets().items():
        classes, train = d["classes"], d["train"]
        C, tot = len(classes), sum(train.values())
        rare = np.array([i for i, c in enumerate(classes) if i and train[c] / tot < 0.01])
        runs = load_runs(d, ds)
        P = {m: [dict(np.load(runs[m] / f"seed_{s}/predictions_stream.npz")) for s in SEEDS] for m in ORDER}
        ref = P["A"][0]
        for m in P:
            for p in P[m]:
                assert np.array_equal(p["window_id"], ref["window_id"]) and np.array_equal(p["y_multi"], ref["y_multi"])
        zero = np.zeros(len(ref["y_multi"]), int)
        point = {m: [metrics(*(x.sum(0) for x in suff(p, zero, C)), classes, rare) for p in P[m]] for m in P}
        for m in P:
            for k, p in enumerate(P[m]):
                point[m][k]["binary average precision"] = average_precision_score(p["y_bin"], p["prob_attack"])
        names = [k for k in point["A"][0] if k != "binary average precision"]
        # main table
        tab = {}
        for m in ORDER:
            inf = info_of(m, runs[m], ds)
            tab[NAMES[m]] = {**{k: ms([x[k] for x in point[m]]) for k in KEY},
                             "parameters": f"{inf['parameters']:,}" if inf["parameters"] else "–",
                             "latency per window (ms)": f"{inf['latency_ms']:.2f}" if np.isfinite(inf["latency_ms"]) else "–"}
        mains[ds] = point
        t = pd.DataFrame(tab).T
        t.index.name = "model"
        t.to_csv(OUT / f"table_main_{DSDIR[ds]}.csv")
        tex = t[["multiclass macro-F1", "multiclass accuracy", "binary F1", "binary FPR", "binary FNR",
                 "rare F1", "latency per window (ms)"]].to_latex(
            escape=True, caption=f"{ds}: TGN-JointIDS vs baselines (test, mean ± SD of five seeds).",
            label=f"tab:baselines_{DSDIR[ds]}")
        (OUT / f"table_main_{DSDIR[ds]}.tex").write_text(tex)
        for i, c in enumerate(classes):
            per_class.append({"dataset": ds, "class": SHORT[c], "train windows": train[c],
                              **{NAMES[m]: round(float(np.nanmean([x[f'{SHORT[c]} F1'] if i else np.nan
                                                                   for x in point[m]])), 4) if i else np.nan
                                 for m in ORDER}})
        # paired bootstrap
        t0 = int(ref["window_id"].min())
        ci = {}
        comps = [m for m in ORDER if m != "A"]
        for cmin in (10, 60):
            cl = (ref["window_id"] - t0) // (cmin * 60000)
            S = {m: [suff(p, cl, C) for p in P[m]] for m in P}
            K = S["A"][0][0].shape[0]
            rng = np.random.default_rng(0)
            boots = {b: {k: [] for k in names} for b in comps}
            for _ in range(args.n_boot):
                w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
                agg = {m: [metrics(*(np.tensordot(w, x, 1) for x in S[m][s]), classes, rare) for s in range(5)]
                       for m in P}
                for b in comps:
                    for k in names:
                        boots[b][k].append(np.nanmean([agg["A"][s][k] - agg[b][s][k] for s in range(5)]))
            for b in comps:
                for k in names:
                    v = np.array(boots[b][k], float)
                    v = v[np.isfinite(v)]
                    ci[(b, k, cmin)] = np.percentile(v, [2.5, 97.5]) if len(v) > 0.95 * args.n_boot else (np.nan, np.nan)
        for b in comps:
            for k in names + ["binary average precision"]:
                da, db = [x[k] for x in point["A"]], [x[k] for x in point[b]]
                delta = np.array(da, float) - np.array(db, float)
                sign = -1 if k in LOWER_BETTER else 1
                fav = int(np.sum(sign * delta > 0))
                lo10, hi10 = ci.get((b, k, 10), (np.nan, np.nan))
                lo60, hi60 = ci.get((b, k, 60), (np.nan, np.nan))
                fin = all(np.isfinite(x) for x in (lo10, hi10, lo60, hi60))
                if fin and all(sign * lo > 0 and sign * hi > 0 for lo, hi in ((lo10, hi10), (lo60, hi60))):
                    verdict = "significant gain"
                elif fin and all(sign * lo < 0 and sign * hi < 0 for lo, hi in ((lo10, hi10), (lo60, hi60))):
                    verdict = "significant loss"
                elif fav == 5 and np.nanmean(sign * delta) > 0:
                    verdict = "consistent gain (5/5 seeds; CI includes 0)"
                else:
                    verdict = "no clear difference"
                group = "per class" if k.split()[0] in SHORT.values() and not k.startswith("rare") else k.split()[0]
                rows.append({"dataset": ds, "comparison": f"TGN-JointIDS vs {NAMES[b]}", "group": group, "metric": k,
                             "TGN-JointIDS": round(float(np.nanmean(da)), 4), "baseline": round(float(np.nanmean(db)), 4),
                             "Δ": round(float(np.nanmean(delta)), 4), "seeds favourable": f"{fav}/5",
                             "95% CI (10 min)": f"[{lo10:.4f}, {hi10:.4f}]" if np.isfinite(lo10) else "n/a",
                             "95% CI (60 min)": f"[{lo60:.4f}, {hi60:.4f}]" if np.isfinite(lo60) else "n/a",
                             "verdict": verdict})
        print("done", ds, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_metrics_paired.csv", index=False)
    pc = pd.DataFrame(per_class).dropna(subset=["TGN-JointIDS"])
    pc.to_csv(OUT / "table_per_class_f1.csv", index=False)
    gains = df[df.verdict.str.contains("gain")]
    gains.to_csv(OUT / "where_tgn_improves.csv", index=False)
    key_rows = df[df.metric.isin(KEY)]
    cols = ["metric", "TGN-JointIDS", "baseline", "Δ", "seeds favourable", "95% CI (60 min)", "verdict"]
    (OUT / "where_tgn_improves.tex").write_text(key_rows.set_index(["dataset", "comparison"])[cols].to_latex(
        escape=True, caption="TGN-JointIDS minus each baseline on the key metrics (test, five paired seeds; 95\\% "
        "CI from a 60-min cluster bootstrap; verdict uses the 10- and 60-min CIs; lower is better for FPR/FNR).",
        label="tab:baselines_paired"))
    df[df.verdict == "significant loss"].to_csv(OUT / "where_tgn_is_worse.csv", index=False)
    figure_main(mains)
    figure_per_class(pc)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    for ds in DSDIR.values():
        print(pd.read_csv(OUT / f"table_main_{ds}.csv", index_col=0)[
            ["multiclass macro-F1", "binary F1", "binary FPR", "rare F1", "latency per window (ms)"]].to_string())
    print(key_rows[["dataset", "comparison", "metric", "TGN-JointIDS", "baseline", "Δ", "seeds favourable",
                    "95% CI (60 min)", "verdict"]].to_string(index=False))


def figure_main(mains):
    keys = [("multiclass macro-F1", "Macro-F1"), ("binary F1", "Binary F1"), ("binary FPR", "False alarms (FPR) ↓"),
            ("binary FNR", "Missed attacks (FNR) ↓"), ("rare F1", "Rare-class F1")]
    fig, axes = plt.subplots(2, len(keys), figsize=(13.5, 5.6))
    for i, (ds, point) in enumerate(mains.items()):
        for j, (k, nm) in enumerate(keys):
            ax = axes[i, j]
            allv = []
            for x, m in enumerate(ORDER):
                v = np.array([p[k] for p in point[m]], float)
                allv += list(v)
                ax.bar(x, v.mean(), 0.7, yerr=v.std(ddof=1), capsize=2, color=COL[m], error_kw={"lw": 0.7})
                ax.plot(np.full(5, x) + np.linspace(-0.15, 0.15, 5), v, "o", ms=2, color="black", alpha=0.6, zorder=3)
            lo, hi = min(allv), max(allv)
            pad = (hi - lo) * 0.25 + 1e-4
            ax.set_ylim(max(0, lo - pad), hi + pad)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                ax.set_title(nm, fontsize=9)
            if j == 0:
                ax.set_ylabel(ds, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[m], label=NAMES[m]) for m in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=8.5)
    fig.text(0.5, -0.02, "Test partition, mean ± SD of five seeds (dots = seeds); y-axes zoomed per panel. "
             "↓ lower is better. Graph-only = TGN-JointIDS without temporal memory.", ha="center", fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_baselines_main.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_per_class(pc):
    dss = list(DSDIR)
    models = [NAMES[m] for m in ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [9, 14]})
    for ax, ds in zip(axes, dss):
        g = pc[pc.dataset == ds].set_index("class")[models]
        im = ax.imshow(g.T.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(g)), g.index, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(models)), models if ds == dss[0] else [], fontsize=8)
        for a in range(len(models)):
            for c in range(len(g)):
                v = g.T.values[a, c]
                best = v >= g.T.values[:, c].max() - 1e-9
                ax.text(c, a, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 else "black", fontweight="bold" if best else "normal")
        ax.set_title(ds, fontsize=9.5)
    fig.colorbar(im, ax=axes, fraction=0.02, label="F1 (test, mean of five seeds)")
    fig.suptitle("Per-class F1: TGN-JointIDS vs baselines (bold = best per class)", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_baselines_per_class.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
