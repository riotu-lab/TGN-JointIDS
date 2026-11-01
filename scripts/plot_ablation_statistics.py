"""Charts for results/ablation_statistics/ (reads all_metrics_paired.csv written by
analyze_ablation_statistics.py).

    python scripts/plot_ablation_statistics.py

fig_ablation_overall       forest plots: multiclass + binary metrics, Δ (TGN-JointIDS − ablation) with
                           95% CIs (thick: 60-min clusters, thin: 10-min clusters), datasets × comparisons
fig_ablation_rare_classes  same for rare-class and per-class metrics (only metrics that are a gain or a
                           significant loss in at least one panel are shown)
fig_ablation_summary       key metric values of TGN-JointIDS vs each ablation, both datasets
FPR and FNR are plotted as improvement (sign flipped: positive = fewer false alarms / missed attacks).
Colours: dark green = significant gain, light green = consistent gain (5/5 seeds, CI includes 0),
red = significant loss, grey = no clear difference.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ablation_statistics"
DATASETS = ["NF-ToN-IoT-v3", "NF-CICIDS2018-v3"]
COMPS = ["TGN-JointIDS vs Graph-only", "TGN-JointIDS vs w/o ENS", "TGN-JointIDS vs w/o focal"]
FLIP = {"binary FPR", "binary FNR"}
VCOL = {"significant gain": "#1b7837", "consistent gain (5/5 seeds; CI includes 0)": "#7fbf7b",
        "significant loss": "#c0392b", "no clear difference": "#9e9e9e"}
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def ci(s):
    if not isinstance(s, str) or s == "n/a":
        return np.nan, np.nan
    a, b = s.strip("[]").split(",")
    return float(a), float(b)


def label(m):
    return {"binary FPR": "binary FPR ↓ (fewer false alarms)", "binary FNR": "binary FNR ↓ (fewer missed attacks)",
            "rare detection (binary head)": "rare detection (binary)"}.get(m, m)


def forest(df, metrics_by_ds, name, title, xlabel):
    if not isinstance(metrics_by_ds, dict):
        metrics_by_ds = {ds: metrics_by_ds for ds in DATASETS}
    heights = [len(metrics_by_ds[ds]) for ds in DATASETS]
    fig, axes = plt.subplots(2, len(COMPS), figsize=(5.6 * len(COMPS), 0.32 * sum(heights) + 2.2),
                             gridspec_kw={"height_ratios": heights})
    for i, ds in enumerate(DATASETS):
        metrics = metrics_by_ds[ds]
        for j, cp in enumerate(COMPS):
            ax = axes[i, j]
            sub = df[(df.dataset == ds) & (df.comparison == cp)].set_index("metric")
            for k, m in enumerate(metrics):
                if m not in sub.index:
                    continue
                r = sub.loc[m]
                sg = -1 if m in FLIP else 1
                d = sg * r["Δ"]
                lo10, hi10 = (sg * x for x in ci(r["95% CI (10 min)"]))
                lo60, hi60 = (sg * x for x in ci(r["95% CI (60 min)"]))
                c = VCOL[r["verdict"]]
                if np.isfinite(lo10):
                    ax.plot([min(lo10, hi10), max(lo10, hi10)], [k, k], color=c, lw=1.0)
                if np.isfinite(lo60):
                    ax.plot([min(lo60, hi60), max(lo60, hi60)], [k, k], color=c, lw=3.2, alpha=0.75,
                            solid_capstyle="butt")
                ax.plot(d, k, "o", color=c, ms=4.5, mec="black", mew=0.4)
                ax.text(1.0, k, r["seeds favourable"], transform=ax.get_yaxis_transform(), fontsize=6.5,
                        va="center", ha="left", color="#444")
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_yticks(range(len(metrics)), [label(m) for m in metrics] if j == 0 else [])
            ax.set_ylim(len(metrics) - 0.5, -0.5)
            ax.set_title(f"{ds}: {cp}", fontsize=9)
            if i == 1:
                ax.set_xlabel(xlabel)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=3, label=v.split(" (")[0]) for v, c in VCOL.items()]
    handles += [plt.Line2D([], [], color="k", lw=3, alpha=0.6, label="95% CI, 60-min clusters"),
                plt.Line2D([], [], color="k", lw=1, label="95% CI, 10-min clusters")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.suptitle(title, y=1.05, fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def headline(df):
    keys = [("multiclass macro-F1", "Macro-F1"), ("binary F1", "Binary F1"), ("binary FNR", "Missed attacks (FNR) ↓"),
            ("binary FPR", "False alarms (FPR) ↓"), ("rare F1", "Rare-class F1"), ("rare recall", "Rare-class recall")]
    colors = {"Graph-only": "#8c8c8c", "w/o ENS": "#c8762b", "w/o focal": "#7b5ea7", "TGN-JointIDS": "#1f3a68"}
    fig, axes = plt.subplots(2, len(keys), figsize=(14, 5.4))
    for i, ds in enumerate(DATASETS):
        for j, (m, nm) in enumerate(keys):
            ax = axes[i, j]
            g = df[(df.dataset == ds) & (df.metric == m)].set_index("comparison")
            vals = [g.loc[c, "ablation"] for c in COMPS] + [g.loc[COMPS[0], "TGN-JointIDS"]]
            verd = [g.loc[c, "verdict"] for c in COMPS] + [None]
            bars = ax.bar(range(len(vals)), vals, 0.7, color=list(colors.values()))
            lo, hi = min(vals), max(vals)
            pad = (hi - lo) * 0.6 + 1e-4
            ax.set_ylim(max(0, lo - pad), hi + pad)
            for b, v, vd in zip(bars, vals, verd):
                mark = {"significant gain": "*", "significant loss": "†"}.get(vd, "")
                ax.text(b.get_x() + b.get_width() / 2, v + pad * 0.08, f"{v:.3f}{mark}", ha="center", va="bottom",
                        fontsize=6, color={"*": "#1b7837", "†": "#c0392b"}.get(mark, "black"))
            ax.set_xticks([])
            ax.spines["left"].set_linewidth(0.6)
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                ax.set_title(nm, fontsize=9)
            if j == 0:
                ax.set_ylabel(ds, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=n) for n, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=9)
    fig.text(0.5, -0.02, "Test values, mean of five seeds (y-axes zoomed per panel). * after an ablation's value: "
             "TGN-JointIDS is significantly better; † significantly worse (10- and 60-min cluster CIs both exclude 0). "
             "↓ lower is better.", ha="center", fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_ablation_summary.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(OUT / "all_metrics_paired.csv")
    overall = ["multiclass macro-F1", "multiclass accuracy", "multiclass weighted F1", "binary accuracy",
               "binary precision", "binary recall", "binary F1", "binary FPR", "binary FNR"]
    forest(df, overall, "fig_ablation_overall", "Overall and binary detection: TGN-JointIDS minus ablation",
           "Δ (improvement of TGN-JointIDS; FPR/FNR sign-flipped)")
    rare = ["rare F1", "rare recall", "rare precision", "rare detection (binary head)"]
    by_ds = {}
    for ds in DATASETS:
        rest = df[(df.dataset == ds) & (df.group == "per class")]
        keep = rest[rest.verdict != "no clear difference"].metric.unique().tolist()
        by_ds[ds] = rare + [m for m in rest.metric.unique() if m in keep]
    forest(df, by_ds, "fig_ablation_rare_classes",
           "Rare-class and per-class detection: TGN-JointIDS minus ablation (classes with a gain or loss)",
           "Δ (improvement of TGN-JointIDS)")
    headline(df)
    print(sorted(p.name for p in OUT.glob("fig_*.png")))


if __name__ == "__main__":
    main()
