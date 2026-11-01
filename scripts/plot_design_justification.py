"""Paper figures that justify the design choices of TGN-JointIDS, one figure per component.

    python scripts/plot_design_justification.py

fig_justify_memory  TGN-JointIDS vs Graph-only (no temporal memory)
fig_justify_ens     TGN-JointIDS vs w/o ENS (focal loss without ENS class weights)
fig_justify_focal   TGN-JointIDS vs w/o focal (ENS-weighted cross-entropy); only when those runs exist
Rows = datasets (NF-ToN-IoT-v3; NF-CICIDS2018-v3 block-stratified split); one panel per metric with
two bars (ablation, TGN-JointIDS): mean of five seeds, ± SD, dots = seeds. Above each panel:
Δ = TGN-JointIDS − ablation and its 95% CI (60-min cluster bootstrap, results/ablation_statistics/
all_metrics_paired.csv); * = significant gain, † = significant loss (both 10- and 60-min CIs exclude
0). Figures go to results/ablation_statistics/ and to both paper_figures folders.
"""

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_ablation_statistics import SEEDS, datasets, metrics, suff  # noqa: E402

OUT = ROOT / "results/ablation_statistics"
PAPER = {"NF-ToN-IoT-v3": (ROOT / "results/nf_ton_iot_v3/paper_figures", "ton"),
         "NF-CICIDS2018-v3": (ROOT / "results/nf_cicids2018_v3/paper_figures", "cicids")}
CHOICES = {
    "memory": ("graph_only", "Graph-only", "Temporal memory: TGN-JointIDS vs the same model without memory",
               [("multiclass macro-F1", "Macro-F1"), ("rare F1", "Rare-class F1"),
                ("binary FNR", "Missed attacks (FNR) ↓"), ("multiclass accuracy", "Accuracy")]),
    "ens": ("B", "w/o ENS", "ENS class weighting: TGN-JointIDS vs focal loss without ENS weights",
            [("binary FPR", "False alarms (FPR) ↓"), ("rare recall", "Rare-class recall"),
             ("binary precision", "Binary precision"), ("multiclass macro-F1", "Macro-F1")]),
    "focal": ("C", "w/o focal", "Focal loss: TGN-JointIDS vs ENS-weighted cross-entropy",
              [("multiclass macro-F1", "Macro-F1"), ("rare F1", "Rare-class F1"),
               ("rare recall", "Rare-class recall"), ("binary FPR", "False alarms (FPR) ↓")]),
}
COL = {"ablation": "#b0b0b0", "TGN-JointIDS": "#1f3a68"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def seed_values():
    """{dataset: {model: [metrics dict per seed]}} for every model whose runs all exist."""
    out = {}
    for ds, d in datasets().items():
        classes, train = d["classes"], d["train"]
        C, tot = len(classes), sum(train.values())
        rare = np.array([i for i, c in enumerate(classes) if i and train[c] / tot < 0.01])
        out[ds] = {}
        for m, path in d["runs"].items():
            files = [path / f"seed_{s}/predictions_stream.npz" for s in SEEDS]
            if not all(f.exists() for f in files):
                continue
            vals = []
            for f in files:
                p = dict(np.load(f))
                vals.append(metrics(*(x.sum(0) for x in suff(p, np.zeros(len(p["y_multi"]), int), C)),
                                    classes, rare))
            out[ds][m] = vals
    return out


def figure(key, V, pairs):
    abl, abl_name, title, mets = CHOICES[key]
    fig, axes = plt.subplots(2, len(mets), figsize=(3.0 * len(mets), 6.0))
    for i, ds in enumerate(V):
        for j, (m, nm) in enumerate(mets):
            ax = axes[i, j]
            a = np.array([v[m] for v in V[ds][abl]])
            t = np.array([v[m] for v in V[ds]["A"]])
            for k, (vals, lab) in enumerate(((a, abl_name), (t, "TGN-JointIDS"))):
                ax.bar(k, vals.mean(), 0.6, yerr=vals.std(ddof=1), capsize=3,
                       color=COL["TGN-JointIDS" if k else "ablation"], error_kw={"lw": 0.8})
                ax.plot(np.full(len(vals), k) + np.linspace(-0.12, 0.12, len(vals)), vals, "o", ms=2.8,
                        color="#e0a030" if k else "#555555", zorder=3)
            lo, hi = min(a.min(), t.min()), max(a.max(), t.max())
            pad = (hi - lo) * 0.9 + 1e-4
            ax.set_ylim(max(0, lo - pad), hi + 1.6 * pad)
            if hi <= 1.0:   # bounded metric: no tick labels above 1 (the headroom holds the Δ text)
                ax.set_yticks([v for v in ax.get_yticks() if max(0, lo - pad) <= v <= 1.0])
            ax.set_xticks([0, 1], [abl_name, "TGN-JointIDS"], fontsize=8)
            ax.tick_params(axis="y", labelsize=7.5)
            r = pairs[(pairs.dataset == ds) & (pairs.metric == m) & (pairs.comparison == f"TGN-JointIDS vs {abl_name}")]
            if len(r):
                r = r.iloc[0]
                mark = {"significant gain": " *", "significant loss": " †"}.get(r["verdict"], "")
                col = {"significant gain": "#1b7837", "significant loss": "#c0392b"}.get(r["verdict"], "#444444")
                ax.text(0.5, 0.97, f"Δ = {r['Δ']:+.4f}{mark}\n95% CI {r['95% CI (60 min)']}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7.2, color=col)
            if i == 0:
                ax.set_title(nm, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(ds, fontsize=9.5)
    fig.suptitle(title, fontsize=10.5, y=1.0)
    fig.text(0.5, -0.02, "Test partition, mean ± SD of five seeds (dots = seeds); y-axes zoomed per panel. "
             "Δ = TGN-JointIDS − ablation with 95% CI (60-min cluster bootstrap). * significant gain, "
             "† significant loss (10- and 60-min CIs exclude 0). ↓ lower is better.", ha="center", fontsize=7.5)
    fig.tight_layout()
    name = f"fig_justify_{key}"
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{e}", dpi=300, bbox_inches="tight")
        for ds, (folder, tag) in PAPER.items():
            shutil.copy2(OUT / f"{name}.{e}", folder / f"fig_{tag}_justify_{key}.{e}")
    plt.close(fig)
    return name


def main():
    pairs = pd.read_csv(OUT / "all_metrics_paired.csv")
    V = seed_values()
    made = []
    for key, (abl, abl_name, _, _) in CHOICES.items():
        have_runs = all(abl in V[ds] for ds in V)
        have_stats = (pairs.comparison == f"TGN-JointIDS vs {abl_name}").any()
        if have_runs and have_stats:
            made.append(figure(key, V, pairs))
        else:
            print(f"skip {key}: runs or paired statistics not available yet")
    print("made", made)


if __name__ == "__main__":
    main()
