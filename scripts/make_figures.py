"""Paper figures from a stream run: binary CM (Fig. 2), attack-type CM (Fig. 3),
rolling accuracy (Fig. 5) and confidence distributions (Fig. 6).

    python scripts/make_figures.py --config configs/nf_ton_iot_v3.yaml \
        --stream-results results/nf_ton_iot_v3/seed_2026/stream_results.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from _common import base_parser, load  # noqa: E402
from tgn_jointids.metrics import attack_only_metrics  # noqa: E402
from tgn_jointids.pipeline import make_label_space  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def _heatmap(ax, mat, xlabels, ylabels, counts=None):
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = f"{100 * mat[i, j]:.2f}%" if counts is None else f"{counts[i, j]:,}\n({100 * mat[i, j]:.2f}%)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > 0.5 else "black")
    ax.set(xlabel="Predicted", ylabel="Actual")


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--stream-results", required=True)
    args = p.parse_args()
    cfg, _ = load(args)
    labels = make_label_space(cfg)
    df = pd.read_csv(args.stream_results)
    out = Path(args.stream_results).parent

    cm = confusion_matrix(df["bin_true"], df["bin_pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    _heatmap(ax, cm / cm.sum(1, keepdims=True), ["Benign", "Attack"], ["Benign", "Attack"], cm)
    save(fig, out, "fig_binary_confusion")

    _, _, cm_a = attack_only_metrics(df, labels, cascade=True)
    names = [labels.id2attack[i] for i in labels.attack_ids]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    _heatmap(ax, cm_a / np.maximum(cm_a.sum(1, keepdims=True), 1), names + ["benign"], names)
    save(fig, out, "fig_attack_type_confusion")

    w = cfg.evaluation.rolling_window
    acc = (df["bin_pred"] == df["bin_true"]).astype(float).rolling(w).mean()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.plot(acc.index, acc.values)
    ax.axhline(acc.mean(), ls="--", color="tab:red", label=f"Mean = {acc.mean():.3f}")
    ax.set(xlabel="Streaming window index", ylabel=f"Rolling {w}-window accuracy")
    ax.legend()
    save(fig, out, "fig_rolling_accuracy")

    tp = df[(df["bin_true"] == 1) & (df["bin_pred"] == 1)]["bin_conf"]
    ben = df[df["bin_true"] == 0]["bin_conf"]
    fig, ax = plt.subplots(figsize=(6, 3))
    lo = np.nanmin([tp.min() if len(tp) else 1.0, ben.min() if len(ben) else 1.0, 0.5])
    bins = np.linspace(lo, 1, 60)
    ax.hist(tp, bins=bins, alpha=0.7, label=f"Attack (mean={tp.mean():.4f})", color="tab:red")
    ax.hist(ben, bins=bins, alpha=0.7, label=f"Benign (mean={ben.mean():.4f})", color="tab:blue")
    ax.axvline(cfg.continual.alert_threshold, ls="--", color="k",
               label=f"Alert threshold τ = {cfg.continual.alert_threshold}")
    ax.set(xlabel="Softmax confidence", ylabel="Count")
    ax.legend()
    save(fig, out, "fig_confidence")
    print(f"High-confidence TP coverage: {(tp >= cfg.continual.alert_threshold).mean():.4f}")
    print(f"Figures -> {out}")


if __name__ == "__main__":
    main()
