"""NF-CICIDS2018-v3 later-day test of TGN-JointIDS (forward in time), known classes only.

    python scripts/analyze_cicids_later_day.py --config configs/nf_cicids2018_v3_later_day.yaml

Training and validation use 14-28 February; the test covers 1-2 March: benign windows of both
days and the 1 March infiltration windows, a class seen in training (on an earlier day). Test
windows labelled bot (2 March), which is not a training class (data.heldout_classes), are
excluded from every number. Reads <output_dir>/seed_<s>/ (frozen causal predictions; checkpoint = best validation
macro-F1; binary threshold fixed at 0.5) and the graph bundle (true labels, IP identifiers for
the IP-only baseline). Writes <output_dir>/../report/:
  table_later_day_test.csv     binary and attack-type metrics, five-seed mean ± SD, per seed, and
                               an IP-only baseline (score = highest training attack share of the
                               window's hosts; no traffic features)
  table_compute.csv            parameters, training time, selected update, validation macro-F1
  fig_confusion_binary_test    binary confusion matrix (mean counts over seeds)
  fig_pr_curve_test            precision-recall curves of every seed and of the IP-only baseline
  fig_validation_curve         validation macro-F1 during training
  statistics.json
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)

from _common import base_parser, load  # noqa: E402
from tgn_jointids.utils import env_path  # noqa: E402

SEEDS = [13, 42, 123, 777, 2026]


def ms(v, d=4):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f}" if len(v) == 1 else f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(f"{path}.png", dpi=200)
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def bundle_side(cfg):
    """True labels of the test windows and the IP-only baseline score."""
    b = torch.load(env_path(cfg.paths.graphs), weights_only=False)
    g, sp = b["graphs"], b["split"]
    hist = defaultdict(Counter)            # IP -> training attack-type label history
    for i in sp["train"]:
        for n in set(g[i].node_ids.tolist()):
            hist[n][int(g[i].y_multi)] += 1
    att_frac = {n: 1 - c.get(0, 0) / sum(c.values()) for n, c in hist.items()}
    idx = sp["stream"]
    ids = [set(g[i].node_ids.tolist()) for i in idx]
    out = {"classes_full": b["classes_full"],
           "window_id": np.array([int(g[i].window_id) for i in idx]),
           "y_full": np.asarray(b["y_multi_full"])[idx],
           "ip_score": np.array([max([att_frac.get(n, 0.0) for n in s] or [0.0]) for s in ids])}
    del b
    return out


def binary_block(y, p, score):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"accuracy": float((tp + tn) / len(y)), "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)), "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "fpr": float(fp / max(fp + tn, 1)), "average_precision": float(average_precision_score(y, score)),
            "auroc": float(roc_auc_score(y, score)),
            "cm": [[int(tn), int(fp)], [int(fn), int(tp)]], "n": int(len(y)), "n_attack": int(y.sum())}


def main():
    p = base_parser(__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args()
    cfg, runs_dir = load(args)
    out = runs_dir.parent / "report"
    out.mkdir(parents=True, exist_ok=True)
    classes = list(cfg.data.classes)
    side = bundle_side(cfg)
    full = side["classes_full"]
    yf = side["y_full"]
    known = np.isin(yf, [full.index(c) for c in classes])     # drops held-out (bot) windows
    ki = classes.index("infilteration")
    assert set(np.unique(yf[known])) <= {0, full.index("infilteration")}, "unexpected test classes"
    man = json.load(open(Path(env_path(cfg.paths.graphs)).with_suffix(".manifest.json")))
    R, runs = {}, {}
    for s in args.seeds:
        d = runs_dir / f"seed_{s}"
        st, va = dict(np.load(d / "predictions_stream.npz")), dict(np.load(d / "predictions_val.npz"))
        info, hist = json.load(open(d / "run_info.json")), json.load(open(d / "history.json"))
        assert info["split_sha256"] == man["split_sha256"], "predictions do not belong to this bundle"
        assert np.array_equal(st["window_id"], side["window_id"])
        runs[s] = {"stream": st, "info": info, "hist": hist}
        y, pb, sc = st["y_bin"][known], st["pred_bin"][known], st["prob_attack"][known]
        ym, pm = st["y_multi"][known], st["pred_multi"][known]
        P, Rc, F, _ = precision_recall_fscore_support(ym, pm, labels=[0, ki], zero_division=0)
        R[s] = {**binary_block(y, pb, sc), "multi_accuracy": float(accuracy_score(ym, pm)),
                "multi_macro_f1": float(F.mean()), "infiltration_precision": float(P[1]),
                "infiltration_recall": float(Rc[1]), "infiltration_f1": float(F[1]),
                "val_macro_f1": float(f1_score(va["y_multi"], va["pred_multi"], labels=list(range(len(classes))),
                                               average="macro", zero_division=0))}
    bs = side["ip_score"][known]
    ipb = binary_block(yf[known] > 0, (bs >= 0.5).astype(int), bs)

    names = [("accuracy", "Binary accuracy"), ("precision", "Binary precision"), ("recall", "Binary recall"),
             ("f1", "Binary F1"), ("fpr", "Benign FPR"), ("average_precision", "Average precision"),
             ("auroc", "AUROC"), ("multi_accuracy", "Attack-type accuracy (benign + infiltration)"),
             ("multi_macro_f1", "Attack-type macro-F1 (benign, infiltration)"),
             ("infiltration_precision", "Infiltration precision (attack-type head)"),
             ("infiltration_recall", "Infiltration recall (attack-type head)"),
             ("infiltration_f1", "Infiltration F1 (attack-type head)"),
             ("val_macro_f1", "Validation macro-F1 (selection only; 14-28 Feb)")]
    rows = []
    for k, nm in names:
        vals = [R[s][k] for s in args.seeds]
        rows.append({"metric": nm, "TGN-JointIDS": ms(vals), **{f"seed {s}": round(v, 4) for s, v in zip(args.seeds, vals)},
                     "IP-only baseline": round(ipb[k], 4) if k in ipb else np.nan})
    t = pd.DataFrame(rows).set_index("metric")
    t.to_csv(out / "table_later_day_test.csv")
    pd.DataFrame({s: {k: runs[s]["info"].get(k) for k in ("parameters", "train_seconds_total",
                                                           "peak_gpu_mb_training", "best_update",
                                                           "best_val_macro_f1")}
                  for s in args.seeds}).T.to_csv(out / "table_compute.csv")

    # binary confusion matrix (mean counts)
    cm = np.mean([np.asarray(R[s]["cm"], float) for s in args.seeds], 0)
    v = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(v, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{v[i, j]:.2f}\n({cm[i, j]:.0f})", ha="center", va="center", fontsize=8,
                    color="white" if v[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], ["Benign", "Attack"])
    ax.set_yticks([0, 1], ["Benign", "Infiltration"])
    ax.set_xlabel("Predicted (binary head)")
    ax.set_ylabel("True")
    ax.set_title("Later-day test (1-2 March)", fontsize=10)
    fig.colorbar(im, fraction=0.046)
    savefig(fig, out / "fig_confusion_binary_test")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    for s in args.seeds:
        st = runs[s]["stream"]
        pr, rc, _ = precision_recall_curve(st["y_bin"][known], st["prob_attack"][known])
        ax.plot(rc, pr, lw=1, label=f"TGN-JointIDS seed {s} (AP {R[s]['average_precision']:.3f})")
    pr, rc, _ = precision_recall_curve(yf[known] > 0, bs)
    ax.plot(rc, pr, "--", color="#d69e2e", lw=1, label=f"IP-only baseline (AP {ipb['average_precision']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Later-day test: binary detection (1-2 March)", fontsize=10)
    ax.legend(fontsize=7)
    savefig(fig, out / "fig_pr_curve_test")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for s in args.seeds:
        h = runs[s]["hist"]
        ax.plot(h["update"], h["val_macro_f1"], lw=1, label=f"seed {s}")
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Later-day test: validation curve (checkpoint selection)", fontsize=10)
    ax.legend(fontsize=7)
    savefig(fig, out / "fig_validation_curve")

    json.dump({"seeds": args.seeds, "split_sha256": man["split_sha256"], "test_windows": int(known.sum()),
               "test_infiltration_windows": int((yf[known] > 0).sum()),
               "per_seed": {str(s): R[s] for s in args.seeds}, "ip_only_baseline": ipb},
              open(out / "statistics.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(t.to_string())


if __name__ == "__main__":
    main()
