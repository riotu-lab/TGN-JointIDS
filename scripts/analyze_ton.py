"""NF-ToN-IoT-v3: TGN-JointIDS, five seeds - tables, figures and time-cluster uncertainty.

    python scripts/analyze_ton.py

Inputs: results/nf_ton_iot_v3/tgn_jointids/seed_<s>/ and results/nf_ton_iot_v3/latency_idle.json.
"stream" = the test partition. Writes results/nf_ton_iot_v3/report/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _ton_metrics import CLASSES, low_purity_mask, metrics  # noqa: E402
from tgn_jointids.stats import confusion_by_cluster, macro_f1_from_cm  # noqa: E402

SEEDS = [13, 42, 123, 777, 2026]
BASE = ROOT / "results/nf_ton_iot_v3"
RUNS = BASE / "tgn_jointids"
OUT = BASE / "report"
CRIT = ["mitm", "ransomware", "scanning"]
MINOR = ["mitm", "ransomware", "scanning", "injection"]


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def recall_from_cm(cm, c):
    s = cm[c].sum()
    return cm[c, c] / s if s else np.nan


def f1_from_cm(cm, c):
    tp, d = cm[c, c], cm[c].sum() + cm[:, c].sum()
    return 2 * tp / d if d else np.nan


def cluster_ci(cms, stat, n_boot=2000, seed=0):
    """Bootstrap CI of the seed-mean of ``stat(cm)`` resampling time clusters (shared across seeds)."""
    cms = np.stack(cms)                              # [S, K, C, C]
    K = cms.shape[1]
    rng = np.random.default_rng(seed)
    full = np.mean([stat(cms[s].sum(0)) for s in range(len(cms))])
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
        agg = np.tensordot(w, cms, axes=([0], [1]))
        boots.append(np.nanmean([stat(agg[s]) for s in range(len(cms))]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(full), float(lo), float(hi), int(K)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    low_mask, wid_ref, t0 = low_purity_mask("data/processed/nf_ton_iot_v3.pt", "stream")
    R = {}
    for s in SEEDS:
        run = RUNS / f"seed_{s}"
        st, va = dict(np.load(run / "predictions_stream.npz")), dict(np.load(run / "predictions_val.npz"))
        assert np.array_equal(st["window_id"], wid_ref)
        R[s] = {"st": st, "va": va, "stream": metrics(st, low_mask), "val": metrics(va),
                "hist": json.load(open(run / "history.json")), "info": json.load(open(run / "run_info.json"))}
    shas = {R[s]["info"]["split_sha256"] for s in SEEDS}
    assert len(shas) == 1, "seeds used different partitions"
    lat = json.load(open(BASE / "latency_idle.json")) if (BASE / "latency_idle.json").exists() else {}

    # ---------------- Table 1: overall
    keys = [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"),
            ("macro_f1", "Macro-F1"), ("weighted_f1", "Weighted F1"), ("binary_f1", "Binary F1 (binary head)"),
            ("benign_fpr_binary_head", "Benign FPR (binary head)"),
            ("benign_misclass_multiclass_head", "Benign misclassified (attack-type head)")]
    t1 = []
    for k, name in keys:
        t1.append({"metric": name, "validation": ms([R[s]["val"][k] for s in SEEDS]),
                   "stream": ms([R[s]["stream"][k] for s in SEEDS]),
                   **{f"stream seed {s}": round(R[s]["stream"][k], 4) for s in SEEDS}})
    t1.append({"metric": "Low-identity macro-F1", "validation": "",
               "stream": ms([R[s]["stream"]["low_identity_macro_f1"] for s in SEEDS]),
               **{f"stream seed {s}": round(R[s]["stream"]["low_identity_macro_f1"], 4) for s in SEEDS}})
    t1 = pd.DataFrame(t1).set_index("metric")

    # ---------------- Table 2: per class (stream), per seed + mean ± sd
    rows = []
    for c in CLASSES:
        for k in ("precision", "recall", "f1"):
            vals = [R[s]["stream"]["per_class"][c][k] for s in SEEDS]
            rows.append({"class": c, "metric": k, "mean ± sd": ms(vals),
                         **{f"seed {s}": round(v, 4) for s, v in zip(SEEDS, vals)},
                         "support": R[13]["stream"]["per_class"][c]["support"]})
    t2 = pd.DataFrame(rows).set_index(["class", "metric"])
    rows = []
    for c in CLASSES:
        for k in ("precision", "recall", "f1"):
            vals = [R[s]["val"]["per_class"][c][k] for s in SEEDS]
            rows.append({"class": c, "metric": k, "mean ± sd": ms(vals),
                         "support": R[13]["val"]["per_class"][c]["support"]})
    t2v = pd.DataFrame(rows).set_index(["class", "metric"])

    # ---------------- Table 3: critical attacks
    rows = []
    for c in CRIT:
        pc = [R[s]["stream"]["per_class"][c] for s in SEEDS]
        rows.append({"class": c, "recall (mean ± sd)": ms([p["recall"] for p in pc]),
                     "F1 (mean ± sd)": ms([p["f1"] for p in pc]),
                     **{f"recall s{s}": round(p["recall"], 3) for s, p in zip(SEEDS, pc)},
                     **{f"F1 s{s}": round(p["f1"], 3) for s, p in zip(SEEDS, pc)},
                     "FN (mean ± sd)": ms([p["fn"] for p in pc], 1), "FP (mean ± sd)": ms([p["fp"] for p in pc], 1),
                     "benign→class (mean ± sd)": ms([p["benign_as"] for p in pc], 1),
                     "support": pc[0]["support"]})
    t3 = pd.DataFrame(rows).set_index("class")

    # ---------------- Table 4: compute
    infos = [R[s]["info"] for s in SEEDS]
    t4 = pd.DataFrame([{
        "total parameters": infos[0]["parameters"], "trainable parameters": infos[0]["parameters"],
        "training time (min, mean ± sd)": ms([i["train_seconds_total"] / 60 for i in infos], 1),
        "peak GPU memory, training (MB)": ms([i["peak_gpu_mb_training"] for i in infos], 0),
        "best checkpoint update (mean ± sd)": ms([i["best_update"] for i in infos], 0),
        "best updates per seed": [i["best_update"] for i in infos],
        "latency mean ms (mean over seeds)": np.mean([lat[str(s)]["mean_ms"] for s in SEEDS]) if lat else np.nan,
        "latency median ms": np.mean([lat[str(s)]["median_ms"] for s in SEEDS]) if lat else np.nan,
        "latency P95 ms": np.mean([lat[str(s)]["p95_ms"] for s in SEEDS]) if lat else np.nan,
        "peak GPU memory, inference (MB)": np.mean([lat[str(s)]["peak_gpu_mb"] for s in SEEDS]) if lat else np.nan,
    }]).T

    # ---------------- uncertainty: cluster bootstrap over the five seeds
    C = len(CLASSES)
    unc = {}
    for cm_min in (10, 60):
        clusters = (wid_ref - t0) // (cm_min * 60000)
        cms = [confusion_by_cluster(R[s]["st"]["y_multi"], R[s]["st"]["pred_multi"], clusters, C)[1] for s in SEEDS]
        unc[f"{cm_min}min"] = {"macro_f1": cluster_ci(cms, macro_f1_from_cm)}
        for c in CRIT + ["injection"]:
            i = CLASSES.index(c)
            unc[f"{cm_min}min"][f"{c}_recall"] = cluster_ci(cms, lambda m, i=i: recall_from_cm(m, i))
            unc[f"{cm_min}min"][f"{c}_f1"] = cluster_ci(cms, lambda m, i=i: f1_from_cm(m, i))

    # ---------------- independent support behind minority conclusions
    y = R[13]["st"]["y_multi"]
    blk = (wid_ref - t0) // 60000
    support = {}
    for c in CRIT + ["injection"]:
        m = y == CLASSES.index(c)
        b = np.unique(blk[m])
        runs = 1 + int(np.sum(np.diff(b) > 1)) if len(b) else 0   # contiguous runs of blocks
        support[c] = {"stream_windows": int(m.sum()), "stream_60s_blocks": int(len(b)),
                      "stream_10min_clusters": int(len(np.unique((wid_ref[m] - t0) // 600000))),
                      "stream_60min_clusters": int(len(np.unique((wid_ref[m] - t0) // 3600000))),
                      "contiguous_block_runs": runs, "attack_phases": 1 if c != "benign" else None}

    # ---------------- save
    for name, t in (("table1_overall", t1), ("table2_per_class_stream", t2), ("table2b_per_class_val", t2v),
                    ("table3_critical", t3), ("table4_compute", t4)):
        t.to_csv(OUT / f"{name}.csv")
    json.dump({"uncertainty_cluster_bootstrap": unc, "minority_support": support,
               "per_seed": {s: {"val_macro_f1": R[s]["val"]["macro_f1"], "stream_macro_f1": R[s]["stream"]["macro_f1"],
                                "best_update": R[s]["info"]["best_update"],
                                "gru_grad_norm_min": min(R[s]["hist"]["gru_grad_norm"]),
                                "gru_grad_norm_max": max(R[s]["hist"]["gru_grad_norm"])} for s in SEEDS}},
              open(OUT / "statistics.json", "w"), indent=1, default=float)
    figures(R)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for name, t in (("T1", t1), ("T2 stream", t2), ("T2 val", t2v), ("T3", t3), ("T4", t4)):
        print(f"== {name}\n{t.to_string()}\n")
    print(json.dumps({"uncertainty": unc, "support": support}, indent=1, default=float))


def figures(R):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    col = "#2f4b7c"
    # per-seed macro-F1 and critical recall
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, (split, name) in enumerate((("val", "validation"), ("stream", "stream"))):
        v = [R[s][split]["macro_f1"] for s in SEEDS]
        axes[0].scatter(np.full(len(v), j) + np.linspace(-0.08, 0.08, len(v)), v, color=col, s=24, zorder=3)
        axes[0].errorbar(j + 0.2, np.mean(v), yerr=np.std(v, ddof=1), fmt="o", color="black", capsize=4)
    axes[0].set_xticks([0, 1], ["validation", "stream"]); axes[0].set_ylabel("10-class macro-F1")
    axes[0].set_xlim(-0.4, 1.5); axes[0].set_title("Macro-F1 per seed (dots) and mean ± SD", fontsize=10)
    for j, c in enumerate(MINOR):
        for k, met in enumerate(("recall", "precision")):
            v = [R[s]["stream"]["per_class"][c][met] for s in SEEDS]
            x = j + (k - 0.5) * 0.3
            axes[1].scatter(np.full(len(v), x) + np.linspace(-0.04, 0.04, len(v)), v, s=14,
                            color=["#2f6db3", "#c8762b"][k], label=met if j == 0 else None, zorder=3)
            axes[1].errorbar(x, np.mean(v), yerr=np.std(v, ddof=1), fmt="_", color="black", capsize=3)
    axes[1].set_xticks(range(4), ["MITM", "ransomware", "scanning", "injection"]); axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title("Minority classes on stream, per seed", fontsize=10); axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout(); save(fig, "fig1_seed_results")
    # mean confusion matrix
    cm = np.mean([np.array(R[s]["stream"]["cm"]) for s in SEEDS], axis=0)
    cmn = cm / cm.sum(1, keepdims=True)
    ab = ["benign", "DDoS", "XSS", "password", "scanning", "backdoor", "injection", "DoS", "MITM", "ransomware"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), ab, rotation=45, ha="right"); ax.set_yticks(range(10), ab)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 0.5:
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised recall")
    ax.set_title("TGN-JointIDS, test confusion matrix (mean counts over 5 seeds)", fontsize=10)
    fig.tight_layout(); save(fig, "fig2_confusion_mean")
    # convergence
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for s in SEEDS:
        h = R[s]["hist"]
        axes[0].plot(h["update"], h["val_macro_f1"], lw=1, label=f"seed {s}")
        axes[0].axvline(R[s]["info"]["best_update"], lw=0.5, ls=":", color=axes[0].lines[-1].get_color())
        axes[1].plot(h["update"], h["gru_grad_norm"], lw=1)
    axes[0].set_ylim(0.85, 0.94); axes[0].set_ylabel("validation macro-F1"); axes[0].set_xlabel("optimiser update")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_ylabel("GRU gradient norm (mean per 100 updates)"); axes[1].set_xlabel("optimiser update")
    fig.suptitle("TGN-JointIDS convergence (dotted = selected checkpoint)", fontsize=10)
    fig.tight_layout(); save(fig, "fig3_convergence")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
