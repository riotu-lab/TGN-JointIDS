"""Rare-attack analysis of the TGN-JointIDS ablations on both datasets.

    python scripts/analyze_rare_attacks.py

Models: TGN-JointIDS, Graph-only (no memory) and w/o ENS (focal loss without ENS weights).
Rare classes are fixed a priori as attack classes with < 1% of the training windows:
NF-ToN-IoT-v3 -> MITM, ransomware; NF-CICIDS2018-v3 (block-stratified split) -> GoldenEye,
Slowloris, Hulk, LOIC-UDP, XSS, SQL injection. For each model and seed (13, 42, 123, 777, 2026), on the test (stream) partition:
  rare macro-F1 / recall / precision  (attack-type head, raw arg max, averaged over rare classes)
  rare detection rate                 (share of rare-class windows flagged by the binary head)
Paired differences TGN-JointIDS minus each ablation: seed-mean difference with a 95% CI from a
cluster bootstrap over 10- and 60-min time clusters (2,000 resamples, the same clusters for
both models and all seeds). Writes results/rare_attack_analysis/ (CSV + LaTeX tables, PNG + PDF
figures) and copies the figures into both paper_figures folders.
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "rare_attack_analysis"
SEEDS = [13, 42, 123, 777, 2026]
MODELS = ["A", "B", "graph_only"]
COMPARE = ("graph_only", "B")      # TGN-JointIDS minus each of these
LAB = {"A": "TGN-JointIDS", "B": "w/o ENS", "graph_only": "Graph-only"}
COLORS = {"graph_only": "#8c8c8c", "frozen_gru": "#5e8ec4", "A": "#1f3a68", "B": "#c8762b"}
SHORT = {"mitm": "MITM", "ransomware": "Ransomware", "dos_attacks-goldeneye": "DoS-GoldenEye",
         "dos_attacks-slowloris": "DoS-Slowloris", "dos_attacks-hulk": "DoS-Hulk",
         "ddos_attack-loic-udp": "DDoS-LOIC-UDP", "brute_force_-xss": "XSS", "sql_injection": "SQL-Inj"}
TICK = {"dos_attacks-goldeneye": "GoldenEye", "dos_attacks-slowloris": "Slowloris", "dos_attacks-hulk": "Hulk",
        "ddos_attack-loic-udp": "LOIC-UDP", "brute_force_-xss": "XSS", "sql_injection": "SQL-Inj",
        "mitm": "MITM", "ransomware": "Ransomware"}
TON = R / "nf_ton_iot_v3/ablations"
CIC = R / "nf_cicids2018_v3"
DATASETS = {
    "NF-ToN-IoT-v3": {
        "classes": ["benign", "ddos", "xss", "password", "scanning", "backdoor", "injection", "dos", "mitm",
                    "ransomware"],
        "train": lambda: pd.read_csv(TON / "report/table1_dataset.csv", index_col=0)["train_windows"]
        .drop("total").to_dict(),
        "runs": {"A": R / "nf_ton_iot_v3/tgn_jointids",
                 "frozen_gru": TON / "frozen_gru", "B": TON / "w_o_ens",
                 "graph_only": TON / "graph_only"},
        "paper": R / "nf_ton_iot_v3/paper_figures", "tag": "ton"},
    "NF-CICIDS2018-v3": {
        "classes": None,
        "train": lambda: json.load(open(ROOT / "data/processed/nf_cicids2018_v3.manifest.json"))
        ["graph_label_counts"]["train"],
        "runs": {"A": CIC / "tgn_jointids", "frozen_gru": CIC / "ablations/frozen_gru",
                 "B": CIC / "ablations/w_o_ens", "graph_only": CIC / "ablations/graph_only"},
        "paper": R / "nf_cicids2018_v3/paper_figures", "tag": "cicids"},
}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.{d}f} ± {np.nanstd(v, ddof=1):.{d}f}"


# ---------------------------------------------------------------------------- metrics on counts
def per_cluster(pred, clusters, C):
    """Per-cluster confusion matrix [K,C,C] and binary flags of each true class [K,C]."""
    keys, inv = np.unique(clusters, return_inverse=True)
    cm = np.zeros((len(keys), C, C), np.int64)
    np.add.at(cm, (inv, pred["y_multi"], pred["pred_multi"]), 1)
    flag = np.zeros((len(keys), C), np.int64)
    np.add.at(flag, (inv, pred["y_multi"]), pred["pred_bin"].astype(np.int64))
    return cm, flag


def rare_stats(cm, flag, rare):
    tp = np.diag(cm)[rare].astype(float)
    sup, prd = cm.sum(1)[rare].astype(float), cm.sum(0)[rare].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        present = sup > 0      # only classes present in the ground truth (same set for both models)
        f1 = np.where(present, 2 * tp / (sup + prd), np.nan)
        rec = np.where(present, tp / sup, np.nan)
        prec = np.where(present, np.where(prd > 0, tp / prd, 0.0), np.nan)
    return {"rare_f1": np.nanmean(f1), "rare_recall": np.nanmean(rec), "rare_precision": np.nanmean(prec),
            "rare_detection": flag[rare].sum() / sup.sum(),
            "per_class_f1": f1, "per_class_recall": rec}


def paired(Sa, Sb, rare, key, n_boot=2000, seed=0):
    K = Sa[0][0].shape[0]

    def delta(w):
        out = []
        for (ca, fa), (cb, fb) in zip(Sa, Sb):
            a = rare_stats(np.tensordot(w, ca, 1), w @ fa, rare)[key]
            b = rare_stats(np.tensordot(w, cb, 1), w @ fb, rare)[key]
            out.append(a - b)
        return np.array(out)
    full = delta(np.ones(K))
    rng = np.random.default_rng(seed)
    boots = np.array([np.nanmean(delta(np.bincount(rng.integers(0, K, K), minlength=K).astype(float)))
                      for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"mean_delta": float(full.mean()), "ci95": [float(lo), float(hi)],
            "seeds_positive": int((full > 0).sum()), "per_seed": full.round(4).tolist()}


# ---------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, per_class, paired_rows, stats = [], [], [], {}
    data = {}
    for ds, d in DATASETS.items():
        train = d["train"]()
        classes = d["classes"] or list(train)
        C = len(classes)
        tot = sum(train.values())
        rare_names = [c for c in classes if c != "benign" and train[c] / tot < 0.01]
        rare = np.array([classes.index(c) for c in rare_names])
        preds = {m: [dict(np.load(d["runs"][m] / f"seed_{s}" / "predictions_stream.npz")) for s in SEEDS]
                 for m in MODELS}
        ref = preds["A"][0]
        for m in MODELS:
            for p in preds[m]:
                assert np.array_equal(p["window_id"], ref["window_id"]) and np.array_equal(p["y_multi"], ref["y_multi"])
        t0 = int(ref["window_id"].min())
        res = {m: [rare_stats(*(x.sum(0) for x in per_cluster(p, np.zeros(len(p["y_multi"]), int), C)), rare)
                   for p in preds[m]]
               for m in MODELS}
        data[ds] = {"rare_names": rare_names, "res": res, "train": train, "tot": tot}
        for m in MODELS:
            rr = res[m]
            summary.append({"dataset": ds, "model": LAB[m],
                            "rare F1": ms([r["rare_f1"] for r in rr]),
                            "rare recall": ms([r["rare_recall"] for r in rr]),
                            "rare precision": ms([r["rare_precision"] for r in rr]),
                            "rare detection (binary head)": ms([r["rare_detection"] for r in rr])})
            for k, c in enumerate(rare_names):
                per_class.append({"dataset": ds, "class": SHORT.get(c, c), "train windows": train[c],
                                  "train share %": round(100 * train[c] / tot, 2), "model": LAB[m],
                                  "F1": ms([r["per_class_f1"][k] for r in rr]),
                                  "recall": ms([r["per_class_recall"][k] for r in rr])})
        stats[ds] = {"rare_classes": rare_names}
        for cmin in (10, 60):
            cl = (ref["window_id"] - t0) // (cmin * 60000)
            S = {m: [per_cluster(p, cl, C) for p in preds[m]] for m in MODELS}
            for b in COMPARE:
                for key in ("rare_f1", "rare_recall", "rare_precision", "rare_detection"):
                    v = paired(S["A"], S[b], rare, key)
                    stats[ds][f"{cmin}min TGN-JointIDS - {LAB[b]} {key}"] = v
                    paired_rows.append({"dataset": ds, "clusters": f"{cmin} min",
                                        "comparison": f"TGN-JointIDS − {LAB[b]}",
                                        "metric": key.replace("_", " "), "mean Δ": round(v["mean_delta"], 4),
                                        "95% CI": f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]",
                                        "CI excludes 0": v["ci95"][0] > 0 or v["ci95"][1] < 0,
                                        "seeds > 0": f"{v['seeds_positive']}/5"})
        print(ds, "rare:", rare_names)

    def table(df, name, cap, idx):
        df = df.set_index(idx)
        df.to_csv(OUT / f"{name}.csv")
        (OUT / f"{name}.tex").write_text(df.to_latex(escape=True, caption=cap, label=f"tab:{name}"))
        return df
    t1 = table(pd.DataFrame(summary), "table_rare_summary",
               "Rare attacks (< 1% of training windows): attack-type F1, recall and precision averaged over the "
               "rare classes, and the share of rare-attack windows flagged by the binary head; test partition, "
               "mean ± SD over five seeds.", ["dataset", "model"])
    table(pd.DataFrame(per_class), "table_rare_per_class",
          "Per rare class F1 and recall for every model (test, five seeds).", ["dataset", "class", "model"])
    t3 = table(pd.DataFrame(paired_rows), "table_rare_paired",
               "Paired differences TGN-JointIDS minus each ablation on the rare classes (seed-mean Δ, 95% CI "
               "from a time-cluster bootstrap).", ["dataset", "clusters", "comparison", "metric"])
    json.dump(stats, open(OUT / "statistics.json", "w"), indent=1)

    # ---------------- figure 1: per dataset summary (rows) x (metrics, per class, paired)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), gridspec_kw={"width_ratios": [1.5, 1.7, 1.3], "wspace": 0.42,
                                                                     "hspace": 0.62})
    order = ["graph_only", "B", "A"]
    for row, (ds, d) in enumerate(data.items()):
        res, rn = d["res"], d["rare_names"]
        ax = axes[row, 0]
        mets = [("rare_f1", "F1"), ("rare_recall", "Recall"), ("rare_precision", "Precision"),
                ("rare_detection", "Detected\n(binary)")]
        x = np.arange(len(mets))
        for k, m in enumerate(order):
            v = np.array([[r[key] for key, _ in mets] for r in res[m]])
            ax.bar(x + (k - 1) * 0.27, v.mean(0), 0.27, yerr=v.std(0, ddof=1), color=COLORS[m], capsize=1.5,
                   error_kw={"lw": 0.6}, label=LAB[m])
        ax.set_xticks(x, [n for _, n in mets])
        lo = min(np.mean([r[key] for r in res[m]]) for m in order for key, _ in mets)
        ax.set_ylim(max(0, lo - 0.1), 1.0)
        ax.set_ylabel("Rare-class mean (test)")
        ax.set_title(f"({'ab'[row]}1) {ds}: rare attacks ({len(rn)} classes)", fontsize=9)
        ax = axes[row, 1]
        x = np.arange(len(rn))
        for k, m in enumerate(order):
            v = np.array([r["per_class_f1"] for r in res[m]])
            ax.bar(x + (k - 1) * 0.27, v.mean(0), 0.27, yerr=v.std(0, ddof=1), color=COLORS[m], capsize=1.5,
                   error_kw={"lw": 0.6})
        ax.set_xticks(x, [f"{TICK.get(c, c)} ({100 * d['train'][c] / d['tot']:.2f}%)" for c in rn],
                      fontsize=7.5, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 (test)")
        ax.set_title(f"({'ab'[row]}2) Per rare class (train-window share)", fontsize=9)
        ax = axes[row, 2]
        rows = []
        for b in COMPARE:
            for key, nm, mk in (("rare_f1", "F1", "o"), ("rare_recall", "recall", "s")):
                rows.append((f"vs {LAB[b]}", nm, mk, stats[ds][f"60min TGN-JointIDS - {LAB[b]} {key}"]))
        for k, (cmp_, nm, mk, v) in enumerate(rows):
            y = k // 2 + (0.18 if nm == "recall" else -0.18)
            col = "#1f3a68" if nm == "F1" else "#2e8b57"
            ax.errorbar(v["mean_delta"], y, xerr=[[v["mean_delta"] - v["ci95"][0]], [v["ci95"][1] - v["mean_delta"]]],
                        fmt=mk, color=col, capsize=2.5, ms=5, label=f"rare {nm}" if k < 2 else None)
            ax.text(v["ci95"][1] + 0.004, y, f"{v['seeds_positive']}/5", va="center", fontsize=6.5)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_yticks(range(len(COMPARE)), [f"TGN-JointIDS\nvs {LAB[b]}" for b in COMPARE], fontsize=7.5)
        ax.set_ylim(len(COMPARE) - 0.5, -0.5)
        ax.set_xlabel("Δ (95% CI, 60-min cluster bootstrap)")
        ax.set_title(f"({'ab'[row]}3) Paired differences", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.92), fontsize=8.5)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_rare_attacks_ablation.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for ds, d in DATASETS.items():
        for e in ("png", "pdf"):
            shutil.copy2(OUT / f"fig_rare_attacks_ablation.{e}", d["paper"] / f"fig_{d['tag']}_rare_attacks_ablation.{e}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(t1.to_string())
    print(t3.xs("60 min", level="clusters").to_string())


if __name__ == "__main__":
    main()
