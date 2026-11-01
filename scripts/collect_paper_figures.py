"""Collect the paper figures (PDF + PNG) of both datasets under paper-oriented names, and draw
the NF-ToN-IoT-v3 dataset/graph figure.

    python scripts/collect_paper_figures.py

Run after the analysis scripts (see README.md, "Reproducing the paper"). Sources:
  results/nf_ton_iot_v3/report/, results/nf_ton_iot_v3/ablations/report/   (analyze_ton*.py)
  results/nf_cicids2018_v3/report/                                          (analyze_cicids.py)
  results/ablation_statistics/, results/baselines/report/                   (both datasets)
plot_design_justification.py and analyze_rare_attacks.py copy their own figures.
Writes results/nf_ton_iot_v3/paper_figures/ and results/nf_cicids2018_v3/paper_figures/.
"""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "nf_ton_iot_v3/paper_figures"
OUT_CIC = R / "nf_cicids2018_v3/paper_figures"
CC = R / "nf_ton_iot_v3/ablations/report"
A5 = R / "nf_ton_iot_v3/report"
CR = R / "nf_cicids2018_v3/report"
AS = R / "ablation_statistics"
BL = R / "baselines/report"
BUNDLE = ROOT / "data/processed/nf_ton_iot_v3.pt"
NAMES = {"benign": "Benign", "ddos": "DDoS", "xss": "XSS", "password": "Password", "scanning": "Scanning",
         "backdoor": "Backdoor", "injection": "Injection", "dos": "DoS", "mitm": "MITM", "ransomware": "Ransomware"}
COPY = {  # source -> paper name (NF-ToN-IoT-v3)
    CC / "fig2_multiclass_confusion": "fig_ton_confusion_10class",
    CC / "fig1_binary_confusion": "fig_ton_confusion_binary",
    A5 / "fig1_seed_results": "fig_ton_seed_results_minority",
    CC / "fig5_minority_comparison": "fig_ton_minority_ablation",
    CC / "fig3_memory_ablation": "fig_ton_memory_ablation",
    CC / "fig4_ens_tradeoff": "fig_ton_ens_tradeoff",
    CC / "fig6_binary_pr_threshold": "fig_ton_pr_threshold",
    CC / "fig9_convergence": "fig_ton_convergence",
    AS / "fig_ablation_summary": "fig_ton_ablation_summary",
    BL / "fig_baselines_main": "fig_ton_baselines",
    BL / "fig_baselines_per_class": "fig_ton_baselines_per_class",
}
COPY_CIC = {  # source -> paper name (NF-CICIDS2018-v3)
    CR / "fig6_dataset_and_graphs": "fig_cicids_dataset_graphs",
    CR / "fig1_confusion_15class_raw": "fig_cicids_confusion_15class",
    CR / "fig2b_confusion_binary_counts": "fig_cicids_confusion_binary",
    CR / "fig3_per_class_f1_recall": "fig_cicids_per_class_f1_recall",
    CR / "fig4_pr_and_threshold": "fig_cicids_pr_threshold",
    CR / "fig5_convergence": "fig_cicids_convergence",
    CR / "fig7_ablations": "fig_cicids_ablation_metrics",
    AS / "fig_ablation_summary": "fig_cicids_ablation_summary",
    BL / "fig_baselines_main": "fig_cicids_baselines",
    BL / "fig_baselines_per_class": "fig_cicids_baselines_per_class",
}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def dataset_figure():
    rule = json.load(open(R / "nf_cicids2018_v3/downsampling_rule.json"))
    t = pd.read_csv(CC / "table1_dataset.csv", index_col=0).drop(index="total")
    off = {c: v[0] for c, v in rule["per_class_official_derived_ratio"].items()}
    t["official"] = [off[c] for c in t.index]
    b = torch.load(BUNDLE, weights_only=False)
    g, sp = b["graphs"], b["split"]
    edges = {p: np.array([g[i].edge_index.shape[1] for i in sp[p]]) for p in ("train", "val", "stream")}
    del b, g
    names = [NAMES[c] for c in t.index]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].barh(names, t["official"], color="#bbbbbb", label="official")
    axes[0].barh(names, t["flows"], color="#1f3a68", label="used")
    axes[0].set_xscale("log")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Flows (log)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Flows per class", fontsize=9)
    yy = np.arange(len(t))
    for k, (col, cc) in enumerate(zip(("train_windows", "val_windows", "stream_windows"),
                                      ("#1f3a68", "#5e8ec4", "#c8762b"))):
        axes[1].barh(yy + (k - 1) * 0.27, t[col], 0.27, color=cc, label=col.split("_")[0])
    axes[1].set_xscale("log")
    axes[1].set_ylim(len(t) - 0.5, -0.5)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Window graphs (log)")
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    axes[1].set_title("Windows per class and partition", fontsize=9)
    for p, cc in zip(("train", "val", "stream"), ("#1f3a68", "#5e8ec4", "#c8762b")):
        axes[2].hist(np.log10(edges[p]), bins=60, histtype="step", color=cc, label=p, density=True)
    axes[2].set_xlabel("log10 edges (flows) per window graph")
    axes[2].set_ylabel("Density")
    axes[2].legend(fontsize=7)
    axes[2].set_title("Graph size", fontsize=9)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"fig_ton_dataset_graphs.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_figure()
    OUT_CIC.mkdir(parents=True, exist_ok=True)
    for copy, out in ((COPY, OUT), (COPY_CIC, OUT_CIC)):
        for src, name in copy.items():
            for e in ("png", "pdf"):
                shutil.copy2(f"{src}.{e}", out / f"{name}.{e}")
        print(sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
