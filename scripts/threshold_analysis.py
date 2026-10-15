"""Operating-point analysis (Section 5.6, Table 9, Figure 4) from a stream run.

    python scripts/threshold_analysis.py --config configs/nf_ton_iot_v3.yaml \
        --stream-results results/nf_ton_iot_v3/seed_2026/stream_results.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _common import base_parser, load  # noqa: E402
from tgn_jointids.metrics import pr_curve, threshold_sweep  # noqa: E402
from tgn_jointids.utils import save_json  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--stream-results", required=True)
    args = p.parse_args()
    cfg, _ = load(args)
    df = pd.read_csv(args.stream_results)
    out = Path(args.stream_results).parent

    table = threshold_sweep(df["bin_true"], df["attack_prob"], cfg.evaluation.thresholds)
    table.to_csv(out / "threshold_sweep.csv", index=False)
    prec, rec, _, ap = pr_curve(df["bin_true"], df["attack_prob"])
    save_json({"average_precision": ap}, out / "pr_summary.json")
    print(table.round(4).to_string(index=False))
    print(f"AP = {ap:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    ax[0].plot(rec, prec)
    ax[0].set(xlabel="Recall", ylabel="Precision", title=f"(a) Precision–Recall, AP = {ap:.4f}")
    for col, mk in (("precision", "o"), ("recall", "s"), ("f1", "^")):
        ax[1].plot(table["threshold"], table[col], marker=mk, label=col.capitalize())
    ax[1].set(xlabel="Attack probability threshold", ylabel="Score", title="(b) Threshold sensitivity")
    ax[1].legend()
    ax[2].plot(table["threshold"], table["fpr_pct"], marker="o", label="Benign FPR")
    ax[2].plot(table["threshold"], table["fnr_pct"], marker="s", label="Attack FNR")
    ax[2].set(xlabel="Attack probability threshold", ylabel="Rate (%)", title="(c) FPR–FNR trade-off")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out / "operating_point.pdf", bbox_inches="tight")
    fig.savefig(out / "operating_point.png", dpi=300, bbox_inches="tight")
    print(f"Figures -> {out}")


if __name__ == "__main__":
    main()
