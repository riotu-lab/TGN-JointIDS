"""Where does the focal term help? TGN-JointIDS (ENS-weighted focal loss, gamma = 2) vs w/o focal
(ENS-weighted cross-entropy, loss.focal_gamma = 0), both datasets, five paired seeds.

    python scripts/analyze_focal.py [--ablation C] [--n-boot 1000]

Beyond the standard metrics (results/ablation_statistics/all_metrics_paired.csv, comparison "w/o focal"), this
tests what focal loss is designed to change, on the test (stream) partition:
  hard windows    stealthy   = attack windows whose attack flows are < 10% of the window's flows;
                  low-identity = windows whose every host has training-label purity < 0.6
                  (host history does not identify the class); confusable = true class in a group of
                  mutually confused classes (ToN: xss/injection/password/scanning; CICIDS: web/XSS/SQL-inj)
                  -> recall (stealthy: binary detection), macro-F1 over present classes
  calibration     expected calibration error (15 bins) and Brier score, attack-type head and binary head
  convergence     validation macro-F1 at 1k/2k/3k updates; updates needed to reach 95% / 99% of the run's
                  best validation macro-F1; area under the validation curve
  stability       standard deviation over seeds of test macro-F1 and binary F1
Δ = TGN-JointIDS − ablation; test-set Δs get a 95% CI from a 60-min (and 10-min) time-cluster bootstrap.
Writes results/ablation_statistics/focal/ (CSV + LaTeX + figure).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_ablation_statistics import SEEDS, datasets  # noqa: E402

BUNDLES = {"NF-ToN-IoT-v3": ROOT / "data/processed/nf_ton_iot_v3.pt",
           "NF-CICIDS2018-v3": ROOT / "data/processed/nf_cicids2018_v3.pt"}
CONFUSABLE = {"NF-ToN-IoT-v3": ["xss", "injection", "password", "scanning"],
              "NF-CICIDS2018-v3": ["brute_force_-web", "brute_force_-xss", "sql_injection"]}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def bundle_side(path):
    b = torch.load(path, weights_only=False)
    g, sp = b["graphs"], b["split"]
    hist = defaultdict(Counter)
    for i in sp["train"]:
        for n in set(g[i].node_ids.tolist()):
            hist[n][int(g[i].y_multi)] += 1
    purity = {n: max(c.values()) / sum(c.values()) for n, c in hist.items()}
    idx = sp["stream"]
    out = {"window_id": np.array([int(g[i].window_id) for i in idx]),
           "attack_ratio": np.array([float(g[i].attack_ratio) for i in idx]),
           "low_identity": np.array([all(purity.get(n, 0.0) < 0.6 for n in g[i].node_ids.tolist()) for i in idx])}
    del b, g
    return out


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def stats(p, side, classes):
    """Metrics of one run on one (possibly weighted) sample; w = per-window weights."""
    y, yh, yb, pb = p["y_multi"], p["pred_multi"], p["y_bin"], p["pred_bin"]
    conf = p["prob_multi"].max(1)
    pa = p["prob_attack"]
    conf_set = [classes.index(c) for c in CONFUSABLE[side["ds"]] if c in classes]
    stealth = (yb == 1) & (side["attack_ratio"] < 0.10)
    low = side["low_identity"]
    conf_m = np.isin(y, conf_set)

    def macro_f1(mask):
        out = []
        for c in np.unique(y[mask]):
            tp = np.sum(mask & (y == c) & (yh == c))
            out.append(2 * tp / (np.sum(mask & (y == c)) + np.sum(mask & (yh == c))))
        return np.mean(out) if out else np.nan
    onehot = np.eye(p["prob_multi"].shape[1])[y]
    return {
        "stealthy attack windows: binary detection": pb[stealth].mean() if stealth.any() else np.nan,
        "stealthy attack windows: attack-type recall": (yh[stealth] == y[stealth]).mean() if stealth.any() else np.nan,
        "low-identity windows: macro-F1": macro_f1(low),
        "low-identity windows: accuracy": (yh[low] == y[low]).mean() if low.any() else np.nan,
        "confusable classes: recall": (yh[conf_m] == y[conf_m]).mean(),
        "confusable classes: mean F1": np.mean([2 * np.sum((y == c) & (yh == c)) / (np.sum(y == c) + np.sum(yh == c))
                                                for c in conf_set if np.sum(y == c) + np.sum(yh == c) > 0]),
        "attack-type head: ECE ↓": ece(conf, (yh == y).astype(float)),
        "attack-type head: Brier ↓": np.mean(np.sum((p["prob_multi"] - onehot) ** 2, 1)),
        "binary head: ECE ↓": ece(np.maximum(pa, 1 - pa), (pb == yb).astype(float)),
        "binary head: Brier ↓": np.mean((pa - yb) ** 2),
    }


def boot_ci(PA, PB, side, classes, cmin, n_boot, seed=0):
    cl = (side["window_id"] - side["window_id"].min()) // (cmin * 60000)
    keys, inv = np.unique(cl, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx_by = [np.flatnonzero(inv == k) for k in range(len(keys))]
    boots = defaultdict(list)
    for _ in range(n_boot):
        pick = np.concatenate([idx_by[k] for k in rng.integers(0, len(keys), len(keys))])
        sub = {k: v[pick] for k, v in side.items() if isinstance(v, np.ndarray)} | {"ds": side["ds"]}
        d = defaultdict(list)
        for a, b in zip(PA, PB):
            sa = stats({k: v[pick] for k, v in a.items() if np.ndim(v) >= 1}, sub, classes)
            sb = stats({k: v[pick] for k, v in b.items() if np.ndim(v) >= 1}, sub, classes)
            for k in sa:
                d[k].append(sa[k] - sb[k])
        for k, v in d.items():
            boots[k].append(np.nanmean(v))
    return {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in boots.items()}


def curve(path):
    h = json.load(open(path / "history.json"))
    u, f = np.array(h["update"]), np.array(h["val_macro_f1"])
    best = f.max()
    first = lambda q: int(u[np.argmax(f >= q * best)])  # noqa: E731
    at = lambda U: float(f[u == U][0]) if (u == U).any() else np.nan  # noqa: E731
    return {"val macro-F1 at 1,000 updates": at(1000), "val macro-F1 at 2,000 updates": at(2000),
            "val macro-F1 at 3,000 updates": at(3000), "updates to 95% of best val macro-F1 ↓": first(0.95),
            "updates to 99% of best val macro-F1 ↓": first(0.99),
            "area under val curve (mean val macro-F1)": float(f.mean())}, (u, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="C", help="C = w/o focal (B: dry run with w/o ENS)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default=str(ROOT / "results/ablation_statistics/focal"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for ds, d in datasets().items():
        classes = d["classes"]
        side = bundle_side(BUNDLES[ds]) | {"ds": ds}
        PA = [dict(np.load(d["runs"]["A"] / f"seed_{s}/predictions_stream.npz")) for s in SEEDS]
        PB = [dict(np.load(d["runs"][args.ablation] / f"seed_{s}/predictions_stream.npz")) for s in SEEDS]
        for p in PA + PB:
            assert np.array_equal(p["window_id"], side["window_id"])
        stealth_n = int(((PA[0]["y_bin"] == 1) & (side["attack_ratio"] < 0.10)).sum())
        conf_n = int(np.isin(PA[0]["y_multi"], [classes.index(c) for c in CONFUSABLE[ds] if c in classes]).sum())
        nwin = {"stealthy": stealth_n, "low-identity": int(side["low_identity"].sum()), "confusable": conf_n}
        SA = [stats(p, side, classes) for p in PA]
        SB = [stats(p, side, classes) for p in PB]
        ci = {c: boot_ci(PA, PB, side, classes, c, args.n_boot) for c in (10, 60)}
        for k in SA[0]:
            a, b = np.array([x[k] for x in SA]), np.array([x[k] for x in SB])
            lower = "↓" in k
            dlt = a - b
            fav = int(np.sum((-dlt if lower else dlt) > 0))
            lo10, hi10 = ci[10][k]
            lo60, hi60 = ci[60][k]
            sign = -1 if lower else 1
            sig = all(sign * lo > 0 and sign * hi > 0 for lo, hi in ((lo10, hi10), (lo60, hi60)))
            loss = all(sign * lo < 0 and sign * hi < 0 for lo, hi in ((lo10, hi10), (lo60, hi60)))
            n = next((v for kk, v in nwin.items() if k.startswith(kk)), int(len(side["window_id"])))
            rows.append({"dataset": ds, "metric": k, "windows": n, "TGN-JointIDS": round(a.mean(), 4), "w/o focal": round(b.mean(), 4),
                         "Δ": round(dlt.mean(), 4), "seeds favourable": f"{fav}/5",
                         "95% CI (10 min)": f"[{lo10:.4f}, {hi10:.4f}]", "95% CI (60 min)": f"[{lo60:.4f}, {hi60:.4f}]",
                         "verdict": "significant gain" if sig else "significant loss" if loss else
                         ("consistent gain (5/5 seeds; CI includes 0)" if fav == 5 else "no clear difference")})
        # convergence and stability (per-seed, no time-cluster CI)
        ca, cb = [], []
        for s in SEEDS:
            x, ua = curve(d["runs"]["A"] / f"seed_{s}")
            y, ub = curve(d["runs"][args.ablation] / f"seed_{s}")
            ca.append(x)
            cb.append(y)
            curves.setdefault(ds, []).append((ua, ub))
        for k in ca[0]:
            a, b = np.array([x[k] for x in ca], float), np.array([x[k] for x in cb], float)
            lower = "↓" in k
            dlt = a - b
            fav = int(np.sum((-dlt if lower else dlt) > 0))
            rows.append({"dataset": ds, "metric": k, "windows": "val curve", "TGN-JointIDS": round(np.nanmean(a), 4),
                         "w/o focal": round(np.nanmean(b), 4), "Δ": round(np.nanmean(dlt), 4),
                         "seeds favourable": f"{fav}/5", "95% CI (10 min)": "per seed", "95% CI (60 min)": "per seed",
                         "verdict": "gain on 5/5 seeds" if fav == 5 else "gain on 4/5 seeds" if fav == 4 else
                         "no clear difference" if fav >= 2 else "worse on ≥ 4/5 seeds"})
        mf = lambda P: [np.mean([  # noqa: E731
            2 * np.sum((p["y_multi"] == c) & (p["pred_multi"] == c)) /
            (np.sum(p["y_multi"] == c) + np.sum(p["pred_multi"] == c)) for c in np.unique(p["y_multi"])]) for p in P]
        sa, sb = np.std(mf(PA), ddof=1), np.std(mf(PB), ddof=1)
        rows.append({"dataset": ds, "metric": "seed SD of test macro-F1 ↓", "windows": int(len(side["window_id"])), "TGN-JointIDS": round(sa, 4),
                     "w/o focal": round(sb, 4), "Δ": round(sa - sb, 4), "seeds favourable": "—",
                     "95% CI (10 min)": "—", "95% CI (60 min)": "—",
                     "verdict": "more stable" if sa < sb else "less stable"})
        print("done", ds, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "focal_specific_metrics.csv", index=False)
    (out / "focal_specific_metrics.tex").write_text(df.set_index(["dataset", "metric"]).to_latex(
        escape=True, caption="What the focal term changes: TGN-JointIDS (ENS-weighted focal loss) vs ENS-weighted "
        "cross-entropy, five paired seeds (test partition unless stated).", label="tab:focal"))
    # figure: validation curves (mean over seeds) for both datasets
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (ds, cv) in zip(axes, curves.items()):
        for k, (lab, col) in enumerate((("TGN-JointIDS (focal)", "#1f3a68"), ("w/o focal (cross-entropy)", "#7b5ea7"))):
            u = cv[0][k][0]
            f = np.array([c[k][1][:len(u)] for c in cv])
            ax.plot(u, f.mean(0), color=col, lw=1.6, label=lab)
            ax.fill_between(u, f.min(0), f.max(0), color=col, alpha=0.15)
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("Validation macro-F1")
        ax.set_title(ds, fontsize=10)
        lo = min(np.array([c[k][1] for c in cv for k in (0, 1)], dtype=object)[0].min(), 0.5)
        ax.set_ylim(max(lo, 0.5), None)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Convergence with and without the focal term (mean of five seeds; band = seed range)", fontsize=10)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"fig_focal_convergence.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
