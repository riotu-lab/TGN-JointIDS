"""Where does TGN-JointIDS improve detection? Systematic paired check of every metric.

    python scripts/analyze_ablation_statistics.py [--n-boot 1000]

For both datasets (NF-ToN-IoT-v3; NF-CICIDS2018-v3 block-stratified split) and both
comparisons (TGN-JointIDS vs Graph-only = no temporal memory; TGN-JointIDS vs w/o ENS = focal
loss without ENS class weights; TGN-JointIDS vs w/o focal = ENS-weighted cross-entropy, i.e.
loss.focal_gamma=0), on the test (stream) partition, five paired seeds:
  multiclass  accuracy, macro-F1, weighted F1 (attack-type head, raw arg max)
  binary      accuracy, precision, recall, F1, FPR, FNR (binary head, threshold 0.5), AP (no CI)
  rare        mean F1 / recall / precision over classes with < 1% of training windows, and the
              share of rare-class windows flagged by the binary head
  per class   F1 and recall of every attack class
Δ = TGN-JointIDS minus the ablation (seed mean). 95% CIs from a cluster bootstrap over 10- and
60-min time clusters (the same resampled clusters for both models and all seeds). Verdict:
  significant gain  favourable Δ and BOTH CIs exclude 0 in the favourable direction
  consistent gain   favourable on 5/5 seeds, but a CI includes 0
  significant loss  both CIs exclude 0 in the unfavourable direction
  no clear difference otherwise
(For FPR and FNR lower is better.) Writes results/ablation_statistics/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "ablation_statistics"
SEEDS = [13, 42, 123, 777, 2026]
TON = R / "nf_ton_iot_v3/ablations"
CIC = R / "nf_cicids2018_v3"
LAB = {"graph_only": "Graph-only", "B": "w/o ENS", "C": "w/o focal"}
SHORT = {"benign": "Benign", "ddos": "DDoS", "xss": "XSS", "password": "Password", "scanning": "Scanning",
         "backdoor": "Backdoor", "injection": "Injection", "dos": "DoS", "mitm": "MITM", "ransomware": "Ransomware",
         "ftp-bruteforce": "FTP-BF", "ssh-bruteforce": "SSH-BF", "dos_attacks-goldeneye": "DoS-GoldenEye",
         "dos_attacks-slowloris": "DoS-Slowloris", "dos_attacks-slowhttptest": "DoS-SlowHTTP",
         "dos_attacks-hulk": "DoS-Hulk", "ddos_attacks-loic-http": "DDoS-LOIC-HTTP",
         "ddos_attack-loic-udp": "DDoS-LOIC-UDP", "ddos_attack-hoic": "DDoS-HOIC", "brute_force_-web": "Web-BF",
         "brute_force_-xss": "XSS", "sql_injection": "SQL-Inj", "infilteration": "Infiltration", "bot": "Bot"}
LOWER_BETTER = {"binary FPR", "binary FNR"}


def datasets():
    ton_train = pd.read_csv(TON / "report/table1_dataset.csv", index_col=0)["train_windows"].drop("total").to_dict()
    cic_train = json.load(open(ROOT / "data/processed/nf_cicids2018_v3.manifest.json"))[
        "graph_label_counts"]["train"]
    return {
        "NF-ToN-IoT-v3": {"train": ton_train, "classes": list(ton_train),
                          "runs": {"A": R / "nf_ton_iot_v3/tgn_jointids",
                                   "B": TON / "w_o_ens", "graph_only": TON / "graph_only",
                                   "C": TON / "w_o_focal"}},
        "NF-CICIDS2018-v3": {"train": cic_train, "classes": list(cic_train),
                             "runs": {"A": CIC / "tgn_jointids", "B": CIC / "ablations/w_o_ens",
                                      "graph_only": CIC / "ablations/graph_only",
                                      "C": CIC / "ablations/w_o_focal"}},
    }


def suff(p, clusters, C):
    """Per-cluster sufficient statistics: confusion [K,C,C], binary confusion [K,2,2], flagged per class [K,C]."""
    _, inv = np.unique(clusters, return_inverse=True)
    K = inv.max() + 1
    cm = np.zeros((K, C, C))
    np.add.at(cm, (inv, p["y_multi"], p["pred_multi"]), 1)
    bcm = np.zeros((K, 2, 2))
    np.add.at(bcm, (inv, p["y_bin"], p["pred_bin"]), 1)
    fl = np.zeros((K, C))
    np.add.at(fl, (inv, p["y_multi"]), p["pred_bin"])
    return cm, bcm, fl


def metrics(cm, bcm, fl, classes, rare):
    with np.errstate(invalid="ignore", divide="ignore"):
        tp = np.diag(cm)
        sup, prd = cm.sum(1), cm.sum(0)
        present = sup > 0      # average only over classes present in the ground truth (same set for both models)
        f1 = np.where(present, 2 * tp / (sup + prd), np.nan)
        rec = np.where(present, tp / sup, np.nan)
        prec = np.where(present, np.where(prd > 0, tp / prd, 0.0), np.nan)
        n = cm.sum()
        (tn, fp), (fn, btp) = bcm
        out = {"multiclass accuracy": tp.sum() / n, "multiclass macro-F1": np.nanmean(f1),
               "multiclass weighted F1": np.nansum(f1 * sup) / sup.sum(),
               "binary accuracy": (tn + btp) / bcm.sum(), "binary precision": btp / (btp + fp),
               "binary recall": btp / (btp + fn), "binary F1": 2 * btp / (2 * btp + fp + fn),
               "binary FPR": fp / (fp + tn), "binary FNR": fn / (fn + btp),
               "rare F1": np.nanmean(f1[rare]), "rare recall": np.nanmean(rec[rare]),
               "rare precision": np.nanmean(prec[rare]), "rare detection (binary head)": fl[rare].sum() / sup[rare].sum()}
        for i, c in enumerate(classes):
            if i:
                out[f"{SHORT[c]} F1"] = f1[i]
                out[f"{SHORT[c]} recall"] = rec[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, d in datasets().items():
        classes, train = d["classes"], d["train"]
        C, tot = len(classes), sum(train.values())
        rare = np.array([i for i, c in enumerate(classes) if i and train[c] / tot < 0.01])
        P = {m: [dict(np.load(d["runs"][m] / f"seed_{s}/predictions_stream.npz")) for s in SEEDS] for m in d["runs"]}
        ref = P["A"][0]
        for m in P:
            for p in P[m]:
                assert np.array_equal(p["window_id"], ref["window_id"]) and np.array_equal(p["y_multi"], ref["y_multi"])
        t0 = int(ref["window_id"].min())
        point = {m: [metrics(*(x.sum(0) for x in suff(p, np.zeros(len(p["y_multi"]), int), C)), classes, rare)
                     for p in P[m]] for m in P}
        apv = {m: [average_precision_score(p["y_bin"], p["prob_attack"]) for p in P[m]] for m in P}
        names = list(point["A"][0])
        ci = {}
        for cmin in (10, 60):
            cl = (ref["window_id"] - t0) // (cmin * 60000)
            S = {m: [suff(p, cl, C) for p in P[m]] for m in P}
            K = S["A"][0][0].shape[0]
            rng = np.random.default_rng(0)
            boots = {b: {k: [] for k in names} for b in LAB}
            for _ in range(args.n_boot):
                w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
                agg = {m: [metrics(*(np.tensordot(w, x, 1) for x in S[m][s]), classes, rare) for s in range(5)]
                       for m in P}
                for b in LAB:
                    for k in names:
                        boots[b][k].append(np.nanmean([agg["A"][s][k] - agg[b][s][k] for s in range(5)]))
            for b in LAB:
                for k in names:
                    v = np.array(boots[b][k], float)
                    v = v[np.isfinite(v)]
                    ci[(b, k, cmin)] = np.percentile(v, [2.5, 97.5]) if len(v) > 0.95 * args.n_boot else (np.nan, np.nan)
        for b in LAB:
            for k in names + ["binary average precision"]:
                if k == "binary average precision":
                    da, db = apv["A"], apv[b]
                else:
                    da, db = [x[k] for x in point["A"]], [x[k] for x in point[b]]
                delta = np.array(da, float) - np.array(db, float)
                sign = -1 if k in LOWER_BETTER else 1
                fav = int(np.sum(sign * delta > 0))
                lo10, hi10 = ci.get((b, k, 10), (np.nan, np.nan))
                lo60, hi60 = ci.get((b, k, 60), (np.nan, np.nan))
                both = [(lo10, hi10), (lo60, hi60)]
                if all(np.isfinite(x) for x in (lo10, hi10, lo60, hi60)) and all(
                        (sign * lo > 0 and sign * hi > 0) for lo, hi in both):
                    verdict = "significant gain"
                elif all(np.isfinite(x) for x in (lo10, hi10, lo60, hi60)) and all(
                        (sign * lo < 0 and sign * hi < 0) for lo, hi in both):
                    verdict = "significant loss"
                elif fav == 5 and np.nanmean(sign * delta) > 0:
                    verdict = "consistent gain (5/5 seeds; CI includes 0)"
                else:
                    verdict = "no clear difference"
                group = ("per class" if k.split()[0] in SHORT.values() and not k.startswith("rare") else
                         k.split()[0])
                rows.append({"dataset": ds, "comparison": f"TGN-JointIDS vs {LAB[b]}", "group": group, "metric": k,
                             "TGN-JointIDS": round(float(np.nanmean(da)), 4), "ablation": round(float(np.nanmean(db)), 4),
                             "Δ": round(float(np.nanmean(delta)), 4), "seeds favourable": f"{fav}/5",
                             "95% CI (10 min)": f"[{lo10:.4f}, {hi10:.4f}]" if np.isfinite(lo10) else "n/a",
                             "95% CI (60 min)": f"[{lo60:.4f}, {hi60:.4f}]" if np.isfinite(lo60) else "n/a",
                             "verdict": verdict})
        print("done", ds, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_metrics_paired.csv", index=False)
    gains = df[df["verdict"].str.contains("gain")].copy()
    order = {"significant gain": 0, "consistent gain (5/5 seeds; CI includes 0)": 1}
    gains = gains.sort_values(["dataset", "comparison", "verdict", "group"], key=lambda s: s.map(order) if s.name == "verdict" else s)
    gains.to_csv(OUT / "where_tgn_improves.csv", index=False)
    cols = ["metric", "TGN-JointIDS", "ablation", "Δ", "seeds favourable", "95% CI (10 min)", "95% CI (60 min)", "verdict"]
    tex = gains.set_index(["dataset", "comparison"])[cols].to_latex(
        escape=True, caption="Metrics on which TGN-JointIDS improves on its ablations (test partition, five paired "
        "seeds; 95\\% CIs from 10- and 60-min cluster bootstraps; lower is better for FPR/FNR).",
        label="tab:ablation_statistics")
    (OUT / "where_tgn_improves.tex").write_text(tex)
    losses = df[df["verdict"] == "significant loss"]
    losses.to_csv(OUT / "where_tgn_is_worse.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_columns", 20)
    print(gains[["dataset", "comparison", "metric", "TGN-JointIDS", "ablation", "Δ", "seeds favourable",
                 "95% CI (10 min)", "95% CI (60 min)", "verdict"]].to_string(index=False))
    print("\nSIGNIFICANT LOSSES")
    print(losses[["dataset", "comparison", "metric", "TGN-JointIDS", "ablation", "Δ", "seeds favourable",
                  "95% CI (10 min)", "95% CI (60 min)"]].to_string(index=False))


if __name__ == "__main__":
    main()
