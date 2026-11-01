"""Tree baselines on per-window feature vectors (scripts/baseline_features.py): XGBoost and
Random Forest, one seed per call.

    python scripts/train_baseline_tree.py --feats data/processed/baselines/<dataset>_winfeat.npz \
        --model xgboost --seed 13 --out results/baselines/<dataset>/xgboost

Same windows, split and labels as TGN-JointIDS. Classes are balanced with inverse-frequency
sample weights. Hyper-parameters come from a small grid chosen on validation macro-F1 only
(XGBoost: early stopping on validation log-loss as well); the test (stream) partition is
predicted once with the selected model. The binary decision is P(attack) = 1 - P(benign) >= 0.5.
Writes <out>/seed_<s>/: predictions_val.npz, predictions_stream.npz (same keys and window
order as TGN-JointIDS), run_info.json (grid, selection, timings, single-window latency).
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

GRIDS = {"xgboost": {"max_depth": [6, 10], "learning_rate": [0.1], "min_child_weight": [1, 5]},
         "random_forest": {"max_depth": [None, 24], "min_samples_leaf": [1, 3]}}


def make(model, hp, seed, C):
    if model == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=2000, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             device="cuda", objective="multi:softprob", num_class=C, eval_metric="mlogloss",
                             early_stopping_rounds=50, random_state=seed, **hp)
    return RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced_subsample",
                                  n_jobs=8, random_state=seed, **hp)


def full_proba(clf, X, C):
    """predict_proba over all C classes (a class absent from training gets probability 0)."""
    p = clf.predict_proba(X)
    out = np.zeros((len(X), C), np.float32)
    out[:, clf.classes_] = p
    return out


def predictions(prob, y_multi, y_bin, wid):
    pa = 1.0 - prob[:, 0]
    return {"prob_multi": prob, "prob_attack": pa, "y_multi": y_multi, "y_bin": y_bin,
            "pred_multi": prob.argmax(1), "pred_bin": (pa >= 0.5).astype(int), "window_id": wid,
            "loss": np.array(np.nan)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--model", required=True, choices=list(GRIDS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = np.load(args.feats)
    C = len(d["classes"])
    Xtr, ytr = d["train_X"], d["train_y_multi"]
    Xva, yva = d["val_X"], d["val_y_multi"]
    w = compute_sample_weight("balanced", ytr)
    labels = list(range(C))
    grid = GRIDS[args.model]
    trials, best = [], None
    t_start = time.perf_counter()
    for vals in itertools.product(*grid.values()):
        hp = dict(zip(grid, vals))
        clf = make(args.model, hp, args.seed, C)
        t0 = time.perf_counter()
        if args.model == "xgboost":
            clf.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], verbose=False)
        else:
            clf.fit(Xtr, ytr)
        f = f1_score(yva, full_proba(clf, Xva, C).argmax(1), labels=labels, average="macro", zero_division=0)
        trial = {**hp, "val_macro_f1": float(f), "fit_seconds": time.perf_counter() - t0}
        if args.model == "xgboost":
            trial["best_iteration"] = int(clf.best_iteration)
        trials.append(trial)
        print(trial, flush=True)
        if best is None or f > best[0]:
            best = (f, hp, clf)
    t_total = time.perf_counter() - t_start
    _, hp, clf = best
    run = Path(args.out) / f"seed_{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    for part in ("val", "stream"):
        pr = predictions(full_proba(clf, d[f"{part}_X"], C), d[f"{part}_y_multi"], d[f"{part}_y_bin"],
                         d[f"{part}_window_id"])
        np.savez_compressed(run / f"predictions_{part}.npz", **pr)
    Xs = d["stream_X"][:2000]
    lat = []
    for i in range(len(Xs)):          # one window per call, as in streaming
        t0 = time.perf_counter()
        clf.predict_proba(Xs[i:i + 1])
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat[20:])
    info = {"model": args.model, "seed": args.seed, "feats": args.feats, "selected": hp,
            "best_val_macro_f1": float(best[0]), "grid": trials, "train_seconds_total": t_total,
            "latency": {"windows": int(len(lat)), "mean_ms": float(lat.mean()), "median_ms": float(np.median(lat)),
                        "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)),
                        "note": "model inference only, one window per call; feature summary not included"}}
    json.dump(info, open(run / "run_info.json", "w"), indent=1)
    print("selected", hp, "val macro-F1", round(best[0], 4))


if __name__ == "__main__":
    main()
