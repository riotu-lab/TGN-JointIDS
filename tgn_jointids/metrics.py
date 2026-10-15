"""Evaluation metrics used in Section 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_recall_fscore_support)


def binary_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_attack": float(p[0]), "recall_attack": float(r[0]), "f1_attack": float(f[0]),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "benign_fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "fnr": fn / (fn + tp) if fn + tp else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def attack_type_predictions(df: pd.DataFrame, cascade: bool = True,
                            benign_id: int = 0) -> np.ndarray:
    """Cascade rule (Section 4.4): benign if the binary head says benign,
    otherwise the arg max of the attack-type head."""
    pred = df["multi_pred"].to_numpy().copy()
    if cascade:
        pred[df["bin_pred"].to_numpy() == 0] = benign_id
    return pred


def attack_only_metrics(df: pd.DataFrame, label_space, cascade: bool = True):
    """Attack-type metrics restricted to ground-truth attack windows.

    Attack windows predicted as benign are kept as errors.
    Returns (summary dict, per-class DataFrame incl. one-vs-rest TPR/FPR,
    confusion matrix over attack classes + benign column).
    """
    mask = df["bin_true"].to_numpy() == 1
    y_true = df["multi_true"].to_numpy()[mask]
    y_pred = attack_type_predictions(df, cascade)[mask]
    ids = label_space.attack_ids

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    rows = []
    for k, cid in enumerate(ids):
        pos, neg = y_true == cid, y_true != cid
        fpr = float(np.mean(y_pred[neg] == cid)) if neg.any() else float("nan")
        rows.append({"class": label_space.id2attack[cid], "precision": p[k], "recall": r[k],
                     "f1": f[k], "support": int(s[k]), "tpr": r[k], "fpr_ovr": fpr})
    per_class = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, labels=ids, average="micro", zero_division=0),
        "n_attack_windows": int(mask.sum()), "cascade": cascade,
    }
    cm = confusion_matrix(y_true, y_pred, labels=ids + [label_space.attack2id[label_space.benign_label]])
    return summary, per_class, cm[: len(ids)]


def full_classification_report(df: pd.DataFrame, label_space) -> str:
    return classification_report(df["multi_true"], df["multi_pred"],
                                 labels=list(range(label_space.num_classes)),
                                 target_names=label_space.classes, digits=4, zero_division=0)


def threshold_sweep(y_true, attack_prob, thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)):
    """Table 9: precision/recall/F1/FNR/FPR as a function of τ."""
    y_true, attack_prob = np.asarray(y_true), np.asarray(attack_prob)
    rows = []
    for t in thresholds:
        m = binary_metrics(y_true, (attack_prob >= t).astype(int))
        rows.append({"threshold": t, "precision": m["precision_attack"], "recall": m["recall_attack"],
                     "f1": m["f1_attack"], "fnr_pct": 100 * m["fnr"], "fpr_pct": 100 * m["benign_fpr"]})
    return pd.DataFrame(rows)


def pr_curve(y_true, attack_prob):
    prec, rec, thr = precision_recall_curve(y_true, attack_prob)
    return prec, rec, thr, average_precision_score(y_true, attack_prob)


def latency_summary(latencies_ms, window_ms: int = 5000) -> dict:
    lat = np.asarray(latencies_ms)
    mean, p95 = float(lat.mean()), float(np.percentile(lat, 95))
    return {"mean_ms": mean, "p95_ms": p95, "budget_mean_pct": 100 * mean / window_ms,
            "budget_p95_pct": 100 * p95 / window_ms}


def class_recall(df: pd.DataFrame, label_space, name: str, cascade: bool = True) -> float:
    cid = label_space.attack2id[name]
    mask = (df["bin_true"].to_numpy() == 1) & (df["multi_true"].to_numpy() == cid)
    if not mask.any():
        return float("nan")
    return float(np.mean(attack_type_predictions(df, cascade)[mask] == cid))
