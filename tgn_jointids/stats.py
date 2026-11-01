"""Paired, dependence-aware comparison of two arms over matched seeds.

Windows overlap (5 s / 2.5 s) and labels persist across neighbouring blocks,
so graphs are not independent test samples. Uncertainty is therefore estimated
with a cluster bootstrap over contiguous time clusters (default 10 min), which
contain whole 60 s units, and is paired twice:

  * the same resampled clusters are applied to both arms and to every seed;
  * the statistic is the mean over seeds of the per-seed difference
    metric(arm_a, seed) - metric(arm_b, seed) (seeds share init + schedule).

With 5 seeds a sign/Wilcoxon test cannot reach p < 0.05 (minimum two-sided
p = 0.0625), so seed-level evidence is reported as a count of positive seeds
next to the bootstrap interval rather than as a p-value.
"""

from __future__ import annotations

import numpy as np


def cluster_ids(window_id: np.ndarray, cluster_ms: int, t0: int) -> np.ndarray:
    return (np.asarray(window_id, dtype=np.int64) - int(t0)) // int(cluster_ms)


def confusion_by_cluster(y, p, clusters, num_classes: int):
    """Returns (cluster_keys, cms [K, C, C]) with cms[k, true, pred] counts."""
    keys, inv = np.unique(clusters, return_inverse=True)
    cms = np.zeros((len(keys), num_classes, num_classes), dtype=np.int64)
    np.add.at(cms, (inv, np.asarray(y), np.asarray(p)), 1)
    return keys, cms


def macro_f1_from_cm(cm: np.ndarray, labels=None) -> float:
    """Macro-F1 over classes present in the ground truth of ``cm`` (or ``labels``)."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1).astype(float), cm.sum(0).astype(float)
    if labels is None:
        labels = np.flatnonzero(support > 0)
    denom = support[labels] + predicted[labels]
    f1 = np.where(denom > 0, 2 * tp[labels] / np.maximum(denom, 1e-12), 0.0)
    return float(f1.mean()) if len(labels) else float("nan")


def recall_from_cm(cm: np.ndarray, c: int) -> float:
    s = cm[c].sum()
    return float(cm[c, c] / s) if s else float("nan")


def paired_cluster_bootstrap(cms_a, cms_b, metric=macro_f1_from_cm, n_boot: int = 2000,
                             seed: int = 0, labels=None) -> dict:
    """cms_a / cms_b: lists over seeds of [K, C, C] arrays on the SAME clusters."""
    cms_a, cms_b = np.stack(cms_a), np.stack(cms_b)          # [S, K, C, C]
    if cms_a.shape != cms_b.shape:
        raise ValueError("Arms must be evaluated on identical clusters and seeds.")
    S, K = cms_a.shape[:2]
    if labels is None:
        labels = np.flatnonzero(cms_a[0].sum(0).sum(1) > 0)

    def stat(w):
        ca = np.tensordot(w, cms_a, axes=([0], [1]))          # [S, C, C]
        cb = np.tensordot(w, cms_b, axes=([0], [1]))
        da = np.array([metric(ca[s], labels) for s in range(S)])
        db = np.array([metric(cb[s], labels) for s in range(S)])
        return da - db

    full = stat(np.ones(K))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
        boots[b] = stat(w).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_delta": float(full.mean()), "per_seed_delta": full.tolist(),
            "seeds_positive": int((full > 0).sum()), "n_seeds": int(S),
            "seed_sd": float(full.std(ddof=1)) if S > 1 else float("nan"),
            "ci95": [float(lo), float(hi)], "boot_p_le_0": float(np.mean(boots <= 0)),
            "n_clusters": int(K), "n_boot": int(n_boot)}
