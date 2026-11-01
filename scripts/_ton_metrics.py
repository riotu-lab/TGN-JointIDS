"""NF-ToN-IoT-v3 class list, per-run window metrics and the low-identity mask shared by
analyze_ton.py and analyze_ton_ablations.py."""

from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

CLASSES = ["benign", "ddos", "xss", "password", "scanning", "backdoor", "injection", "dos",
           "mitm", "ransomware"]


def metrics(pred, low_mask=None):
    y, p = pred["y_multi"], pred["pred_multi"]
    labels = list(range(10))
    P, R, F, S = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    cm = confusion_matrix(y, p, labels=labels)
    benign = pred["y_bin"] == 0
    out = {"macro_f1": f1_score(y, p, labels=labels, average="macro", zero_division=0),
           "accuracy": accuracy_score(y, p),
           "weighted_f1": f1_score(y, p, labels=labels, average="weighted", zero_division=0),
           "macro_precision": float(P.mean()), "macro_recall": float(R.mean()),
           "binary_f1": f1_score(pred["y_bin"], pred["pred_bin"], zero_division=0),
           "benign_fpr_binary_head": float(np.mean(pred["pred_bin"][benign] == 1)),
           "benign_misclass_multiclass_head": float(np.mean(p[benign] != 0)),
           "per_class": {c: {"precision": float(P[i]), "recall": float(R[i]), "f1": float(F[i]),
                             "support": int(S[i]), "fn": int(S[i] - cm[i, i]),
                             "fp": int(cm[:, i].sum() - cm[i, i]),
                             "benign_as": int(cm[0, i]) if i else 0}
                         for i, c in enumerate(CLASSES)},
           "cm": cm.tolist()}
    if low_mask is not None:
        yl, pl = y[low_mask], p[low_mask]
        out["low_identity_macro_f1"] = f1_score(yl, pl, labels=sorted(set(yl.tolist())),
                                                average="macro", zero_division=0)
    return out


def low_purity_mask(bundle_path, split: str, threshold: float = 0.6):
    """Windows whose hosts all have a training-label purity below ``threshold`` (hosts with no
    clear attack-type history; unseen hosts count as purity 0). Returns (mask, window_id, t0)."""
    from tgn_jointids.utils import env_path
    b = torch.load(env_path(bundle_path), weights_only=False)
    graphs, split_idx = b["graphs"], b["split"]
    hist = defaultdict(Counter)
    for i in split_idx["train"]:
        for n in set(graphs[i].node_ids.tolist()):
            hist[n][int(graphs[i].y_multi)] += 1
    purity = {n: max(c.values()) / sum(c.values()) for n, c in hist.items()}
    t0 = min(int(g.window_id) for g in graphs)
    ev = [graphs[i] for i in split_idx[split]]
    mask = np.array([all(purity.get(n, 0.0) < threshold for n in g.node_ids.tolist()) for g in ev])
    return mask, np.array([int(g.window_id) for g in ev]), t0
