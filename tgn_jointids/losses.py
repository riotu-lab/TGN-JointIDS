"""Long-tail-aware multi-task objective (Section 3.5, Eqs. 14-16)."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def compute_ens_weights(class_counts: Sequence[float], beta: float = 0.9999,
                        device="cpu") -> torch.Tensor:
    """Effective-Number-of-Samples weights (Cui et al., CVPR 2019), Eq. 14.

    E_c = (1 - β^{n_c}) / (1 - β);  w_c ∝ 1 / E_c;  normalised to mean 1.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.where(counts <= 0, 1.0, counts)
    eff_num = (1.0 - beta ** counts) / (1.0 - beta)
    w = 1.0 / eff_num
    w = w / (w.mean() + 1e-12)
    return torch.tensor(w, dtype=torch.float32, device=device)


def cap_weight_ratio(weights: torch.Tensor, max_ratio: float = 30.0) -> torch.Tensor:
    """Clamp so that max(w) / min(w) <= max_ratio."""
    return torch.clamp(weights, max=weights.min() * max_ratio)


def class_weights_from_config(cfg_loss, label_space, train_graphs, device):
    """Build (binary, multi-class) ENS weights.

    multi-class source:
      "flow_counts"  -> pre-balancing flow counts given in the config
                        (NF-ToN-IoT-v3 setting; preserves true rarity)
      "train_graphs" -> graph-level label counts on the training split
                        (NF-CICIDS2018-v3 setting)
    """
    bin_counter = Counter(int(g.y_bin.item()) for g in train_graphs)
    bin_counts = np.array([bin_counter[i] for i in range(2)], dtype=np.float64)
    bin_w = compute_ens_weights(bin_counts + 1, beta=cfg_loss.ens_beta_binary, device=device)

    if cfg_loss.ens_source == "flow_counts":
        counts = np.array([cfg_loss.original_flow_counts[c] for c in label_space.classes],
                          dtype=np.float64)
    elif cfg_loss.ens_source == "train_graphs":
        c = Counter(int(g.y_multi.item()) for g in train_graphs)
        counts = np.array([c.get(i, 0) for i in range(label_space.num_classes)], dtype=np.float64)
    else:
        raise ValueError(f"Unknown ens_source: {cfg_loss.ens_source}")

    multi_w = compute_ens_weights(counts, beta=cfg_loss.ens_beta, device=device)
    multi_w = cap_weight_ratio(multi_w, cfg_loss.ens_cap_ratio)
    # Ablation switch ``loss.multi_weighting``: "ens" (default) or "none" (unit
    # weights for the attack-type loss only; the binary weights are unchanged).
    weighting = cfg_loss.get("multi_weighting", "ens") if hasattr(cfg_loss, "get") else "ens"
    if weighting == "none":
        multi_w = torch.ones_like(multi_w)
    elif weighting != "ens":
        raise ValueError(f"Unknown loss.multi_weighting: {weighting}")
    return bin_w, multi_w, counts


def focal_loss_with_smoothing(logits, targets, weight=None, gamma: float = 2.0,
                              label_smoothing: float = 0.05) -> torch.Tensor:
    """Legacy focal loss (``loss.focal_impl: legacy``), as used for the reported runs.

    NOTE: ``pt = exp(-ce)`` is computed from the class-WEIGHTED, label-SMOOTHED
    cross-entropy, i.e. pt = exp(-w_y · CE_ε), which is not the model probability
    p* of Eq. 15. The modulating factor therefore re-applies the class weights.
    Kept unchanged for reproducibility; see ``focal_loss_standard``.
    """
    ce = F.cross_entropy(logits, targets, weight=weight,
                         label_smoothing=label_smoothing, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def focal_loss_standard(logits, targets, weight=None, gamma: float = 2.0,
                        label_smoothing: float = 0.05) -> torch.Tensor:
    """w_y · (1 - p*)^γ · CE_ε  averaged over the batch (Eq. 15, ``focal_impl: standard``).

    p* = softmax(logits)[y] is the unsmoothed model probability of the true
    class; the class weight w_y is applied exactly once.
    """
    log_p = F.log_softmax(logits, dim=1)
    p_true = log_p.gather(1, targets.view(-1, 1)).squeeze(1).exp()
    ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing, reduction="none")
    loss = (1.0 - p_true) ** gamma * ce
    if weight is not None:
        loss = weight[targets] * loss
    return loss.mean()


FOCAL_IMPLS = {"legacy": focal_loss_with_smoothing, "standard": focal_loss_standard}


def compute_joint_loss(bin_logits, multi_logits, node_logits, y_bin, y_multi, y_node,
                       bin_w, multi_w, lambda_bin: float = 1.0, lambda_multi: float = 0.7,
                       lambda_node: float = 0.2, focal_gamma: float = 2.0,
                       label_smoothing: float = 0.05, focal_impl: str = "legacy"):
    """L_base = λ_bin L_bin + λ_multi L_multi + λ_node L_node  (Eq. 16)."""
    if focal_impl not in FOCAL_IMPLS:
        raise ValueError(f"Unknown focal_impl: {focal_impl}")
    bin_loss = F.cross_entropy(bin_logits, y_bin, weight=bin_w, label_smoothing=label_smoothing)
    multi_loss = FOCAL_IMPLS[focal_impl](multi_logits, y_multi, weight=multi_w,
                                         gamma=focal_gamma, label_smoothing=label_smoothing)
    node_loss = F.cross_entropy(node_logits, y_node) if y_node is not None else 0
    total = lambda_bin * bin_loss + lambda_multi * multi_loss + lambda_node * node_loss
    return total, bin_loss, multi_loss, node_loss
