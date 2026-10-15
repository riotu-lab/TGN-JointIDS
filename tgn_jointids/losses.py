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
    return bin_w, multi_w, counts


def focal_loss_with_smoothing(logits, targets, weight=None, gamma: float = 2.0,
                              label_smoothing: float = 0.05) -> torch.Tensor:
    """(1 - p*)^γ · CE_{ENS,ε}  averaged over the batch (Eq. 15)."""
    ce = F.cross_entropy(logits, targets, weight=weight,
                         label_smoothing=label_smoothing, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def compute_joint_loss(bin_logits, multi_logits, node_logits, y_bin, y_multi, y_node,
                       bin_w, multi_w, lambda_bin: float = 1.0, lambda_multi: float = 0.7,
                       lambda_node: float = 0.2, focal_gamma: float = 2.0,
                       label_smoothing: float = 0.05):
    """L_base = λ_bin L_bin + λ_multi L_multi + λ_node L_node  (Eq. 16)."""
    bin_loss = F.cross_entropy(bin_logits, y_bin, weight=bin_w, label_smoothing=label_smoothing)
    multi_loss = focal_loss_with_smoothing(multi_logits, y_multi, weight=multi_w,
                                           gamma=focal_gamma, label_smoothing=label_smoothing)
    node_loss = F.cross_entropy(node_logits, y_node) if y_node is not None else 0
    total = lambda_bin * bin_loss + lambda_multi * multi_loss + lambda_node * node_loss
    return total, bin_loss, multi_loss, node_loss
