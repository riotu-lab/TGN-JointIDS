"""Base trainer: optimiser, joint three-head loss, model-selection bookkeeping and early
stopping (used by ``tbptt.TBPTTTrainer``)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score

from .losses import compute_joint_loss


class EarlyStopping:
    """Stops when validation loss has not improved by ``min_delta`` for
    ``patience`` epochs."""

    def __init__(self, patience: int = 20, min_delta: float = 5e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.counter, self.stop = float("inf"), 0, False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.counter = val_loss, 0
        else:
            self.counter += 1
            self.stop = self.counter >= self.patience


class JointTrainer:
    """Optimiser (AdamW), ReduceLROnPlateau scheduler (epoch mode only; its ``min_lr`` is also the
    floor of the fixed-budget cosine schedule), joint loss of the binary, attack-type and node
    heads, and best-checkpoint bookkeeping (``selection_metric``: "val_macro_f1" or "val_acc")."""

    def __init__(self, model, tgn_memory, device, bin_weights, multi_weights,
                 loss_cfg, lr: float = 1e-3, weight_decay: float = 1e-4, min_lr: float = 1e-5,
                 scheduler_patience: int = 10, early_stopping_patience: int = 20,
                 early_stopping_min_delta: float = 5e-4, grad_clip: float = 1.0,
                 selection_metric: str = "val_macro_f1"):
        self.model = model.to(device)
        self.tgn_memory = tgn_memory
        self.device = device
        self.bin_weights = bin_weights.to(device)
        self.multi_weights = multi_weights.to(device)
        self.loss_kw = dict(lambda_bin=loss_cfg.lambda_bin, lambda_multi=loss_cfg.lambda_multi,
                            lambda_node=loss_cfg.lambda_node, focal_gamma=loss_cfg.focal_gamma,
                            label_smoothing=loss_cfg.label_smoothing,
                            focal_impl=loss_cfg.get("focal_impl", "legacy"))
        self.grad_clip = grad_clip
        self.es_patience, self.es_min_delta = early_stopping_patience, early_stopping_min_delta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=scheduler_patience, min_lr=min_lr)
        if selection_metric not in ("val_acc", "val_macro_f1"):
            raise ValueError(f"Unknown selection_metric: {selection_metric}")
        self.selection_metric = selection_metric
        self.best_state, self.best_acc, self.best_epoch = None, 0.0, -1
        self.best_score = 0.0 if selection_metric == "val_acc" else -1.0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [],
                        "val_macro_f1": [], "train_static_acc": [], "train_static_macro_f1": []}

    def _loss(self, data, bin_l, multi_l, node_l):
        y_node = data.y_node.view(-1) if hasattr(data, "y_node") else None
        return compute_joint_loss(bin_l, multi_l, node_l, data.y_bin.view(-1),
                                  data.y_multi.view(-1), y_node,
                                  self.bin_weights, self.multi_weights, **self.loss_kw)[0]

    @staticmethod
    def _macro_f1(y, p) -> float:
        return float(f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0))
