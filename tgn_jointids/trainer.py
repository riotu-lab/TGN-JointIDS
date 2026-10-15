"""Offline training, memory warm-up and incremental adaptation."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

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
    """Joint three-head trainer with TGN memory write-back.

    Model selection: the checkpoint with the best validation attack-type
    accuracy is restored at the end of training; early stopping and the
    ReduceLROnPlateau scheduler monitor validation loss.
    """

    def __init__(self, model, tgn_memory, device, bin_weights, multi_weights,
                 loss_cfg, lr: float = 1e-3, weight_decay: float = 1e-4, min_lr: float = 1e-5,
                 scheduler_patience: int = 10, early_stopping_patience: int = 20,
                 early_stopping_min_delta: float = 5e-4, grad_clip: float = 1.0,
                 replay_buffer=None, replay_batch: int = 64):
        self.model = model.to(device)
        self.tgn_memory = tgn_memory
        self.device = device
        self.bin_weights = bin_weights.to(device)
        self.multi_weights = multi_weights.to(device)
        self.loss_kw = dict(lambda_bin=loss_cfg.lambda_bin, lambda_multi=loss_cfg.lambda_multi,
                            lambda_node=loss_cfg.lambda_node, focal_gamma=loss_cfg.focal_gamma,
                            label_smoothing=loss_cfg.label_smoothing)
        self.grad_clip = grad_clip
        self.es_patience, self.es_min_delta = early_stopping_patience, early_stopping_min_delta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=scheduler_patience, min_lr=min_lr)
        self.replay_buffer = replay_buffer
        self.replay_batch = replay_batch
        self.ewc = None
        self.best_state, self.best_acc = None, 0.0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    # ------------------------------------------------------------------
    def _loss(self, data, bin_l, multi_l, node_l):
        y_node = data.y_node.view(-1) if hasattr(data, "y_node") else None
        return compute_joint_loss(bin_l, multi_l, node_l, data.y_bin.view(-1),
                                  data.y_multi.view(-1), y_node,
                                  self.bin_weights, self.multi_weights, **self.loss_kw)[0]

    def _step(self, data):
        """One optimisation step; memory written back AFTER the update
        from detached node embeddings."""
        data = data.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        bin_l, multi_l, node_l, node_emb, nids = self.model(data, self.tgn_memory)
        loss = self._loss(data, bin_l, multi_l, node_l)
        if self.ewc is not None:
            loss = loss + self.ewc.penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if nids is not None:
            self.tgn_memory.update_with_gru(nids, node_emb.detach(),
                                            self.model.memory_updater, self.device)
        return loss.item(), multi_l.argmax(1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for data in loader:
            data = data.to(self.device)
            bin_l, multi_l, node_l, _, _ = self.model(data, self.tgn_memory)
            total_loss += self._loss(data, bin_l, multi_l, node_l).item() * data.num_graphs
            correct += (multi_l.argmax(1) == data.y_multi.view(-1)).sum().item()
            n += data.num_graphs
        return total_loss / max(n, 1), correct / max(n, 1)

    def train(self, train_loader, val_loader, epochs: int = 150, verbose: bool = True):
        es = EarlyStopping(self.es_patience, self.es_min_delta)
        self.tgn_memory.reset()
        for epoch in range(epochs):
            tot, correct, n = 0.0, 0, 0
            for data in train_loader:
                loss, preds = self._step(data)
                tot += loss * data.num_graphs
                correct += (preds == data.y_multi.view(-1).to(self.device)).sum().item()
                n += data.num_graphs
            train_loss, train_acc = tot / max(n, 1), correct / max(n, 1)
            val_loss, val_acc = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]["lr"]
            for k, v in zip(self.history, (train_loss, val_loss, train_acc, val_acc, lr)):
                self.history[k].append(v)
            if verbose:
                print(f"Epoch {epoch + 1:03d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} "
                      f"| TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f} | LR {lr:.6f}")
            es.step(val_loss)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            if es.stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def warmup_memory(self, graphs, batch_size: int = 16) -> None:
        """Chronological pass over the training graphs that populates the
        TGN memory without touching the encoder parameters."""
        self.tgn_memory.reset()
        self.model.eval()
        for data in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            data = data.to(self.device)
            *_, node_emb, nids = self.model(data, self.tgn_memory)
            if nids is not None:
                self.tgn_memory.update_with_gru(nids, node_emb, self.model.memory_updater,
                                                self.device)

    def incremental_update(self, graphs_batch, n_steps: int = 1) -> None:
        """Insert newly labelled graphs into replay, then take ``n_steps``
        gradient steps on L_base + L_EWC over replay mini-batches."""
        for g in graphs_batch:
            self.replay_buffer.add(g)
        for _ in range(n_steps):
            replayed = self.replay_buffer.sample(self.replay_batch)
            if not replayed:
                break
            self._step(Batch.from_data_list(replayed))
        self.model.eval()
