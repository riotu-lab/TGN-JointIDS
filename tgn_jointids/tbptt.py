"""Chronological-lane training with truncated BPTT and causal unit-reset evaluation
(``memory.mode: tbptt``), the training procedure of TGN-JointIDS.

Variants (``memory.arm``), identical in data, schedule, budget and evaluation:
  trainable   TGN-JointIDS: memory read/written with gradients, GRU updater trained
  graph_only  Graph-only ablation: memory is never read or written
  frozen_gru  memory read/written with gradients, GRU updater frozen at init

Per step (one window from each active lane):
  1. reset the memory of lanes that start a new unit
  2. read memory (state after the previous step) -> encode -> predict -> loss
  3. write memory from this step's node embeddings (gradients kept)
Every ``tbptt_k`` steps: backward on the mean step loss, clip, optimiser
step, then detach all lane memories. Labels never enter the memory; a window
is always predicted before its own memory update.

The memory is short-term: it is reset at every unit (60 s block) and carries host state only
across the windows of that block.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch_geometric.data import Batch

from .lanes import UnitLaneScheduler
from .models.lane_memory import LaneMemory
from .trainer import EarlyStopping, JointTrainer

ARMS = ("graph_only", "frozen_gru", "trainable")


def grad_norm(params) -> float:
    sq = [p.grad.detach().pow(2).sum() for p in params if p.grad is not None]
    return float(torch.stack(sq).sum().sqrt()) if sq else 0.0


class TBPTTTrainer(JointTrainer):

    def __init__(self, model, device, bin_weights, multi_weights, loss_cfg, *, arm: str,
                 unit_ms: int, t0: int, num_lanes: int = 32, tbptt_k: int = 24,
                 memory_alpha: float = 0.7, memory_decay: float = 1.0, eval_lanes: int = 256,
                 schedule_seed: int = 0, train_eval_units: int = 0, **kw):
        if arm not in ARMS:   # note: YAML reads a bare `off` as False, hence "graph_only"
            raise ValueError(f"Unknown memory.arm: {arm!r}; expected one of {ARMS}")
        super().__init__(model, LaneMemory(model.memory_dim, memory_alpha, memory_decay), device,
                         bin_weights, multi_weights, loss_cfg, **kw)
        self.arm, self.unit_ms, self.t0 = arm, int(unit_ms), int(t0)
        self.num_lanes, self.tbptt_k, self.eval_lanes = int(num_lanes), int(tbptt_k), int(eval_lanes)
        self.memory_alpha, self.memory_decay = memory_alpha, memory_decay
        self.schedule_seed = int(schedule_seed)
        self.train_eval_units = int(train_eval_units)
        if arm == "frozen_gru":
            for p in self.model.memory_updater.parameters():
                p.requires_grad_(False)
        for k in ("gru_grad_norm", "total_grad_norm", "optimizer_steps", "epoch_seconds",
                  "val_macro_f1", "train_static_acc", "train_static_macro_f1"):
            self.history.setdefault(k, [])

    # ------------------------------------------------------------------ helpers
    @property
    def uses_memory(self) -> bool:
        return self.arm != "graph_only"

    def new_memory(self) -> LaneMemory:
        return LaneMemory(self.model.memory_dim, self.memory_alpha, self.memory_decay)

    def _forward(self, graphs, step, memory: LaneMemory):
        for lane, start in zip(step.lanes, step.reset):
            if start:
                memory.reset_lane(lane)
        data = Batch.from_data_list([graphs[i] for i in step.idx]).to(self.device)
        node_lanes = torch.as_tensor(step.lanes)[data.batch.cpu()].tolist()
        out = self.model(data, memory.view(node_lanes) if self.uses_memory else None)
        return data, node_lanes, out

    def _write(self, memory: LaneMemory, node_lanes, nids, node_emb, keep_grad: bool):
        if self.uses_memory and nids is not None:
            memory.write(node_lanes, nids, node_emb if keep_grad else node_emb.detach(),
                         self.model.memory_updater)

    def _optimise(self, losses, memory: LaneMemory, stats: dict) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        stats["gru"].append(grad_norm(self.model.memory_updater.parameters()))
        stats["total"].append(grad_norm(self.model.parameters()))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        memory.detach()
        stats["opt_steps"] += 1

    # ------------------------------------------------------------------ training
    def train_epoch(self, graphs, epoch: int, max_steps: int | None = None,
                    on_update=None) -> dict:
        """``on_update`` (optional) is called after every optimiser update; if it
        returns True the epoch ends immediately (used for exact update budgets)."""
        sched = UnitLaneScheduler(graphs, self.num_lanes, self.unit_ms, self.t0,
                                  seed=self.schedule_seed, shuffle=True)
        memory = self.new_memory()
        self.tgn_memory = memory
        self.model.train()
        stats = {"gru": [], "total": [], "opt_steps": 0}
        losses, tot, correct, n, steps = [], 0.0, 0, 0, 0
        t_start = time.perf_counter()
        stop = False
        for step in sched.steps(epoch):
            if max_steps is not None and steps >= max_steps:
                break
            data, node_lanes, (bin_l, multi_l, node_l, node_emb, nids) = \
                self._forward(graphs, step, memory)
            loss = self._loss(data, bin_l, multi_l, node_l)
            losses.append(loss)
            self._write(memory, node_lanes, nids, node_emb, keep_grad=True)
            tot += loss.item() * data.num_graphs
            correct += (multi_l.argmax(1) == data.y_multi.view(-1)).sum().item()
            n += data.num_graphs
            steps += 1
            if len(losses) == self.tbptt_k:
                self._optimise(losses, memory, stats)
                losses = []
                if on_update is not None and on_update():
                    stop = True
                    break
        if losses and not stop:
            self._optimise(losses, memory, stats)
            if on_update is not None:
                on_update()
        return {"loss": tot / max(n, 1), "acc": correct / max(n, 1), "steps": steps,
                "opt_steps": stats["opt_steps"], "seconds": time.perf_counter() - t_start,
                "gru_grad_norm": float(np.mean(stats["gru"])) if stats["gru"] else 0.0,
                "total_grad_norm": float(np.mean(stats["total"])) if stats["total"] else 0.0,
                "gru_grad_norms": stats["gru"]}

    # ------------------------------------------------------------------ causal evaluation
    @torch.no_grad()
    def causal_predict(self, graphs, num_lanes: int | None = None) -> dict:
        """Deployment-consistent, frozen-parameter pass: each unit starts from
        empty memory, windows in time order, predict-then-update. Units are
        independent, so the result does not depend on ``num_lanes``."""
        self.model.eval()
        sched = UnitLaneScheduler(graphs, num_lanes or self.eval_lanes, self.unit_ms, self.t0,
                                  shuffle=False)
        memory = self.new_memory()
        n, C = len(graphs), self.model.num_classes
        out = {"prob_multi": np.zeros((n, C), np.float32), "prob_attack": np.zeros(n, np.float32),
               "unit": np.zeros(n, np.int64)}
        total_loss = 0.0
        for step in sched.steps(0):
            data, node_lanes, (bin_l, multi_l, node_l, node_emb, nids) = \
                self._forward(graphs, step, memory)
            total_loss += self._loss(data, bin_l, multi_l, node_l).item() * data.num_graphs
            idx = np.asarray(step.idx)
            out["prob_multi"][idx] = torch.softmax(multi_l, 1).cpu().numpy()
            out["prob_attack"][idx] = torch.softmax(bin_l, 1)[:, 1].cpu().numpy()
            out["unit"][idx] = step.units
            self._write(memory, node_lanes, nids, node_emb, keep_grad=False)
        out["y_multi"] = np.array([int(g.y_multi.item()) for g in graphs])
        out["y_bin"] = np.array([int(g.y_bin.item()) for g in graphs])
        out["window_id"] = np.array([int(g.window_id.item()) for g in graphs])
        out["pred_multi"] = out["prob_multi"].argmax(1)
        out["pred_bin"] = (out["prob_attack"] >= 0.5).astype(np.int64)
        out["loss"] = total_loss / max(n, 1)
        return out

    def _static_subset(self, graphs):
        if self.train_eval_units <= 0:
            return None
        sched = UnitLaneScheduler(graphs, 1, self.unit_ms, self.t0, shuffle=False)
        rng = np.random.default_rng(self.schedule_seed)
        units = rng.choice(sched.unit_order, min(self.train_eval_units, len(sched.unit_order)),
                           replace=False)
        return [graphs[i] for u in sorted(units) for i in sched.units[int(u)]]

    # ------------------------------------------------------------------ fit
    def fit(self, train_graphs, val_graphs, epochs: int, verbose: bool = True,
            max_steps_per_epoch: int | None = None):
        es = EarlyStopping(self.es_patience, self.es_min_delta)
        static = self._static_subset(train_graphs)
        for epoch in range(epochs):
            tr = self.train_epoch(train_graphs, epoch, max_steps=max_steps_per_epoch)
            va = self.causal_predict(val_graphs)
            val_acc = float(np.mean(va["pred_multi"] == va["y_multi"]))
            val_f1 = self._macro_f1(va["y_multi"], va["pred_multi"])
            st_acc = st_f1 = float("nan")
            if static is not None:
                st = self.causal_predict(static)
                st_acc = float(np.mean(st["pred_multi"] == st["y_multi"]))
                st_f1 = self._macro_f1(st["y_multi"], st["pred_multi"])
            self.scheduler.step(va["loss"])
            lr = self.optimizer.param_groups[0]["lr"]
            rows = {"train_loss": tr["loss"], "val_loss": va["loss"], "train_acc": tr["acc"],
                    "val_acc": val_acc, "lr": lr, "val_macro_f1": val_f1,
                    "train_static_acc": st_acc, "train_static_macro_f1": st_f1,
                    "gru_grad_norm": tr["gru_grad_norm"], "total_grad_norm": tr["total_grad_norm"],
                    "optimizer_steps": tr["opt_steps"], "epoch_seconds": tr["seconds"]}
            for k, v in rows.items():
                self.history[k].append(v)
            if verbose:
                print(f"Epoch {epoch + 1:03d} | arm {self.arm} | TrainLoss {tr['loss']:.4f} | "
                      f"ValLoss {va['loss']:.4f} | ValAcc {val_acc:.4f} | ValMacroF1 {val_f1:.4f} | "
                      f"TrainStaticAcc {st_acc:.4f} | GRU|g| {tr['gru_grad_norm']:.2e} | "
                      f"opt {tr['opt_steps']} | {tr['seconds']:.0f}s | LR {lr:.6f}", flush=True)
            es.step(va["loss"])
            score = val_acc if self.selection_metric == "val_acc" else val_f1
            if score > self.best_score:
                self.best_score, self.best_epoch, self.best_acc = score, epoch + 1, val_acc
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in self.model.state_dict().items()}
            if es.stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history


    # ------------------------------------------------------------------ fixed update budget
    def fit_fixed_updates(self, train_graphs, val_graphs, max_updates: int, eval_every: int,
                          lr_schedule: str = "cosine", verbose: bool = True):
        """Exact optimiser-update budget (``training.max_updates``), opt-in.

        * training stops after exactly ``max_updates`` updates (mid-epoch if needed);
        * no early stopping;
        * learning rate follows a schedule defined on the update index only
          ("cosine": lr0 -> min_lr over max_updates; "constant": lr0), so it is
          identical for every variant regardless of its loss scale;
        * causal validation every ``eval_every`` updates (and at the last one);
          the checkpoint with the best validation macro-F1 is kept.
        """
        if lr_schedule not in ("cosine", "constant"):
            raise ValueError(f"Unknown lr_schedule: {lr_schedule}")
        lr0 = self.optimizer.param_groups[0]["lr"]
        min_lr = self.scheduler.min_lrs[0] if hasattr(self.scheduler, "min_lrs") else 0.0
        static = self._static_subset(train_graphs)
        hist = {k: [] for k in ("update", "epoch", "lr", "val_loss", "val_acc", "val_macro_f1",
                                "train_static_acc", "train_static_macro_f1", "train_loss_window",
                                "gru_grad_norm", "total_grad_norm", "elapsed_seconds")}
        state = {"u": 0, "epoch": 0, "loss_acc": [], "t0": time.perf_counter()}

        def lr_at(u):
            if lr_schedule == "constant":
                return lr0
            return min_lr + 0.5 * (lr0 - min_lr) * (1 + np.cos(np.pi * min(u, max_updates) / max_updates))

        def evaluate():
            va = self.causal_predict(val_graphs)
            f1 = self._macro_f1(va["y_multi"], va["pred_multi"])
            acc = float(np.mean(va["pred_multi"] == va["y_multi"]))
            st_acc = st_f1 = float("nan")
            if static is not None:
                st = self.causal_predict(static)
                st_acc = float(np.mean(st["pred_multi"] == st["y_multi"]))
                st_f1 = self._macro_f1(st["y_multi"], st["pred_multi"])
            self.model.train()
            row = {"update": state["u"], "epoch": state["epoch"],
                   "lr": self.optimizer.param_groups[0]["lr"], "val_loss": va["loss"],
                   "val_acc": acc, "val_macro_f1": f1, "train_static_acc": st_acc,
                   "train_static_macro_f1": st_f1,
                   "train_loss_window": float(np.mean(state["loss_acc"])) if state["loss_acc"] else float("nan"),
                   "gru_grad_norm": float(np.mean(self._gru_window)) if self._gru_window else 0.0,
                   "total_grad_norm": float(np.mean(self._tot_window)) if self._tot_window else 0.0,
                   "elapsed_seconds": time.perf_counter() - state["t0"]}
            for k, v in row.items():
                hist[k].append(v)
            state["loss_acc"], self._gru_window, self._tot_window = [], [], []
            if f1 > self.best_score:
                self.best_score, self.best_epoch, self.best_acc = f1, state["u"], acc
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in self.model.state_dict().items()}
            if verbose:
                print(f"Update {state['u']:5d} | epoch {state['epoch']:3d} | ValLoss {va['loss']:.4f} | "
                      f"ValAcc {acc:.4f} | ValMacroF1 {f1:.4f} | TrainStaticAcc {st_acc:.4f} | "
                      f"GRU|g| {row['gru_grad_norm']:.2e} | LR {row['lr']:.2e} | "
                      f"{row['elapsed_seconds']:.0f}s", flush=True)

        def on_update():
            state["u"] += 1
            for g in self.optimizer.param_groups:
                g["lr"] = lr_at(state["u"])
            if state["u"] % eval_every == 0 or state["u"] >= max_updates:
                evaluate()
            return state["u"] >= max_updates

        self._gru_window, self._tot_window = [], []
        orig_optimise = self._optimise

        def tracked_optimise(losses, memory, stats):
            orig_optimise(losses, memory, stats)
            self._gru_window.append(stats["gru"][-1])
            self._tot_window.append(stats["total"][-1])
            state["loss_acc"].append(float(torch.stack([l.detach() for l in losses]).mean()))

        self._optimise = tracked_optimise
        try:
            self.best_score = -1.0
            for g in self.optimizer.param_groups:
                g["lr"] = lr_at(0)
            while state["u"] < max_updates:
                self.train_epoch(train_graphs, state["epoch"], on_update=on_update)
                state["epoch"] += 1
        finally:
            self._optimise = orig_optimise
        self.history_fixed = hist
        self.total_updates = state["u"]
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return hist
