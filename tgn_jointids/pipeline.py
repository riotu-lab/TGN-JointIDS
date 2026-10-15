"""High-level helpers shared by the scripts in ``scripts/``."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .continual import EWC, make_replay_buffer
from .data.features import LabelSpace
from .data.graph_builder import EnhancedGraphBuilder
from .losses import class_weights_from_config
from .metrics import attack_only_metrics, binary_metrics, class_recall, latency_summary
from .models.memory import TGNNodeMemory
from .models.tgn_jointids import TGNJointIDS
from .trainer import JointTrainer
from .utils import env_path


def make_label_space(cfg) -> LabelSpace:
    return LabelSpace(cfg.data.classes, benign_label=cfg.data.benign_label)


def make_builder(cfg, label_space) -> EnhancedGraphBuilder:
    g = cfg.graph
    return EnhancedGraphBuilder(
        node_features=list(cfg.features.statistical) + list(cfg.features.behavioral),
        edge_features=cfg.features.edge, label_space=label_space,
        priority_classes=cfg.graph.priority_classes, use_edge_types=g.use_edge_types,
        use_temporal_encoding=g.temporal_dim > 0, structural_features=g.structural_features,
        temporal_dim=g.temporal_dim, edge_decay_factor=g.edge_decay)


def load_graph_bundle(cfg) -> dict:
    """Load graphs + fixed split produced by scripts/build_graphs.py."""
    path = Path(env_path(cfg.paths.graphs))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/build_graphs.py first.")
    bundle = torch.load(path, weights_only=False)
    g = bundle["graphs"]
    bundle["train"] = [g[i] for i in bundle["split"]["train"]]
    bundle["val"] = [g[i] for i in bundle["split"]["val"]]
    bundle["stream"] = [g[i] for i in bundle["split"]["stream"]]
    return bundle


def make_model(cfg, in_dim: int, edge_dim: int, num_classes: int) -> TGNJointIDS:
    m = cfg.model
    return TGNJointIDS(in_dim=in_dim, hidden_dim=m.hidden_dim, num_classes=num_classes,
                       memory_dim=m.memory_dim, num_layers=m.num_layers, num_heads=m.num_heads,
                       dropout=m.dropout, edge_dim=edge_dim, sagpool_ratio=m.sagpool_ratio)


def make_memory(cfg) -> TGNNodeMemory:
    return TGNNodeMemory(cfg.model.memory_dim, decay=cfg.memory.decay, alpha=cfg.memory.alpha)


def make_trainer(cfg, model, memory, train_graphs, label_space, device) -> JointTrainer:
    bin_w, multi_w, _ = class_weights_from_config(cfg.loss, label_space, train_graphs, device)
    t = cfg.training
    return JointTrainer(
        model, memory, device, bin_w, multi_w, cfg.loss, lr=t.lr, weight_decay=t.weight_decay,
        min_lr=t.min_lr, scheduler_patience=t.scheduler_patience,
        early_stopping_patience=t.early_stopping_patience,
        early_stopping_min_delta=t.early_stopping_min_delta, grad_clip=t.grad_clip,
        replay_buffer=make_replay_buffer(cfg.continual.replay, label_space.num_classes),
        replay_batch=cfg.continual.replay.batch_size)


def make_loaders(cfg, train_graphs, val_graphs):
    bs = cfg.training.batch_size
    return (DataLoader(train_graphs, batch_size=bs, shuffle=False),
            DataLoader(val_graphs, batch_size=bs, shuffle=False))


def prepare_for_streaming(cfg, trainer, train_graphs, device) -> None:
    """Phase 2: memory warm-up on training graphs + EWC reference."""
    c = cfg.continual
    trainer.warmup_memory(train_graphs, batch_size=c.warmup_batch_size)
    ewc_loader = DataLoader(train_graphs[: c.ewc_n_graphs], batch_size=c.ewc_batch_size,
                            shuffle=False)
    trainer.ewc = EWC(trainer.model, ewc_loader, device, lambda_ewc=c.lambda_ewc)


def summarize_stream(cfg, df, label_space, tgn_memory=None) -> dict:
    out = {"n_windows": len(df), "binary": binary_metrics(df["bin_true"], df["bin_pred"])}
    for cascade in (True, False):
        s, per_class, _ = attack_only_metrics(df, label_space, cascade=cascade)
        key = "attack_type_cascade" if cascade else "attack_type_raw"
        out[key] = s
        out[key]["recall"] = {r["class"]: r["recall"] for _, r in per_class.iterrows()}
    out["tracked_recall"] = {c: class_recall(df, label_space, c, cascade=True)
                             for c in cfg.evaluation.tracked_classes}
    out["latency"] = latency_summary(df["latency_ms"], cfg.graph.window_ms)
    if tgn_memory is not None:
        out["memory"] = {"keys": len(tgn_memory), "bytes": tgn_memory.nbytes()}
    return out
