"""Factories shared by the scripts in ``scripts/``: label space, graph builder, graph bundle
(with train-only feature scaling), model and TBPTT trainer."""

from __future__ import annotations

from pathlib import Path

import torch

from .data.features import LabelSpace
from .data.graph_builder import EnhancedGraphBuilder
from .data.normalization import assert_finite_and_bounded, scaler_from_config
from .losses import class_weights_from_config
from .models.tgn_jointids import TGNJointIDS
from .utils import env_path, save_json


def make_label_space(cfg) -> LabelSpace:
    return LabelSpace(cfg.data.classes, benign_label=cfg.data.benign_label,
                      strict=bool(cfg.data.get("strict_labels", False)))


def make_builder(cfg, label_space) -> EnhancedGraphBuilder:
    g = cfg.graph
    return EnhancedGraphBuilder(
        node_features=list(cfg.features.statistical) + list(cfg.features.behavioral),
        edge_features=cfg.features.edge, label_space=label_space,
        priority_classes=cfg.graph.priority_classes, use_edge_types=g.use_edge_types,
        use_temporal_encoding=g.temporal_dim > 0, structural_features=g.structural_features,
        temporal_dim=g.temporal_dim, edge_decay_factor=g.edge_decay,
        role_mean=g.get("role_mean", "legacy_half"),
        strict_features=bool(g.get("strict_features", False)))


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
    bundle["t0"] = min(int(x.window_id.item()) for x in g)  # global block origin
    check_split_integrity(bundle)
    scaler = scaler_from_config(cfg, bundle["node_dim"], bundle["edge_dim"])
    if scaler is not None and path.with_suffix(".normalization.json").exists():
        raise ValueError(f"{path} was already normalized offline "
                         f"({path.with_suffix('.normalization.json').name}); applying "
                         "features.normalization again would scale features twice.")
    if scaler is not None:
        # Fitted on the TRAINING partition only, then applied to every graph once.
        scaler.fit(bundle["train"])
        for graph in g:
            scaler.transform_graph(graph)
        bundle["scaler"] = scaler.state()
        bundle["scaler"]["check"] = {name: assert_finite_and_bounded(bundle[name])
                                     for name in ("train", "val", "stream")}
        save_json(bundle["scaler"], Path(env_path(cfg.paths.output_dir)) / "feature_scaler.json")
    return bundle


def check_split_integrity(bundle) -> None:
    """Validation check: the three index sets are disjoint and in range."""
    sets = {k: set(bundle["split"][k]) for k in ("train", "val", "stream")}
    n = len(bundle["graphs"])
    for k, s in sets.items():
        if len(s) != len(bundle["split"][k]):
            raise ValueError(f"Duplicate graph indices in split '{k}'.")
        if s and (min(s) < 0 or max(s) >= n):
            raise ValueError(f"Out-of-range graph indices in split '{k}'.")
    for a, b in (("train", "val"), ("train", "stream"), ("val", "stream")):
        if sets[a] & sets[b]:
            raise ValueError(f"Splits '{a}' and '{b}' share {len(sets[a] & sets[b])} graphs.")


def make_model(cfg, in_dim: int, edge_dim: int, num_classes: int) -> TGNJointIDS:
    m = cfg.model
    return TGNJointIDS(in_dim=in_dim, hidden_dim=m.hidden_dim, num_classes=num_classes,
                       memory_dim=m.memory_dim, num_layers=m.num_layers, num_heads=m.num_heads,
                       dropout=m.dropout, edge_dim=edge_dim, sagpool_ratio=m.sagpool_ratio,
                       pooling=m.get("pooling", "sagpool"),
                       use_edge_attr=bool(m.get("use_edge_attr", True)))


def make_tbptt_trainer(cfg, model, train_graphs, label_space, device, t0: int):
    """TBPTT trainer; every variant (``memory.arm``: trainable, frozen_gru, graph_only) gets the
    same loss, optimiser, schedule and evaluation from the config."""
    from .tbptt import TBPTTTrainer
    bin_w, multi_w, _ = class_weights_from_config(cfg.loss, label_space, train_graphs, device)
    t, m = cfg.training, cfg.memory
    return TBPTTTrainer(
        model, device, bin_w, multi_w, cfg.loss, arm=m.arm, unit_ms=int(m.unit_ms), t0=t0,
        num_lanes=int(t.lanes), tbptt_k=int(m.tbptt_k), memory_alpha=float(m.alpha),
        memory_decay=float(m.decay), eval_lanes=int(t.get("eval_lanes", 256)),
        schedule_seed=int(t.get("schedule_seed", 0)),
        train_eval_units=int(t.get("train_eval_units", 0)),
        lr=t.lr, weight_decay=t.weight_decay, min_lr=t.min_lr,
        scheduler_patience=t.scheduler_patience,
        early_stopping_patience=t.early_stopping_patience,
        early_stopping_min_delta=t.early_stopping_min_delta, grad_clip=t.grad_clip,
        selection_metric=t.get("selection_metric", "val_macro_f1"))
