"""Exact optimiser-update budget (training.max_updates) (7,000 updates, cosine LR).

Run:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_memory_tbptt import LOSS_CFG, make_graphs, make_model  # noqa: E402
from tgn_jointids.config import Config  # noqa: E402
from tgn_jointids.tbptt import TBPTTTrainer  # noqa: E402

C = 4


def trainer(loss_cfg=LOSS_CFG, pooling="sagpool", seed=0):
    torch.manual_seed(seed)
    from tgn_jointids.models.tgn_jointids import TGNJointIDS
    m = TGNJointIDS(in_dim=5, hidden_dim=16, num_classes=C, memory_dim=8, num_layers=1,
                    num_heads=2, dropout=0.0, edge_dim=3, pooling=pooling)
    return TBPTTTrainer(m, "cpu", torch.ones(2), torch.ones(C), loss_cfg, arm="trainable",
                        unit_ms=60000, t0=0, num_lanes=3, tbptt_k=4, eval_lanes=5,
                        schedule_seed=seed, selection_metric="val_macro_f1", lr=1e-3, min_lr=1e-5)


def test_stops_at_exact_update_count_mid_epoch_and_evaluates_on_grid():
    graphs = make_graphs(n_units=8, per_unit=5)
    tr = trainer()
    hist = tr.fit_fixed_updates(graphs[:30], graphs[30:], max_updates=7, eval_every=3, verbose=False)
    assert tr.total_updates == 7
    assert hist["update"] == [3, 6, 7]                       # identical grid + final update
    assert tr.best_epoch in hist["update"]                   # best checkpoint = an evaluated update
    assert tr.best_score == max(hist["val_macro_f1"])


def test_no_early_stopping_even_when_validation_worsens():
    graphs = make_graphs(n_units=8, per_unit=5)
    tr = trainer()
    tr.es_patience = 1                                       # would stop at once in fit()
    tr.fit_fixed_updates(graphs[:30], graphs[30:], max_updates=12, eval_every=2, verbose=False)
    assert tr.total_updates == 12


def test_cosine_schedule_depends_only_on_update_index():
    graphs = make_graphs(n_units=8, per_unit=5)
    ce_cfg = Config(dict(LOSS_CFG, focal_gamma=0.0))
    a, b = trainer(), trainer(loss_cfg=ce_cfg, pooling="mean")
    ha = a.fit_fixed_updates(graphs[:30], graphs[30:], max_updates=9, eval_every=3, verbose=False)
    hb = b.fit_fixed_updates(graphs[:30], graphs[30:], max_updates=9, eval_every=3, verbose=False)
    assert ha["lr"] == hb["lr"]                              # same schedule for every variant
    assert ha["lr"][-1] == pytest.approx(1e-5)               # reaches min_lr at the budget
    expected = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * 3 / 9))
    assert ha["lr"][0] == pytest.approx(expected)


def test_constant_schedule_and_invalid_schedule():
    graphs = make_graphs(n_units=8, per_unit=5)
    tr = trainer()
    h = tr.fit_fixed_updates(graphs[:30], graphs[30:], max_updates=4, eval_every=2,
                             lr_schedule="constant", verbose=False)
    assert all(lr == pytest.approx(1e-3) for lr in h["lr"])
    with pytest.raises(ValueError):
        trainer().fit_fixed_updates(graphs[:30], graphs[30:], max_updates=2, eval_every=1,
                                    lr_schedule="step", verbose=False)


def test_variants_share_initialisation_and_gru_trains_under_budget():
    a, b = trainer(), trainer(pooling="mean")
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    graphs = make_graphs(n_units=8, per_unit=5)
    before = {k: v.clone() for k, v in a.model.memory_updater.state_dict().items()}
    h = a.fit_fixed_updates(graphs[:30], graphs[30:], max_updates=6, eval_every=3, verbose=False)
    assert all(g > 0 for g in h["gru_grad_norm"])
    after = a.model.memory_updater.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_original_fit_unchanged_by_on_update_hook():
    graphs = make_graphs(n_units=6, per_unit=5)
    t1, t2 = trainer(), trainer()
    s1 = t1.train_epoch(graphs, 0)
    s2 = t2.train_epoch(graphs, 0, on_update=lambda: False)
    assert s1["opt_steps"] == s2["opt_steps"]
    p1, p2 = t1.model.state_dict(), t2.model.state_dict()
    assert all(torch.equal(p1[k], p2[k]) for k in p1)
