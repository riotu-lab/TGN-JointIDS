"""Lane memory, chronological lanes, TBPTT gradient flow,
causality, resets, reproducibility and the paired statistics.

Run:  python -m pytest tests -q
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tgn_jointids.config import Config  # noqa: E402
from tgn_jointids.lanes import UnitLaneScheduler, unit_ids  # noqa: E402
from tgn_jointids.models.lane_memory import LaneMemory  # noqa: E402
from tgn_jointids.models.tgn_jointids import TGNJointIDS  # noqa: E402
from tgn_jointids.stats import (confusion_by_cluster, macro_f1_from_cm,  # noqa: E402
                                paired_cluster_bootstrap)
from tgn_jointids.tbptt import TBPTTTrainer  # noqa: E402

UNIT = 60000
C = 4
LOSS_CFG = Config({"lambda_bin": 1.0, "lambda_multi": 0.7, "lambda_node": 0.2,
                   "focal_gamma": 2.0, "label_smoothing": 0.05, "focal_impl": "standard"})


def make_graphs(n_units=6, per_unit=5, seed=0, shuffle_storage=True):
    """Toy timeline: unit u starts at u*UNIT; windows every 2.5 s; label = u % C.
    Nodes come from a small IP pool so IPs recur across windows and units."""
    g = torch.Generator().manual_seed(seed)
    graphs = []
    for u in range(n_units):
        for k in range(per_unit):
            ids = torch.randperm(8, generator=g)[:4] + 100
            x = torch.randn(4, 5, generator=g) + float(u % C)
            graphs.append(Data(
                x=x, edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                edge_attr=torch.randn(3, 3, generator=g), y_multi=torch.tensor([u % C]),
                y_bin=torch.tensor([int(u % C != 0)]), y_node=torch.zeros(4, dtype=torch.long),
                node_ids=ids, window_id=torch.tensor([u * UNIT + k * 2500]), num_nodes=4))
    if shuffle_storage:   # storage order must not matter: the scheduler sorts by time
        perm = torch.randperm(len(graphs), generator=g).tolist()
        graphs = [graphs[i] for i in perm]
    return graphs


def make_model(seed=0):
    torch.manual_seed(seed)
    return TGNJointIDS(in_dim=5, hidden_dim=16, num_classes=C, memory_dim=8, num_layers=1,
                       num_heads=2, dropout=0.0, edge_dim=3)


def make_trainer(arm="trainable", seed=0, lanes=3, k=4, model=None):
    model = model or make_model(seed)
    return TBPTTTrainer(model, "cpu", torch.ones(2), torch.ones(C), LOSS_CFG, arm=arm,
                        unit_ms=UNIT, t0=0, num_lanes=lanes, tbptt_k=k, eval_lanes=5,
                        schedule_seed=seed, selection_metric="val_macro_f1")


# ====================================================================== scheduler
def test_scheduler_each_graph_once_chronological_and_units_whole():
    graphs = make_graphs(n_units=9, per_unit=4)
    s = UnitLaneScheduler(graphs, num_lanes=3, unit_ms=UNIT, t0=0, seed=1)
    seen, lane_seq = [], {}
    for step in s.steps(0):
        assert len(set(step.lanes)) == len(step.lanes)          # <= 1 graph per lane per step
        for i, lane, start, unit in zip(step.idx, step.lanes, step.reset, step.units):
            seen.append(i)
            lane_seq.setdefault(lane, []).append((i, start, unit))
    assert sorted(seen) == list(range(len(graphs)))
    u = unit_ids(graphs, UNIT, 0)
    unit_lane = {}
    for lane, seq in lane_seq.items():
        for j, (i, start, unit) in enumerate(seq):
            assert unit == u[i]
            assert unit_lane.setdefault(unit, lane) == lane       # a unit never spans lanes
            first_of_unit = j == 0 or seq[j - 1][2] != unit
            assert start == first_of_unit                         # reset exactly at unit start
            if not first_of_unit:                                 # time order inside the unit
                assert graphs[i].window_id > graphs[seq[j - 1][0]].window_id


def test_scheduler_deterministic_and_reshuffled_per_epoch():
    graphs = make_graphs(n_units=12)
    a = UnitLaneScheduler(graphs, 3, UNIT, 0, seed=7)
    b = UnitLaneScheduler(graphs, 3, UNIT, 0, seed=7)
    assert a.lane_plan(0) == b.lane_plan(0)
    assert a.lane_plan(0) != a.lane_plan(1)
    assert a.lane_plan(0) != UnitLaneScheduler(graphs, 3, UNIT, 0, seed=8).lane_plan(0)
    ev = UnitLaneScheduler(graphs, 3, UNIT, 0, shuffle=False)
    assert ev.lane_plan(0) == ev.lane_plan(5)


# ====================================================================== lane memory
def test_lane_isolation_and_reset():
    mem, gru = LaneMemory(8), nn.GRUCell(6, 8)
    mem.write([0, 1], [5, 5], torch.randn(2, 6), gru)            # same IP, two lanes
    r = mem.read([0, 1], [5, 5], "cpu")
    assert not torch.allclose(r[0], r[1])
    mem.reset_lane(0)
    r2 = mem.read([0, 1], [5, 5], "cpu")
    assert torch.all(r2[0] == 0) and torch.equal(r2[1], r[1])


def test_read_is_pure_and_duplicates_rejected():
    mem, gru = LaneMemory(8), nn.GRUCell(6, 8)
    mem.read([0, 0], [1, 2], "cpu")
    assert mem.n_keys() == 0
    with pytest.raises(ValueError):
        mem.write([0, 0], [3, 3], torch.randn(2, 6), gru)


def test_eq7_first_visit_and_ema():
    torch.manual_seed(0)
    mem, gru = LaneMemory(8, alpha=0.7), nn.GRUCell(6, 8)
    h1, h2 = torch.randn(1, 6), torch.randn(1, 6)
    mem.write([0], [9], h1, gru)
    m1 = gru(h1, torch.zeros(1, 8))[0]
    assert torch.allclose(mem.read([0], [9], "cpu")[0], m1)
    mem.write([0], [9], h2, gru)
    m2 = 0.7 * m1 + 0.3 * gru(h2, m1.unsqueeze(0))[0]
    assert torch.allclose(mem.read([0], [9], "cpu")[0], m2, atol=1e-6)


def test_gradient_reaches_gru_and_stops_at_detach():
    torch.manual_seed(0)
    mem, gru = LaneMemory(8), nn.GRUCell(6, 8)
    h = torch.randn(1, 6, requires_grad=True)
    mem.write([0], [1], h, gru)
    mem.write([0], [1], torch.randn(1, 6), gru)                   # second update, step 2
    mem.read([0], [1], "cpu").sum().backward()                    # loss two updates later
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in gru.parameters())
    assert h.grad is not None and h.grad.abs().sum() > 0          # reaches the encoder input
    gru.zero_grad()
    mem.detach()
    assert not mem.requires_grad_any()
    out = mem.read([0], [1], "cpu").sum()
    assert not out.requires_grad                                  # nothing flows past detach


# ====================================================================== trainer: gradients
@pytest.mark.parametrize("arm", ["graph_only", "frozen_gru", "trainable"])
def test_gru_gradients_by_arm(arm):
    tr = make_trainer(arm)
    before = copy.deepcopy(tr.model.memory_updater.state_dict())
    stats = tr.train_epoch(make_graphs(), epoch=0)
    after = tr.model.memory_updater.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    if arm == "trainable":
        assert stats["gru_grad_norm"] > 0 and changed
    else:
        assert stats["gru_grad_norm"] == 0 and not changed
    assert stats["opt_steps"] == int(np.ceil(stats["steps"] / tr.tbptt_k))
    assert not tr.tgn_memory.requires_grad_any()                  # detached at epoch end


def test_no_inplace_autograd_errors_over_many_truncations():
    tr = make_trainer("trainable", k=3)
    with torch.autograd.set_detect_anomaly(True):
        tr.train_epoch(make_graphs(n_units=8, per_unit=7), epoch=0)
        tr.train_epoch(make_graphs(n_units=8, per_unit=7), epoch=1)


# ====================================================================== causality
def _probs(tr, graphs, lanes=None):
    return tr.causal_predict(graphs, num_lanes=lanes)["prob_multi"]


def _sorted_unit(graphs, unit):
    idx = [i for i, g in enumerate(graphs) if int(g.window_id) // UNIT == unit]
    return sorted(idx, key=lambda i: int(graphs[i].window_id))


def test_future_window_cannot_change_past_predictions():
    tr = make_trainer("trainable")
    graphs = make_graphs()
    base = _probs(tr, graphs)
    order = _sorted_unit(graphs, 2)
    future = copy.deepcopy(graphs)
    future[order[3]].x = future[order[3]].x + 10.0                 # perturb window 4 of unit 2
    pert = _probs(tr, future)
    assert np.allclose(base[order[:3]], pert[order[:3]])          # windows 1-3 unchanged
    assert not np.allclose(base[order[3]], pert[order[3]])
    assert not np.allclose(base[order[4]], pert[order[4]])        # memory carries it forward
    others = [i for i in range(len(graphs)) if i not in order]
    assert np.allclose(base[others], pert[others])                # other units unaffected


def test_prediction_before_update_matches_manual_recurrence():
    tr = make_trainer("trainable")
    graphs = make_graphs()
    probs = _probs(tr, graphs)
    order = _sorted_unit(graphs, 1)
    from torch_geometric.data import Batch
    mem = LaneMemory(8)
    tr.model.eval()
    with torch.no_grad():
        for i in order:
            d = Batch.from_data_list([graphs[i]])
            _, ml, _, emb, nids = tr.model(d, mem.view([0] * d.num_nodes))
            assert np.allclose(torch.softmax(ml, 1)[0].numpy(), probs[i], atol=1e-6)
            mem.write([0] * d.num_nodes, nids, emb, tr.model.memory_updater)


def test_unit_start_equals_empty_memory_and_lane_count_invariance():
    tr = make_trainer("trainable")
    off = make_trainer("graph_only", model=tr.model)
    graphs = make_graphs()
    p_mem, p_off = _probs(tr, graphs), _probs(off, graphs)
    starts = [_sorted_unit(graphs, u)[0] for u in range(6)]
    assert np.allclose(p_mem[starts], p_off[starts], atol=1e-6)    # reset = no history
    later = [_sorted_unit(graphs, u)[2] for u in range(6)]
    assert not np.allclose(p_mem[later], p_off[later])
    assert np.allclose(_probs(tr, graphs, lanes=1), _probs(tr, graphs, lanes=7), atol=1e-6)


def test_labels_never_enter_memory():
    tr = make_trainer("trainable")
    graphs = make_graphs()
    relabeled = copy.deepcopy(graphs)
    for g in relabeled:
        g.y_multi = (g.y_multi + 1) % C
        g.y_bin = 1 - g.y_bin
        g.y_node = 1 - g.y_node
    assert np.allclose(_probs(tr, graphs), _probs(tr, relabeled))


def test_pseudonymisation_invariance_under_unit_reset():
    tr = make_trainer("trainable")
    graphs = make_graphs()
    pseudo = copy.deepcopy(graphs)
    rng = np.random.default_rng(0)
    for u in range(6):                                           # a fresh bijection per unit
        perm = dict(zip(range(100, 108), (rng.permutation(8) + 5000 + 10 * u).tolist()))
        for i in _sorted_unit(pseudo, u):
            pseudo[i].node_ids = torch.tensor([perm[int(n)] for n in pseudo[i].node_ids])
    assert np.allclose(_probs(tr, graphs), _probs(tr, pseudo), atol=1e-6)


# ====================================================================== reproducibility / matching
def test_reproducible_given_seed_and_sensitive_to_seed():
    def run(seed):
        tr = make_trainer("trainable", seed=seed)
        tr.train_epoch(make_graphs(), epoch=0)
        return tr.model.state_dict()
    a, b, c = run(3), run(3), run(4)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_arms_start_identical_and_share_schedule():
    m = {arm: make_trainer(arm, seed=5).model.state_dict() for arm in ("graph_only", "trainable")}
    assert all(torch.equal(m["graph_only"][k], m["trainable"][k]) for k in m["graph_only"])
    graphs = make_graphs()
    s1 = [(s.idx, s.lanes) for s in UnitLaneScheduler(graphs, 3, UNIT, 0, seed=5).steps(2)]
    s2 = [(s.idx, s.lanes) for s in UnitLaneScheduler(graphs, 3, UNIT, 0, seed=5).steps(2)]
    assert s1 == s2


def test_fit_runs_and_restores_best_state():
    tr = make_trainer("trainable")
    graphs = make_graphs(n_units=8)
    hist = tr.fit(graphs[:30], graphs[30:], epochs=2, verbose=False)
    assert len(hist["val_macro_f1"]) == 2 and tr.best_epoch in (1, 2)
    assert len(hist["gru_grad_norm"]) == 2


# ====================================================================== data guard
def test_refuses_double_normalisation(tmp_path):
    from tgn_jointids.pipeline import load_graph_bundle
    graphs = make_graphs(n_units=3)
    path = tmp_path / "b.pt"
    torch.save({"graphs": graphs, "split": {"train": [0, 1, 2], "val": [3], "stream": [4]},
                "node_dim": 5, "edge_dim": 3}, path)
    cfg = Config({"paths": {"graphs": str(path), "output_dir": str(tmp_path / "o")},
                  "features": {"normalization": {"mode": "log1p_zscore"}},
                  "graph": {"temporal_dim": 0}})
    load_graph_bundle(cfg)                                        # fine without a sidecar
    path.with_suffix(".normalization.json").write_text("{}")
    with pytest.raises(ValueError):
        load_graph_bundle(cfg)


# ====================================================================== statistics
def test_macro_f1_from_cm_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    cm = np.bincount(y * 4 + p, minlength=16).reshape(4, 4)
    assert macro_f1_from_cm(cm) == pytest.approx(f1_score(y, p, average="macro"))


def test_paired_cluster_bootstrap_null_and_effect():
    rng = np.random.default_rng(1)
    n, K = 2000, 40
    y = rng.integers(0, 4, n)
    clusters = rng.integers(0, K, n)
    noisy = [np.where(rng.random(n) < 0.6, y, rng.integers(0, 4, n)) for _ in range(5)]
    good = [np.where(rng.random(n) < 0.9, y, rng.integers(0, 4, n)) for _ in range(5)]
    cm = lambda p: confusion_by_cluster(y, p, clusters, 4)[1]       # noqa: E731
    same = paired_cluster_bootstrap([cm(p) for p in noisy], [cm(p) for p in noisy], n_boot=200)
    assert same["mean_delta"] == 0 and same["ci95"][0] <= 0 <= same["ci95"][1]
    eff = paired_cluster_bootstrap([cm(p) for p in good], [cm(p) for p in noisy], n_boot=200)
    assert eff["ci95"][0] > 0 and eff["seeds_positive"] == 5
