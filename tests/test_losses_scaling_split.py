"""Unit tests: focal loss and ENS weights, train-only feature scaling, split integrity, graph
roles, block assignment and config overrides.

Run:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tgn_jointids.config import Config, load_config  # noqa: E402
from tgn_jointids.data.features import LabelSpace  # noqa: E402
from tgn_jointids.data.graph_builder import EnhancedGraphBuilder  # noqa: E402
from tgn_jointids.data.normalization import FeatureScaler, scaler_from_config  # noqa: E402
from tgn_jointids.losses import (compute_ens_weights, compute_joint_loss,  # noqa: E402
                                 focal_loss_standard, focal_loss_with_smoothing)
from tgn_jointids.pipeline import check_split_integrity  # noqa: E402

CLASSES = ["benign", "ddos", "xss", "mitm"]


def _graph(y, n=3, x_scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 5, generator=g) * x_scale
    ei = torch.tensor([[0, 1], [1, 2]])
    return Data(x=x, edge_index=ei, edge_attr=torch.rand(2, 3, generator=g) * x_scale,
                y_multi=torch.tensor([y]), y_bin=torch.tensor([int(y != 0)]),
                node_ids=torch.arange(n), num_nodes=n)


# ---------------------------------------------------------------- focal loss (B3)
def test_standard_focal_matches_eq15_by_hand():
    torch.manual_seed(0)
    logits, y = torch.randn(6, 4), torch.tensor([0, 1, 2, 3, 1, 0])
    w = torch.tensor([0.2, 0.5, 1.0, 3.0])
    gamma, eps = 2.0, 0.05
    p = F.softmax(logits, 1)
    expected = []
    for i in range(6):
        q = torch.full((4,), eps / 4)
        q[y[i]] += 1 - eps
        ce = -(q * p[i].log()).sum()
        expected.append(w[y[i]] * (1 - p[i, y[i]]) ** gamma * ce)
    got = focal_loss_standard(logits, y, w, gamma, eps)
    assert torch.allclose(got, torch.stack(expected).mean(), atol=1e-6)


def test_standard_focal_reduces_to_ce_when_gamma_zero_and_no_weights():
    torch.manual_seed(1)
    logits, y = torch.randn(8, 4), torch.randint(0, 4, (8,))
    assert torch.allclose(focal_loss_standard(logits, y, None, 0.0, 0.0),
                          F.cross_entropy(logits, y), atol=1e-6)


def test_legacy_focal_unchanged_and_differs_under_weights():
    """The legacy implementation is kept bit-for-bit; with class weights its
    modulating factor is not (1 - p*)^gamma."""
    torch.manual_seed(2)
    logits, y = torch.randn(16, 4), torch.randint(0, 4, (16,))
    w = torch.tensor([0.2, 0.5, 1.0, 3.0])
    ce = F.cross_entropy(logits, y, weight=w, label_smoothing=0.05, reduction="none")
    legacy_ref = ((1 - torch.exp(-ce)) ** 2 * ce).mean()
    assert torch.allclose(focal_loss_with_smoothing(logits, y, w, 2.0, 0.05), legacy_ref)
    assert not torch.allclose(focal_loss_with_smoothing(logits, y, w, 2.0, 0.05),
                              focal_loss_standard(logits, y, w, 2.0, 0.05))


def test_joint_loss_default_is_legacy():
    torch.manual_seed(3)
    b, m, nd = torch.randn(4, 2), torch.randn(4, 4), torch.randn(9, 2)
    yb, ym, yn = torch.tensor([0, 1, 1, 1]), torch.tensor([0, 1, 2, 3]), torch.randint(0, 2, (9,))
    w2, w4 = torch.ones(2), torch.tensor([0.2, 0.5, 1.0, 3.0])
    a = compute_joint_loss(b, m, nd, yb, ym, yn, w2, w4)[0]
    c = compute_joint_loss(b, m, nd, yb, ym, yn, w2, w4, focal_impl="legacy")[0]
    assert torch.equal(a, c)
    with pytest.raises(ValueError):
        compute_joint_loss(b, m, nd, yb, ym, yn, w2, w4, focal_impl="nope")


def test_ens_weights_mean_one_and_monotone():
    w = compute_ens_weights([10000, 1000, 100, 10], beta=0.9999).numpy()
    assert abs(w.mean() - 1) < 1e-6 and np.all(np.diff(w) > 0)


# ---------------------------------------------------------------- normalisation (B4)
def test_scaler_fits_on_train_only_and_skips_dims():
    train = [_graph(0, x_scale=1e7, seed=s) for s in range(20)]
    val = [_graph(1, x_scale=1e9, seed=100 + s) for s in range(5)]
    sc = FeatureScaler(5, 3, "log1p_zscore", node_skip=[4], edge_skip=[2]).fit(train)
    before_mean = sc.node.mean.clone()
    for g in train + val:
        sc.transform_graph(g)
    assert torch.equal(sc.node.mean, before_mean)          # val never refits
    x = torch.cat([g.x for g in train])
    assert torch.allclose(x[:, :4].mean(0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(x[:, :4].std(0, unbiased=False), torch.ones(4), atol=1e-3)
    assert sc.node.mean[4] == 0 and sc.node.std[4] == 1    # skipped dim untouched


def test_scaler_handles_constant_and_zero_columns():
    gs = [Data(x=torch.tensor([[0.0, 5.0], [0.0, 5.0]]), edge_attr=torch.zeros(1, 1),
               edge_index=torch.tensor([[0], [1]]), num_nodes=2)]
    sc = FeatureScaler(2, 1, "log1p_zscore").fit(gs)
    sc.transform_graph(gs[0])
    assert torch.isfinite(gs[0].x).all() and torch.all(gs[0].x == 0)


def test_scaler_from_config_default_skips_temporal_dims():
    cfg = Config({"features": {"normalization": {"mode": "log1p_zscore"}},
                  "graph": {"temporal_dim": 8}})
    sc = scaler_from_config(cfg, node_dim=35, edge_dim=5)
    assert sc.node.skip == list(range(27, 35)) and sc.edge.skip == [4]
    assert scaler_from_config(Config({"features": {}, "graph": {}}), 35, 5) is None


# ---------------------------------------------------------------- memory (B8)


# ---------------------------------------------------------------- order diagnostics (B1/B7)


def test_split_integrity_check():
    bundle = {"graphs": list(range(10)), "split": {"train": [0, 1, 2], "val": [3, 4], "stream": [5]}}
    check_split_integrity(bundle)
    bundle["split"]["val"] = [2, 3]
    with pytest.raises(ValueError):
        check_split_integrity(bundle)


# ---------------------------------------------------------------- replay


# ---------------------------------------------------------------- graph builder (B9)
def _flows():
    return pd.DataFrame({
        "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
        "IPV4_DST_ADDR": ["10.0.0.2", "10.0.0.3", "10.0.0.3"],
        "FLOW_START_MILLISECONDS": [0, 10, 20], "Attack": ["benign", "ddos", "ddos"],
        "IN_BYTES": [100.0, 300.0, 50.0], "PROTOCOL": [6, 17, 6],
    })


@pytest.mark.parametrize("mode,expected_src_only", [("legacy_half", 100.0), ("present_roles", 200.0)])
def test_role_mean(mode, expected_src_only):
    b = EnhancedGraphBuilder(["IN_BYTES"], ["IN_BYTES"], LabelSpace(CLASSES), temporal_dim=0,
                             structural_features=False, role_mean=mode)
    g = b.build_graph(_flows())
    # node 0 = 10.0.0.1, source only: mean IN_BYTES as source = 200
    assert g.x[0, 0].item() == pytest.approx(expected_src_only)
    # node 1 = 10.0.0.2, both roles: (50 + 100) / 2 in both modes
    assert g.x[1, 0].item() == pytest.approx(75.0)
    assert int(g.y_multi) == 1 and g.y_node.tolist() == [1, 1, 1]


# ---------------------------------------------------------------- block split integrity
def test_assign_blocks_disjoint_and_complete():
    from build_graphs import assign_blocks, rebalance_assignments
    rng = np.random.default_rng(0)
    rows = []
    for blk in range(60):
        cls = CLASSES[blk % 4]
        for k in range(20):
            rows.append({"FLOW_START_MILLISECONDS": blk * 60000 + int(rng.integers(0, 59999)),
                         "Attack": cls})
    df = pd.DataFrame(rows)
    parts, assignment = assign_blocks(df, block_ms=60000, seed=0)
    df["_block_id"] = (df["FLOW_START_MILLISECONDS"] - df["FLOW_START_MILLISECONDS"].min()) // 60000
    parts, assignment = rebalance_assignments(df, assignment, seed=0)
    blocks = {k: set(v["_block_id"]) for k, v in parts.items()}
    assert not (blocks["train"] & blocks["val"]) and not (blocks["train"] & blocks["stream"])
    assert not (blocks["val"] & blocks["stream"])
    assert sum(len(p) for p in parts.values()) == len(df)
    for p in parts.values():
        assert set(p["Attack"]) == set(CLASSES)
        assert p["FLOW_START_MILLISECONDS"].is_monotonic_increasing


def test_config_override_with_dict_is_attribute_accessible():
    cfg = load_config(ROOT / "configs/nf_ton_iot_v3.yaml",
                      {"features.normalization": {"mode": "zscore"}})
    assert cfg.features.normalization.mode == "zscore"


# ---------------------------------------------------------------- interleaved block order


# ---------------------------------------------------------------- chronological-warmup validation
