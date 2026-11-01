"""Architectural / loss ablation switches of TGN-JointIDS.

Each test checks that an ablation changes only its intended component.
Run:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tgn_jointids.config import Config  # noqa: E402
from tgn_jointids.losses import (class_weights_from_config, compute_joint_loss,  # noqa: E402
                                 focal_loss_standard)
from tgn_jointids.models.tgn_jointids import TGNJointIDS  # noqa: E402
from tgn_jointids.data.features import LabelSpace  # noqa: E402

C = 4


def model(seed=0, **kw):
    torch.manual_seed(seed)
    return TGNJointIDS(in_dim=5, hidden_dim=16, num_classes=C, memory_dim=8, num_layers=2,
                       num_heads=2, dropout=0.0, edge_dim=3, **kw).eval()


def batch(seed=0, n_graphs=3):
    g = torch.Generator().manual_seed(seed)
    gs = []
    for k in range(n_graphs):
        gs.append(Data(x=torch.randn(6, 5, generator=g),
                       edge_index=torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 5]]),
                       edge_attr=torch.randn(6, 3, generator=g), y_multi=torch.tensor([k % C]),
                       y_bin=torch.tensor([int(k % C != 0)]),
                       y_node=torch.randint(0, 2, (6,), generator=g),
                       node_ids=torch.arange(6), num_nodes=6))
    return Batch.from_data_list(gs)


# ---------------------------------------------------------------- initialisation
@pytest.mark.parametrize("kw", [{"pooling": "mean"}, {"use_edge_attr": False}])
def test_ablated_models_have_identical_initialisation(kw):
    ref, abl = model().state_dict(), model(**kw).state_dict()
    assert ref.keys() == abl.keys()
    assert all(torch.equal(ref[k], abl[k]) for k in ref)


def test_defaults_reproduce_reference_outputs():
    d = batch()
    a = model()(d)
    b = model(pooling="sagpool", use_edge_attr=True)(d)
    assert all(torch.equal(x, y) for x, y in zip(a[:4], b[:4]))


# ---------------------------------------------------------------- A3: pooling
def test_mean_pooling_replaces_only_the_readout():
    d = batch()
    m_ref, m_mean = model(), model(pooling="mean")
    node_ref, g_ref, _ = m_ref.encode(d)
    node_mean, g_mean, _ = m_mean.encode(d)
    assert torch.equal(node_ref, node_mean)                       # encoder untouched
    expected = m_mean.pool_proj(global_mean_pool(node_mean, d.batch))
    assert torch.allclose(g_mean, expected)
    assert g_mean.shape == g_ref.shape == (3, 16)                 # compatible dimensions
    assert not torch.allclose(g_mean, g_ref)


def test_mean_pooling_leaves_sagpool_parameters_without_gradient():
    m = model(pooling="mean").train()
    bl, ml, nl, _, _ = m(batch())
    (bl.sum() + ml.sum() + nl.sum()).backward()
    assert all(p.grad is None for p in m.sagpool.parameters())
    assert all(conv.lin.weight.grad is not None for conv in m.convs)   # encoder still trained


# ---------------------------------------------------------------- A3: edge attributes
def test_no_edge_attr_ignores_attributes_but_keeps_topology():
    d = batch()
    m = model(use_edge_attr=False)
    base = m(d)[1]
    d2 = d.clone()
    d2.edge_attr = d2.edge_attr * 100 + 7                          # attributes changed
    assert torch.allclose(base, m(d2)[1])
    d3 = d.clone()
    d3.edge_index = d3.edge_index[:, :-2]                           # topology changed
    assert not torch.allclose(base, m(d3)[1])
    ref = model()
    assert not torch.allclose(ref(d)[1], ref(d2)[1])               # reference does use them


def test_no_edge_attr_leaves_edge_projection_without_gradient():
    m = model(use_edge_attr=False).train()
    bl, ml, nl, _, _ = m(batch())
    (bl.sum() + ml.sum() + nl.sum()).backward()
    for conv in m.convs:
        assert conv.lin_edge.weight.grad is None and conv.att_edge.grad is None
        assert conv.lin.weight.grad is not None


def test_invalid_pooling_rejected():
    with pytest.raises(ValueError):
        model(pooling="max")


# ---------------------------------------------------------------- A1: task coefficients
def _losses(lb, ln, d, m):
    bl, ml, nl, _, _ = m(d)
    return compute_joint_loss(bl, ml, nl, d.y_bin, d.y_multi, d.y_node, torch.ones(2),
                              torch.ones(C), lambda_bin=lb, lambda_multi=0.7, lambda_node=ln,
                              focal_impl="standard")


@pytest.mark.parametrize("lb,ln", [(0.0, 0.2), (1.0, 0.0), (0.0, 0.0)])
def test_zero_coefficient_removes_only_that_task_gradient(lb, ln):
    d = batch()
    m = model().train()
    total, b, mu, n = _losses(lb, ln, d, m)
    full = 1.0 * b + 0.7 * mu + 0.2 * n
    assert torch.allclose(total, lb * b + 0.7 * mu + ln * n)
    total.backward()
    bin_grad = sum(p.grad.abs().sum() for p in m.binary_head.parameters() if p.grad is not None)
    node_grad = sum(p.grad.abs().sum() for p in m.node_aux_head.parameters() if p.grad is not None)
    multi_grad = sum(p.grad.abs().sum() for p in m.multi_head.parameters() if p.grad is not None)
    assert (bin_grad == 0) == (lb == 0)
    assert (node_grad == 0) == (ln == 0)
    assert multi_grad > 0                                          # multiclass always trained
    assert not torch.allclose(total, full) or (lb == 1.0 and ln == 0.2)


# ---------------------------------------------------------------- A2: ENS / focal
def _cfg(weighting):
    return Config({"ens_beta_binary": 0.999, "ens_source": "train_graphs", "ens_beta": 0.9999,
                   "ens_cap_ratio": 30.0, "multi_weighting": weighting})


def test_multi_weighting_none_changes_only_multiclass_weights():
    ls = LabelSpace(["benign", "a", "b", "c"])
    gs = [Data(y_multi=torch.tensor([c]), y_bin=torch.tensor([int(c > 0)]))
          for c, n in ((0, 500), (1, 100), (2, 20), (3, 5)) for _ in range(n)]
    bw_e, mw_e, _ = class_weights_from_config(_cfg("ens"), ls, gs, "cpu")
    bw_n, mw_n, _ = class_weights_from_config(_cfg("none"), ls, gs, "cpu")
    assert torch.equal(bw_e, bw_n)                                  # binary weights unchanged
    assert torch.equal(mw_n, torch.ones(4)) and not torch.equal(mw_e, torch.ones(4))
    with pytest.raises(ValueError):
        class_weights_from_config(_cfg("bogus"), ls, gs, "cpu")


def test_gamma_zero_is_weighted_smoothed_ce():
    torch.manual_seed(0)
    z, y = torch.randn(8, C), torch.randint(0, C, (8,))
    w = torch.tensor([0.5, 1.0, 2.0, 3.0])
    ce = F.cross_entropy(z, y, label_smoothing=0.05, reduction="none")
    assert torch.allclose(focal_loss_standard(z, y, w, gamma=0.0, label_smoothing=0.05),
                          (w[y] * ce).mean())
    assert torch.allclose(focal_loss_standard(z, y, None, gamma=0.0, label_smoothing=0.05),
                          ce.mean())


def test_protocol_type_is_not_consumed_by_the_encoder():
    """Documents a code-manuscript discrepancy: edge_type is built but unused."""
    d = batch()
    d.edge_type = torch.zeros(d.edge_index.size(1), dtype=torch.long)
    m = model()
    a = m(d)[1]
    d.edge_type = torch.full_like(d.edge_type, 2)
    assert torch.equal(a, m(d)[1])
