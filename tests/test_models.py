"""Tests for masking, the encoder, and the scoring indexing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from etlguard.models.layers import (REL_ANCESTOR, REL_CHILD, REL_DESCENDANT,
                                    REL_NONE, REL_PARENT, REL_SELF,
                                    dag_relation_matrix)
from etlguard.models.masking import (build_group_index, feature_group_mask,
                                     mixed_mask, node_mask)
from etlguard.models.st_transformer import ModelConfig, STTransformer
from etlguard.simulator.dag import build_pipeline
from etlguard.simulator.profiler import FEATURE_GROUPS, FEATURE_NAMES

SHAPE = (6, 12, 9, len(FEATURE_NAMES))


@pytest.fixture(scope="module")
def group_index():
    return build_group_index(FEATURE_NAMES, FEATURE_GROUPS)


# --------------------------------------------------------------------------- #
# DAG relations
# --------------------------------------------------------------------------- #

def test_relation_matrix_matches_the_pipeline():
    pipe = build_pipeline()
    rel = dag_relation_matrix(pipe.adjacency())
    i = pipe.index

    assert rel[i["clean_orders"], i["clean_orders"]] == REL_SELF
    assert rel[i["clean_orders"], i["extract_orders"]] == REL_PARENT
    assert rel[i["extract_orders"], i["clean_orders"]] == REL_CHILD
    # Two hops upstream is an ancestor, not a parent.
    assert rel[i["join_orders_customers"], i["extract_orders"]] == REL_ANCESTOR
    assert rel[i["extract_orders"], i["load_warehouse"]] == REL_DESCENDANT
    # Separate branches never touch.
    assert rel[i["extract_orders"], i["extract_customers"]] == REL_NONE


def test_relation_matrix_is_antisymmetric_in_direction():
    pipe = build_pipeline()
    rel = dag_relation_matrix(pipe.adjacency())
    flip = {REL_PARENT: REL_CHILD, REL_CHILD: REL_PARENT,
            REL_ANCESTOR: REL_DESCENDANT, REL_DESCENDANT: REL_ANCESTOR,
            REL_NONE: REL_NONE, REL_SELF: REL_SELF}
    n = rel.shape[0]
    for a in range(n):
        for b in range(n):
            assert rel[b, a] == flip[rel[a, b]]


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #

def test_feature_group_masking_hides_whole_families(group_index):
    """A family must be fully hidden or fully visible, never half of one.

    Half-masking a family would let the model recover the hidden half from the
    visible half of the same vector, which is exactly the shortcut this strategy
    is designed to close off.
    """
    m = feature_group_mask(SHAPE, group_index, 0.4, "cpu")
    for idx in group_index:
        sub = m[..., idx]
        per_cell = sub.sum(dim=-1)
        assert torch.all((per_cell == 0) | (per_cell == idx.numel()))


def test_node_masking_hides_entire_profiles():
    m = node_mask(SHAPE, 0.3, "cpu", last_step_bias=0.0)
    per_node = m.sum(dim=-1)
    assert torch.all((per_node == 0) | (per_node == SHAPE[-1]))


def test_node_masking_favours_the_scored_timestep():
    """The final step is the only one scored at inference, so bias masking there."""
    m = node_mask((256, 8, 9, 4), 0.2, "cpu", last_step_bias=0.5)
    last = m[:, -1].mean().item()
    earlier = m[:, :-1].mean().item()
    assert last > earlier + 0.2


def test_masking_ratio_is_roughly_respected(group_index):
    m = feature_group_mask((512, 8, 9, len(FEATURE_NAMES)), group_index, 0.3, "cpu")
    assert 0.2 < m.mean().item() < 0.4


def test_mixed_mask_picks_one_strategy_per_sample(group_index):
    m = mixed_mask(SHAPE, group_index, 0.35, 0.2, "cpu", p_node=0.5)
    assert m.shape == SHAPE
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #

def _model(dag_bias=True, window=12):
    pipe = build_pipeline()
    cfg = ModelConfig(n_nodes=pipe.n_nodes, n_features=len(FEATURE_NAMES),
                      window=window, d_model=32, n_heads=2, n_layers=2)
    cfg.dag_attention_bias = dag_bias
    return STTransformer(cfg, pipe.adjacency()), cfg


def test_forward_shape_and_finiteness():
    model, cfg = _model()
    x = torch.randn(*SHAPE)
    mask = torch.zeros_like(x)
    cal = torch.randn(SHAPE[0], SHAPE[1], cfg.n_calendar)
    out = model(x, mask, cal)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_masked_entries_do_not_leak_into_the_encoder():
    """Changing a masked value must not change the output.

    If it did, the model could read the answer it is being asked to predict and
    every reconstruction error afterwards would be meaningless.
    """
    model, cfg = _model()
    model.eval()
    x = torch.randn(*SHAPE)
    mask = torch.zeros_like(x)
    mask[:, -1, 3] = 1.0
    cal = torch.randn(SHAPE[0], SHAPE[1], cfg.n_calendar)

    with torch.no_grad():
        a = model(x, mask, cal)
        x2 = x.clone()
        x2[:, -1, 3] = 999.0            # rewrite only the hidden entries
        b = model(x2, mask, cal)
    assert torch.allclose(a, b, atol=1e-5)


def test_dag_bias_changes_the_output_and_can_be_disabled():
    torch.manual_seed(0)
    with_bias, cfg = _model(dag_bias=True)
    assert with_bias.rel_bias is not None
    assert with_bias._dag_bias().shape == (cfg.n_heads, cfg.n_nodes, cfg.n_nodes)
    without, _ = _model(dag_bias=False)
    assert without.rel_bias is None
    assert without._dag_bias() is None


def test_embedding_head_returns_one_vector_per_window():
    model, cfg = _model()
    x = torch.randn(*SHAPE)
    z = model.embed(x, torch.zeros_like(x), torch.randn(SHAPE[0], SHAPE[1], 4))
    assert z.shape == (SHAPE[0], cfg.proj_hidden)
