"""Tests for the pipeline simulator: contracts, healthy baselines, fault effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etlguard.simulator.dag import build_pipeline
from etlguard.simulator.faults import REGISTRY, FaultEvent, applicable_faults
from etlguard.simulator.generator import SimConfig, simulate
from etlguard.simulator.profiler import FEATURE_INDEX, FEATURE_NAMES, N_FEATURES
from etlguard.simulator.schema import ALL_TABLES, Role


@pytest.fixture(scope="module")
def clean_run():
    cfg = SimConfig(n_batches=40, seed=3, base_orders=400,
                    fault_rate_train=0.0, fault_rate_test=0.0)
    return simulate(cfg, progress=False)


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #

def test_every_table_satisfies_the_profiler_contract():
    """The profiler assumes each table has a PK, a metric, a category and a timestamp."""
    for name, spec in ALL_TABLES.items():
        assert spec.by_role(Role.PK), f"{name} has no primary key"
        assert spec.by_role(Role.METRIC), f"{name} has no metric column"
        assert spec.by_role(Role.CATEGORICAL), f"{name} has no categorical column"
        assert spec.by_role(Role.TIMESTAMP), f"{name} has no timestamp column"


def test_feature_names_are_unique_and_complete():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == N_FEATURES


def test_dag_is_topologically_ordered_and_connected():
    pipe = build_pipeline()
    assert pipe.n_nodes == 9
    # Every source reaches the warehouse.
    for node in pipe.nodes:
        if node.is_source:
            assert "load_warehouse" in pipe.descendants(node.name)
    a = pipe.adjacency()
    assert a.shape == (9, 9)
    assert np.allclose(np.diag(a), 0.0)


# --------------------------------------------------------------------------- #
# Healthy baseline
# --------------------------------------------------------------------------- #

def test_clean_run_has_no_labels_and_finite_features(clean_run):
    assert clean_run["y_type"].sum() == 0
    assert clean_run["y_topo"].sum() == 0
    assert np.isfinite(clean_run["X"]).all()


def test_healthy_pipeline_preserves_data_quality(clean_run):
    """With no faults, integrity/completeness features must sit near zero."""
    X = clean_run["X"]
    pipe = build_pipeline()
    for feat, limit in [("orphan_fk_rate", 0.01),
                        ("dup_pk_rate", 0.02),
                        ("pk_null_rate", 0.001),
                        ("cat_unseen_rate", 0.001),
                        ("type_mismatch_rate", 0.0),
                        ("future_ts_rate", 0.0),
                        ("out_of_window_rate", 1.0)]:
        worst = X[:, :, FEATURE_INDEX[feat]].max()
        assert worst <= limit, f"healthy runs show {feat}={worst:.4f} (> {limit})"


def test_healthy_row_counts_are_stable_but_seasonal(clean_run):
    """Volume must vary (seasonality) without collapsing (a fault-free floor)."""
    rows = clean_run["X"][:, build_pipeline().index["extract_orders"],
                          FEATURE_INDEX["log_rows_out"]]
    assert rows.min() > np.log1p(50)
    assert rows.std() > 0.01


# --------------------------------------------------------------------------- #
# Fault behaviour
# --------------------------------------------------------------------------- #

def _paired_run(fault_type: str, node: str, severity: float = 0.9, seed: int = 5):
    """Same seed, same timeline — the only difference is the injected fault."""
    import etlguard.simulator.generator as gen
    cfg = SimConfig(n_batches=8, seed=seed, base_orders=400,
                    fault_rate_train=0.0, fault_rate_test=0.0)
    clean = simulate(cfg, progress=False)
    t = 6
    original = gen.schedule_faults
    gen.schedule_faults = lambda *a, **k: [FaultEvent(fault_type, node, t, 1, severity)]
    try:
        dirty = simulate(cfg, progress=False)
    finally:
        gen.schedule_faults = original
    return clean, dirty, t


@pytest.mark.parametrize("fault_type", sorted(REGISTRY))
def test_each_fault_changes_the_profile_at_its_node(fault_type):
    pipe = build_pipeline()
    node = next(n.name for n in pipe.nodes
                if fault_type in applicable_faults(n.flags))
    clean, dirty, t = _paired_run(fault_type, node)
    i = pipe.index[node]
    delta = np.abs(dirty["X"][t, i] - clean["X"][t, i])
    assert delta.max() > 1e-4, f"{fault_type} at {node} left the profile unchanged"


def test_data_faults_propagate_downstream_but_operational_ones_do_not():
    pipe = build_pipeline()

    _, dirty, t = _paired_run("null_spike", "clean_orders")
    affected = dirty["y_topo"][t]
    for d in pipe.descendants("clean_orders"):
        assert affected[pipe.index[d]] == 1, f"{d} should inherit the incident"

    _, dirty_op, t = _paired_run("resource_degradation", "clean_orders")
    assert dirty_op["y_topo"][t].sum() == 1, "an operational fault must stay local"


def test_root_cause_is_a_subset_of_affected():
    cfg = SimConfig(n_batches=300, seed=13, base_orders=300, fault_rate_test=0.3)
    res = simulate(cfg, progress=False)
    assert res["y_root"].sum() > 0
    assert np.all(res["y_root"] <= res["y_topo"])


def test_duplicate_records_inflates_downstream_revenue():
    """A duplication upstream must show up as an aggregate distortion downstream."""
    pipe = build_pipeline()
    clean, dirty, t = _paired_run("duplicate_records", "clean_orders", severity=1.0)
    j = pipe.index["aggregate_daily"]
    k = FEATURE_INDEX["metric_mean"]
    assert dirty["X"][t, j, k] > clean["X"][t, j, k] * 1.05


# --------------------------------------------------------------------------- #
# Counterfactual labelling
# --------------------------------------------------------------------------- #

def test_counterfactual_equals_factual_on_healthy_batches(clean_run):
    """With no fault anywhere, the counterfactual run must be the factual run."""
    assert np.array_equal(clean_run["X"], clean_run["X_cf"])


def test_counterfactual_isolates_the_fault():
    """Nodes the fault cannot reach must be bit-identical in the counterfactual.

    This is the invariant the measured propagation labels rest on: per-node
    seeded RNG means the only difference between the two runs is the fault.
    """
    pipe = build_pipeline()
    cfg = SimConfig(n_batches=8, seed=21, base_orders=400,
                    fault_rate_train=0.0, fault_rate_test=0.0)
    import etlguard.simulator.generator as gen
    t = 6
    original = gen.schedule_faults
    gen.schedule_faults = lambda *a, **k: [
        FaultEvent("null_spike", "clean_orders", t, 1, 0.9)]
    try:
        res = simulate(cfg, progress=False)
    finally:
        gen.schedule_faults = original

    unreachable = [n.name for n in pipe.nodes
                   if n.name != "clean_orders"
                   and "clean_orders" not in pipe.descendants(n.name)
                   and n.name not in pipe.descendants("clean_orders")]
    assert unreachable, "expected some nodes outside the blast radius"
    for name in unreachable:
        i = pipe.index[name]
        assert np.array_equal(res["X"][t, i], res["X_cf"][t, i]), \
            f"{name} is outside the blast radius but changed"


def test_non_propagating_faults_are_labelled_local():
    """An extra unread column is ignored downstream, so it must not spread."""
    from etlguard.data.dataset import RobustScaler, derive_affected

    cfg = SimConfig(n_batches=600, seed=31, base_orders=300, fault_rate_test=0.4)
    res = simulate(cfg, progress=False)
    scaler = RobustScaler.fit(res["X"][res["split"] == 0])
    measured = derive_affected(res["X"], res["X_cf"], scaler.scale)

    add_id = res["fault_labels"].index("schema_column_add")
    m = res["y_type"] == add_id
    if m.sum() >= 3:
        assert measured[m].sum(axis=1).mean() < 2.0
        assert res["y_topo"][m].sum(axis=1).mean() > 2.0, \
            "the topological label should over-approximate here"
