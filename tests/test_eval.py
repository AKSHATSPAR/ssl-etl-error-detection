"""Tests for the scoring configurations and the evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from etlguard.eval.metrics import (best_f1, detection_delay, f1_at_budget,
                                   localization_metrics, point_adjusted_f1)
from etlguard.eval.scoring import ScoreBundle, score_loader
from etlguard.models.st_transformer import ModelConfig, STTransformer
from etlguard.simulator.dag import build_pipeline
from etlguard.simulator.profiler import FEATURE_NAMES

F = len(FEATURE_NAMES)
W, N = 8, 9


class _Fake(Dataset):
    """A handful of windows, enough to exercise the scoring plumbing."""

    def __init__(self, m=5):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(m, W, N, F, generator=g)
        self.m = m

    def __len__(self):
        return self.m

    def __getitem__(self, i):
        return {"x": self.x[i], "cal": torch.zeros(W, 4),
                "y_anom": torch.zeros(N), "y_root": torch.zeros(N),
                "y_type": torch.tensor(0), "severity": torch.tensor(0.0),
                "t": torch.tensor(i)}


def _model():
    pipe = build_pipeline()
    cfg = ModelConfig(n_nodes=N, n_features=F, window=W, d_model=32,
                      n_heads=2, n_layers=2, dropout=0.0)
    return STTransformer(cfg, pipe.adjacency())


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def test_score_loader_shapes_and_finiteness():
    b = score_loader(_model(), DataLoader(_Fake(), batch_size=2), "cpu", N)
    assert b.e_hist.shape == (5, N)
    assert b.e_ctx.shape == (5, N)
    assert b.feat_err.shape == (5, N, F)
    assert np.isfinite(b.e_hist).all() and np.isfinite(b.e_ctx).all()


def test_context_scoring_hides_exactly_one_node():
    """``e_ctx[i]`` must come from the pass where node *i* alone was hidden.

    The batched implementation replicates each window N times and relies on a
    gather to pull the right row back out. An off-by-one there would silently
    read another node's error and every localization number would be wrong, so
    the batched result is checked against a plain per-node loop.
    """
    torch.manual_seed(0)
    model = _model().eval()
    data = _Fake(3)
    batched = score_loader(model, DataLoader(data, batch_size=3), "cpu", N)

    with torch.no_grad():
        for w in range(len(data)):
            x = data[w]["x"].unsqueeze(0)
            cal = torch.zeros(1, W, 4)
            for n in range(N):
                m = torch.zeros_like(x)
                m[:, -1, n] = 1.0
                pred = model(x, m, cal)
                want = ((pred[0, -1, n] - x[0, -1, n]) ** 2).mean().item()
                assert batched.e_ctx[w, n] == pytest.approx(want, abs=1e-5)


def test_history_scoring_hides_the_whole_final_run():
    torch.manual_seed(0)
    model = _model().eval()
    data = _Fake(2)
    b = score_loader(model, DataLoader(data, batch_size=2), "cpu", N)
    with torch.no_grad():
        x = torch.stack([data[i]["x"] for i in range(2)])
        m = torch.zeros_like(x)
        m[:, -1] = 1.0
        pred = model(x, m, torch.zeros(2, W, 4))
        want = ((pred[:, -1] - x[:, -1]) ** 2).mean(dim=-1).numpy()
    assert np.allclose(b.e_hist, want, atol=1e-5)


def test_every_pooling_rule_is_implemented():
    b = score_loader(_model(), DataLoader(_Fake(), batch_size=5), "cpu", N)
    for rule in ScoreBundle.DETECTION_RULES:
        s = b.detection_score(rule)
        assert s.shape == (5,) and np.isfinite(s).all()
    with pytest.raises(ValueError):
        b.detection_score("nonsense")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_best_f1_is_perfect_on_a_separable_problem():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    assert best_f1(y, s)["best_f1"] == pytest.approx(1.0)


def test_best_f1_degrades_when_the_score_is_reversed():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.9, 0.95, 0.8, 0.1, 0.2])
    assert best_f1(y, s)["best_f1"] < 0.75


def test_f1_at_budget_respects_the_alert_limit():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype(int)
    s = rng.random(1000)
    out = f1_at_budget(y, s, 0.01)
    # A 1% budget over 1000 runs cannot recall more than 10 of ~100 incidents.
    assert out["recall@0.01"] <= 0.12


def test_point_adjustment_inflates_a_weak_detector():
    """The reason it is not the headline metric, demonstrated rather than asserted."""
    # A realistic base rate matters here: at a high one, "flag everything"
    # already scores respectably and muddies the comparison.
    y = np.zeros(2000, dtype=int)
    y[100:150] = 1                      # one long incident, 2.5% of runs
    s = np.zeros(2000)
    s[120] = 1.0                        # a single lucky hit inside it
    plain = best_f1(y, s)["best_f1"]
    adjusted = point_adjusted_f1(y, s)
    assert plain < 0.1
    assert adjusted == pytest.approx(1.0)


def test_localization_is_perfect_when_the_origin_ranks_first():
    y_root = np.zeros((4, 5), dtype=int)
    y_root[np.arange(4), [0, 1, 2, 3]] = 1
    r = np.zeros((4, 5))
    r[np.arange(4), [0, 1, 2, 3]] = 10.0
    out = localization_metrics(y_root, r)
    assert out["hit@1"] == pytest.approx(1.0)
    assert out["mrr"] == pytest.approx(1.0)


def test_localization_hit_at_3_is_looser_than_hit_at_1():
    y_root = np.zeros((3, 5), dtype=int)
    y_root[:, 4] = 1
    r = np.tile(np.array([5.0, 4.0, 3.0, 2.0, 4.5]), (3, 1))   # truth ranks 2nd
    out = localization_metrics(y_root, r)
    assert out["hit@1"] == 0.0
    assert out["hit@3"] == 1.0
    assert out["mrr"] == pytest.approx(0.5)


def test_localization_ignores_healthy_runs():
    y_root = np.zeros((5, 4), dtype=int)
    y_root[2, 1] = 1
    r = np.random.default_rng(0).random((5, 4))
    assert localization_metrics(y_root, r)["n_incidents"] == 1


def test_detection_delay_counts_from_incident_start():
    y_type = np.zeros(20, dtype=int)
    y_type[5:10] = 3
    s = np.zeros(20)
    s[7] = 1.0                          # fires two runs into the incident
    out = detection_delay(y_type, s, 0.5)
    assert out["mean_detection_delay"] == pytest.approx(2.0)
    assert out["incidents"] == 1 and out["incidents_missed"] == 0


def test_detection_delay_reports_a_miss():
    y_type = np.zeros(20, dtype=int)
    y_type[5:10] = 3
    out = detection_delay(y_type, np.zeros(20), 0.5)
    assert out["incidents_missed"] == 1
    assert out["incident_recall"] == 0.0
